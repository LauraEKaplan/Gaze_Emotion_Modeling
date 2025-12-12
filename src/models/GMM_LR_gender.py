import numpy as np
import pandas as pd
import scipy
from collections import defaultdict
import scipy.stats as stats
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import KFold
from map_estimator import run_map_for_emotion
from map_estimator import bayesian_linear_regression
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler

CLUSTER_CUTOFF = 0.5


# get Phi and Y for both train and test
def get_train_test_data(emotion, degree = 2, interact = False) :
    # degree: degree of polynomial features
    # interact : add interacting terms (x1x2)

    train = pd.read_csv("../../data/processed/gender/train9001.csv")
    test = pd.read_csv("../../data/processed/gender/test9001.csv")

    # Get x and y (automatically detect based on names of features)
    X_train = train.filter(regex='^(Blink_|Velo_)').values
    Y_train = train[emotion].values

    X_test = test.filter(regex='^(Blink_|Velo_)').values
    Y_test = test[emotion].values

    n_samples_train, n_features = X_train.shape
    n_samples_test, n_features = X_test.shape


    # Initialize an empty list with bias
    X_poly_train = np.ones((n_samples_train, 1))
    X_poly_test = np.ones((n_samples_test, 1))
    
    # Add original and polynomial features
    for d in range(1, degree + 1):
        for feature in range(n_features):
            X_poly_train = np.hstack((X_poly_train, np.power(X_train[:, feature:feature + 1], d)))
            X_poly_test = np.hstack((X_poly_test, np.power(X_test[:, feature:feature + 1], d)))

    
    # Add interaction terms
    if interact :
        for i in range(n_features):
            for j in range(i + 1, n_features):
                X_poly_train = np.hstack((X_poly_train, (X_train[:, i:i + 1] * X_train[:, j:j + 1])))
                X_poly_test = np.hstack((X_poly_test, (X_test[:, i:i + 1] * X_test[:, j:j + 1])))


    return X_poly_train, Y_train, X_poly_test, Y_test



def generate_initial_parameters(x_ND, m, K):
    
    prng = np.random.RandomState(10)
    # Make every component equally probable
    log_pi_K = np.log(1.0 / K * np.ones(K))

    # Select each mean vector to be centered on a randomly chosen data point
    N, D = x_ND.shape
    if N < K:
        x_ND = np.vstack([x_ND, np.random.randn(K, D)])
        N = x_ND.shape[0]
    chosen_rows_K = prng.choice(np.arange(N), K, replace=False)
    mu_KD = x_ND[chosen_rows_K].copy()

    # Select every variance initially as uniform between (mode, 2.0 * mode)
    # So that variances are not too small under prior knowledge
    stddev_KD = np.sqrt(m) * (
        1 + prng.rand(K, D))

    return log_pi_K, mu_KD, stddev_KD


# Calculates r_NK
def e_step(x_ND, K, mu_KD, stddev_KD, log_pi_K) :
    N = x_ND.shape[0]
    r_NK = np.zeros((N, K))

    # Get denominator
    log_denom_NK = np.zeros((N, K))
    for k in range(K):
        log_denom_NK[:, k] = np.sum(stats.norm.logpdf(x_ND, loc = mu_KD[k], scale = stddev_KD[k]), axis = 1) + log_pi_K[k]

    log_denom_N = logsumexp(log_denom_NK, axis = 1)

    for k in range(K):
        #  update k-th col. of r_NK to optimal value given GMM params
        log_numerator_N = log_denom_NK[:,k]
        log_r_N = log_numerator_N - log_denom_N
        r_NK[:, k] = np.exp(log_r_N)
    assert np.allclose(np.sum(r_NK, axis=1), 1.0)

    return r_NK

# --- NEW: fixed-label E-step ---
def e_step_fixed_labels(cluster_labels, K):
    N = len(cluster_labels)
    r_NK = np.zeros((N, K))
    for n in range(N):
        r_NK[n, cluster_labels[n]] = 1.0
    return r_NK
# --- end new ---


def m_step(x_ND, K, r_NK, m, s) :
    N, D = x_ND.shape

    # First pi_K
    pi_K = np.sum(r_NK, axis=0) / N
    assert np.allclose(np.sum(pi_K), 1.0)

    # Avoid any zeros    
    log_pi_k = np.log(1e-13 + pi_K)

    # Next mu_KD
    mu_KD = np.zeros((K, D))   
    for k in range(K) :
        mu_KD[k,:] = np.sum(r_NK[:, k][:, np.newaxis] * x_ND, axis = 0) / np.sum(r_NK[:, k])

    # Finally, stddev_KD

    #  Calc weighted sum of squared differences for each k,d pair
    ssd_KD = np.zeros((K, D)) 
    for k in range(K):
        ssd_KD[k, :] = np.sum(r_NK[:, k][:, np.newaxis] * np.power(x_ND - mu_KD[k, :], 2), axis=0)

    #  compute sum of r_NK for each k
    N_K1 = np.sum(r_NK, axis=0)[:, np.newaxis]

    var_KD = (ssd_KD + 1.0/(s*m)) / (N_K1 + 1.0/(s*m*m))
    stddev_KD = np.sqrt(var_KD)

    return log_pi_k, mu_KD, stddev_KD

def calc_EM_loss(K, r_NK, x_ND, log_pi_K, mu_KD, stddev_KD, m, s) :
    log_prior = np.sum(np.dot(r_NK, log_pi_K))
    log_lik = 0.0
    for k in range(K):
        log_lik_k_N = np.sum(stats.norm.logpdf(x_ND, mu_KD[k], stddev_KD[k]), axis=1)
        log_lik += np.inner(r_NK[:,k], log_lik_k_N)
    entropy = -1.0 * np.sum(r_NK * np.log(r_NK + 1e-100))
    penalty_stddev = calc_penalty_stddev(stddev_KD, m, s)
    return -1.0 * (log_prior + log_lik + entropy) + penalty_stddev


def calc_penalty_stddev(stddev_KD, m, s):

    coef_A = 1.0/(s * m * m)
    coef_B = 1.0/(s * m)
    return (
        np.sum(coef_A * np.log(stddev_KD) 
        + 0.5 * coef_B / np.square(stddev_KD)))


def score_raw(x_ND, log_pi_K, mu_KD, stddev_KD) :
    N, D = x_ND.shape
    assert D == mu_KD.shape[1]
    assert D == stddev_KD.shape[1]
    K = mu_KD.shape[0]

    ##  compute joint probability of x and z given parameters
    logp_z_and_x_NK = np.zeros((N, K))
    for k in range(K):
        #  fill in column k of array logp_z_and_x_NK
        # so that entry [n,k] = log p(x_n, z_n = k | parameters)
        logp_z_and_x_NK[:, k] = np.sum(stats.norm.logpdf(x_ND, loc = mu_KD[k], scale = stddev_KD[k]), axis = 1) + log_pi_K[k]

    #  call logsumexp, with appropriate axis kwarg
    # want an array where n-th entry = log p(x_n | parameters)
    logp_x_N = logsumexp(logp_z_and_x_NK, axis = 1)

    # Return total likelihood, summed over entire dataset
    neg_log_lik = -1.0 * np.sum(logp_x_N)
    return -1.0 * neg_log_lik


def GMM_fit(x_ND, m, s, K, max_iter, fixed_labels=None) :
    """
    If fixed_labels is provided (length N, ints in 0..K-1), responsibilities are fixed
    and EM will only run M-steps to estimate per-cluster stats.
    """
    N, D = x_ND.shape
    log_pi_K, mu_KD, stddev_KD = generate_initial_parameters(x_ND, m, K)
    score_train_history = []
    loss_train_history = []

    # initial responsibilities
    if fixed_labels is not None:
        r_NK = e_step_fixed_labels(fixed_labels, K)
    else:
        r_NK = None

    # Iterate 
    for i in range(max_iter) :
        if fixed_labels is None:
            r_NK = e_step(x_ND, K, mu_KD, stddev_KD, log_pi_K)
        else:
            # skip E-step, keep responsibilities fixed by gender
            pass

        log_pi_K, mu_KD, stddev_KD = m_step(x_ND, K, r_NK, m, s)
        # Normalize score by N (since it changes between validation and test) as well as by D
        score_train_history.append(score_raw(x_ND, log_pi_K, mu_KD, stddev_KD) / (N * D))
        loss_train_history.append(calc_EM_loss(K, r_NK, x_ND, log_pi_K, mu_KD, stddev_KD, m, s))

    return r_NK, log_pi_K, mu_KD, stddev_KD, score_train_history, loss_train_history

def responsibilities_for_x(x, log_pi_K, mu_X, std_X):
    K = len(log_pi_K)
    log_rk = np.zeros(K)
    for k in range(K):
        # Likelihood of x under cluster k
        log_rk[k] = np.sum(norm.logpdf(x, mu_X[k], std_X[k])) + log_pi_K[k]
    log_rk -= logsumexp(log_rk)
    return np.exp(log_rk)

##########################################################################################

def run_kfcv(x_ND, xc_ND, Y, cluster_labels, m, s, K, cv_K, max_iter, alpha, beta):
    """
    cluster_labels: array length N (train set), integers in 0..K-1 indicating gender cluster for each train sample.
    """
    N, D = x_ND.shape
    kf = KFold(n_splits=cv_K, shuffle=True, random_state=42)

    loocv_scores = []

    for train_index, val_index in kf.split(x_ND):
        X_train, X_valid = x_ND[train_index], x_ND[val_index]

        # This is the clustered (scaled) version
        Xc_train, Xc_valid = xc_ND[train_index], xc_ND[val_index]

        Y_train, Y_valid = Y[train_index], Y[val_index]

        # Fixed labels for this fold (gender IDs)
        fixed_labels_train = cluster_labels[train_index]
        fixed_labels_valid = cluster_labels[val_index]

        # Fit model on Xc_train using fixed gender labels
        r_NK, log_pi_K, mu_KD, stddev_KD, score_train_history, loss_train_history = \
            GMM_fit(Xc_train, m, s, K, max_iter, fixed_labels=fixed_labels_train)

        weight_vecs = []
    
        for k in range(K):
            # Get the data points assigned to cluster k in the training fold
            cluster_data = X_train[fixed_labels_train == k]
            cluster_Y = Y_train[fixed_labels_train == k]

            # If a cluster has no points in this fold, handle it (skip or regularize).
            if len(cluster_Y) == 0:
                # fallback: zero weights or small-random vector to avoid errors
                # Here we make zeros with correct dimensionality:
                weight_vecs.append(np.zeros(X_train.shape[1]))
                continue
            
            # Train a regression model for this cluster
            weight_vec = bayesian_linear_regression(cluster_data, cluster_Y, alpha, beta)
            weight_vecs.append(weight_vec)

        y_preds = []
        
        # For validation, use their known gender label to select the regressor
        for i_idx, onec_X in enumerate(Xc_valid):
            one_X = X_valid[i_idx]
            cluster = fixed_labels_valid[i_idx]  # use true gender for validation
            # ensure cluster index is valid
            if cluster < 0 or cluster >= K:
                # fallback to mixing based on GMM if invalid (shouldn't happen)
                r_NK_predict = responsibilities_for_x(onec_X, log_pi_K, mu_KD, stddev_KD)
                y_pred = sum(r_NK_predict[k] * np.dot(one_X, weight_vecs[k]) for k in range(K))
            else:
                y_pred = np.dot(one_X, weight_vecs[cluster])  # single-cluster prediction

            # clamp predictions
            if y_pred > 1 :
                y_pred = 1
            if y_pred < -1 :
                y_pred = -1
            y_preds.append(y_pred)

        loocv_scores.append(calc_score(Y_valid, np.array(y_preds), beta))

    return sum(loocv_scores)/cv_K


def tune_hyperparameters(x_ND, xc_ND, Y, cluster_labels, cv_K, max_iter, alpha, beta, m_list = [1,2], s_list = [3,5,10,25], k_list = [2,3,5]) :
    best_m = -1
    best_s = -1
    best_k = -1
    best_score = -1
    for m in m_list :
        for s in s_list :
            for k in k_list :
                # k should match number of genders; we normally will call with k_list=[K]
                score = run_kfcv(x_ND, xc_ND, Y, cluster_labels, m, s, k, cv_K, max_iter, alpha, beta)
                if score > best_score :
                    best_score = score
                    best_m = m
                    best_s = s
                    best_k = k

    return best_m, best_s, best_k, best_score

def train_and_eval(X_train, Xc_train, Y_train, X_test, Xc_test, Y_test, cluster_labels_train, cluster_labels_test, m, s, K, max_iter, alpha, beta):
    N, D = X_train.shape

    # Fit model on joint X (training set) using fixed gender labels
    r_NK, log_pi_K, mu_KD, stddev_KD, score_train_history, loss_train_history = GMM_fit(Xc_train, m, s, K, max_iter, fixed_labels=cluster_labels_train)


    weight_vecs = []
    
    for k in range(K):
        # Get the data points assigned to cluster k
        cluster_data = X_train[cluster_labels_train == k]
        cluster_Y = Y_train[cluster_labels_train == k]
        
        if len(cluster_Y) == 0:
            # fallback zero weight vector
            weight_vecs.append(np.zeros(X_train.shape[1]))
            continue

        # Train a regression model for this cluster
        weight_vec = bayesian_linear_regression(cluster_data, cluster_Y, alpha, beta)
        weight_vecs.append(weight_vec)

    y_preds = []
    # Use the test sample's known gender to pick the regressor
    for i, onec_X in enumerate(Xc_test) :
        one_X = X_test[i]
        cluster = cluster_labels_test[i]
        if cluster < 0 or cluster >= K:
            # fallback to mixture if cluster invalid
            r_NK_predict = responsibilities_for_x(onec_X, log_pi_K, mu_KD, stddev_KD)
            y_pred = sum(r_NK_predict[k] * np.dot(one_X, weight_vecs[k]) for k in range(K))
        else:
            y_pred = np.dot(one_X, weight_vecs[cluster])

        if y_pred > 1 :
            y_pred = 1
        if y_pred < -1 :
            y_pred = -1

        y_preds.append(y_pred)

    score = calc_score(Y_test, np.array(y_preds), beta)
    print(f"Score: {score}")


    # plt.plot(np.array(score_train_history))
    # plt.show() 
    return score


def calc_score(y_true, y_pred, beta = 10) :

    # Get the log likelihood of the test
    total_log_proba = scipy.stats.norm.logpdf(
        y_true, y_pred, 1.0/np.sqrt(beta))
    return np.sum(total_log_proba) / len(y_true)


# ---------------- Main execution loop ----------------

score_total = 0

# Read train/test once to grab gender columns
train_df = pd.read_csv("../../data/processed/gender/train6001.csv")
test_df  = pd.read_csv("../../data/processed/gender/test6001.csv")

# Option B: Gender already coded as integers in CSV.
# Still create a mapping to ensure ids are 0..K-1 and consistent.
gender_train_raw = train_df["Gender"].values
gender_test_raw  = test_df["Gender"].values

# create internal mapping (in case codes are not contiguous)
unique_genders = np.unique(gender_train_raw)
gender_map = {g:i for i,g in enumerate(unique_genders)}
cluster_labels_train_all = np.array([gender_map[g] for g in gender_train_raw])

# For test mapping: ensure all test genders appear in train mapping
# If a test-only gender appears, map it to a new index appended after existing ones.
test_unique = np.unique(gender_test_raw)
next_idx = len(gender_map)
for g in test_unique:
    if g not in gender_map:
        gender_map[g] = next_idx
        next_idx += 1

cluster_labels_test_all = np.array([gender_map[g] for g in gender_test_raw])

K_all = len(np.unique(cluster_labels_train_all))
print(f"Detected gender clusters (K) = {K_all}")

for emotion in ["Joy","Happiness","Calmness","Relaxation","Anger","Disgust","Fear","Anxiousness","Sadness"] :

    print(f"Emotion: {emotion}")

    alpha, beta, degree, interactions = run_map_for_emotion(emotion)

    X_train, y_train, X_test, y_test = get_train_test_data(emotion, degree = 2, interact=True)

    X_train2, y_train2, X_test2, y_test2 = get_train_test_data(emotion, degree = 1, interact=False)


    scaler = StandardScaler() 

    X_train_for_cluster = scaler.fit_transform(X_train2)
    X_test_for_cluster  = scaler.transform(X_test2)

    # Use K equal to number of genders detected in training
    K = K_all

    # When tuning, force k_list to [K] so K is determined by gender only
    best_m, best_s, best_k, best_score = tune_hyperparameters(
        X_train, X_train_for_cluster, y_train, cluster_labels_train_all,
        cv_K=5, max_iter=6, alpha=alpha, beta=beta,
        m_list=[1,2], s_list=[3,5,10], k_list=[K]
    )

    print(f"Best m: {best_m}")
    print(f"Best s: {best_s}")
    print(f"Best k: {best_k}")
    print(f"kfcv score {best_score}")

    # Train and evaluate using fixed gender labels
    score_total += train_and_eval(
        X_train, X_train_for_cluster, y_train,
        X_test, X_test_for_cluster, y_test,
        cluster_labels_train_all, cluster_labels_test_all,
        best_m, best_s, K, 6, alpha, beta
    )
    print()

print(f"Average score: {score_total/9}")
