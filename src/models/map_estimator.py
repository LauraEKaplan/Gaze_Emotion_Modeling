import numpy as np
import pandas as pd
import scipy



# get Phi and Y for both train and test
def get_train_test_data(emotion, degree = 2, interact = False) :
    # degree: degree of polynomial features
    # interact : add interacting terms (x1x2)

    train = pd.read_csv("../../data/processed/train.csv")
    test = pd.read_csv("../../data/processed/test.csv")

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




def bayesian_linear_regression(X, y, alpha=1.0, beta = 1.0):
    n_samples, n_features = X.shape

    # Compute the MAP estimate of w
    w_map = np.linalg.solve(((X.T @ X) + (alpha/beta) * np.identity(n_features)), (X.T @ y))
    
    return w_map

def calc_log_evidence(X, y, alpha, beta):

    w_map = bayesian_linear_regression(X, y, alpha, beta)

    ## Transform the raw features to polynomial ones
    N, M = X.shape

    precision = (beta * (X.T @ X) + (alpha) * np.identity(M))
    
    avg_log_proba = (M * np.log(alpha))/2 + (N * np.log(beta))/2 
    avg_log_proba = avg_log_proba - (beta/2)*(np.dot(y - X @ w_map, y - X @ w_map)) - (alpha/2)*(w_map.T @ w_map)
    avg_log_proba = avg_log_proba - np.log(np.linalg.det(precision))/2
    avg_log_proba = avg_log_proba - (N * np.log(2 * np.pi))/2
    return avg_log_proba

# Takes one mean for all predictions
def calc_score_baseline(X, y, mean, beta = 1.0) :

    # Predict
    N, M = X.shape

    # Get the log likelihood of the test
    total_log_proba = scipy.stats.norm.logpdf(
        y, mean, 1.0/np.sqrt(beta))
    return np.sum(total_log_proba) / N


def calc_score(X, y, w, beta = 1.0) :

    # Predict
    N, M = X.shape
    y_est = X @ w

    # Get the log likelihood of the test
    total_log_proba = scipy.stats.norm.logpdf(
        y, y_est, 1.0/np.sqrt(beta))
    return np.sum(total_log_proba) / N

# Evidence maximization algorithm
def get_alpha_beta_evidence(emotion,
                            alpha_grid = [10,15,20,25,30,35,40,45,50,55,60,65,70, 65, 80, 85, 90, 95, 100, 105], 
                            beta_grid = [0.1,0.5,1,5,10,15,20,25,30,35],
                            degree_grid = [1,2,3]) :
    


    best_alpha = 0
    best_beta = 0
    best_evidence = -np.finfo(np.float32).max
    best_degree = 0
    interactions = False
    for d in degree_grid :
        for interaction in [True, False] :
            X, y, X_test, Y_test = get_train_test_data(emotion, d, interaction)

            for a in alpha_grid :
                for b in beta_grid :
                    evidence = calc_log_evidence(X, y, a, b)
                    if evidence > best_evidence :
                        best_alpha = a
                        best_beta = b
                        best_evidence = evidence
                        best_degree = d
                        interactions = interaction

    print(f"Best alpha: {best_alpha}")
    print(f"Best beta: {best_beta}")
    print(f"Best degree: {best_degree}")
    print(f"Interactions: {interactions}")
    print(f"Evidence: {best_evidence}")
    return best_alpha, best_beta, best_degree, interactions


# Parameters for the param transform

# Test all emotions
total_improvement = 0
total_ratio = 0

for emotion in ["Joy","Happiness","Calmness","Relaxation","Anger","Disgust","Fear","Anxiousness","Sadness"] :
    print(f"Emotion: {emotion}")

    # get best alpha and beta by evidence 
    alpha, beta, degree, interactions = get_alpha_beta_evidence(emotion)

    X_train, Y_train, X_test, Y_test = get_train_test_data(emotion, degree, interactions)


    # train model
    w_map = bayesian_linear_regression(X_train, Y_train, alpha, beta)

    # Print results
    print(f"Baseline (mean is zero) score: {calc_score_baseline(X_test, Y_test, 0, beta)}")
    print(f"Baseline (mean is train Y mean) score: {calc_score_baseline(X_test, Y_test, np.mean(Y_train), beta)}")
    print(f"MAP score: {calc_score(X_test, Y_test, w_map, beta)}")
    print("----------------------------------")
    print()

    total_improvement += calc_score(X_test, Y_test, w_map, beta) - calc_score_baseline(X_test, Y_test, np.mean(Y_train), beta)
    total_ratio += calc_score(X_test, Y_test, w_map, beta) / calc_score_baseline(X_test, Y_test, np.mean(Y_train), beta)

print(f"Average Improvement (Linear): {total_improvement/9}")
print(f"Average Ratio: {total_ratio/9}")
