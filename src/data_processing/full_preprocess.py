
# Note: this calculates percentages based on each segment, not time. Segments should be good enough as a proxy

import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split


with open('velo.pkl', "rb") as f:
    raw_velo = pickle.load(f)

#####################################################
# SETTINGS

# If each box is linearly spaced
# Nonlinear is not implemented
LINEAR_SPLIT = True

# If we look at the global max instead of each person's max
VELO_USE_GLOBAL_STATS = True
BLINK_USE_GLOBAL_STATS = False

VELO_SPLIT_COUNT = 9
BLINK_SPLIT_COUNT = 6

# You can get rid of the biggest outliers when doing the splits (does NOT remove the data)
VELO_TOP_CUTOFF = 95
VELO_BOTTOM_CUTOFF = 0

BLINK_TOP_CUTOFF = 90
BLINK_BOTTOM_CUTOFF = 0

#######################################################

# Gets the velocity cutoffs per person
# {person: [cutoff1, cutoff2, ...]}
def get_velo_cutoffs(raw_velo) :

    # {person: [velo1, velo2, ...]}
    velo_person = defaultdict(list)

    # {person: [cutoff1, cutoff2, ...]}
    cutoffs_velo = defaultdict(list)

    for person_num in raw_velo :
        for data_pair in raw_velo[person_num] :
            data = data_pair[1]
            # get rid of blinks
            vel_no_blink = data["velocity"]
            vel_no_blink = [v for v in vel_no_blink if not np.isnan(v)]

            velo_person[person_num].extend(vel_no_blink)
        
        # Get per person
        if VELO_USE_GLOBAL_STATS == False :
            bottom = np.percentile(velo_person[person_num], VELO_BOTTOM_CUTOFF)
            top = np.percentile(velo_person[person_num], VELO_TOP_CUTOFF)
            cutoffs_velo[person_num] = np.linspace(bottom, top, VELO_SPLIT_COUNT + 1)[1:-1]

    # If we are using global, then re-aggregate all velocities
    if VELO_USE_GLOBAL_STATS :
        all_velo = []
        for key, value in velo_person.items() :
            all_velo.extend(value)

        bottom = np.percentile(all_velo, VELO_BOTTOM_CUTOFF)
        top = np.percentile(all_velo, VELO_TOP_CUTOFF)

        for key, value in velo_person.items() :

            cutoffs_velo[key] = np.linspace(bottom, top, VELO_SPLIT_COUNT + 1)[1:-1]

    return cutoffs_velo


# Gets the velocity cutoffs per person
# {person: [cutoff1, cutoff2, ...]}
def get_blink_cutoffs(raw_velo) :

    # {person: [dur1, dur2, ...]}
    blink_person = defaultdict(list)

    # {person: [cutoff1, cutoff2, ...]}
    cutoffs_blink = defaultdict(list)

    for person_num in raw_velo :
        for data_pair in raw_velo[person_num] :
            data = data_pair[1]
            # get rid of blinks
            blink_durs = [v for v, b in zip(data["duration"], data["blink"]) if b != 0]
            blink_person[person_num].extend(blink_durs)
        
        # Get per person
        if BLINK_USE_GLOBAL_STATS == False :
            bottom = np.percentile(blink_person[person_num], BLINK_BOTTOM_CUTOFF)
            top = np.percentile(blink_person[person_num], BLINK_TOP_CUTOFF)
            cutoffs_blink[person_num] = np.linspace(bottom, top, BLINK_SPLIT_COUNT + 1)[1:-1]

    # If we are using global, then re-aggregate all velocities
    if BLINK_USE_GLOBAL_STATS :
        all_blink = []
        for key, value in blink_person.items() :
            all_blink.extend(value)

        bottom = np.percentile(all_blink, BLINK_BOTTOM_CUTOFF)
        top = np.percentile(all_blink, BLINK_TOP_CUTOFF)

        for key, value in blink_person.items() :

            cutoffs_blink[key] = np.linspace(bottom, top, BLINK_SPLIT_COUNT + 1)[1:-1]
    

    return cutoffs_blink

def create_array(cutoffs_velo, cutoffs_blink, raw_velo) :
    trial_no_to_code = {0 : "BNS", 1: "BRZ", 2: "DST", 3: "EXR", 4: "JNG", 5: "PRS", 6: "BOT", 7: "RFS", 8: "RPW", 9: "RST", 10: "TNT", 11: "ZMZ"}

    df = pd.DataFrame(columns = ["ID", "Trial_no", "Trial_quad", "Str_Code"])

    # Create dynamic number of columns based on split count
    for i in range(BLINK_SPLIT_COUNT) :
        df[f"Blink_s{i}"] = []
    for i in range(VELO_SPLIT_COUNT) :
        df[f"Velo_s{i}"] = []

    for person_num in raw_velo :
        for idx, data_pair in enumerate(raw_velo[person_num]) :
            row = dict.fromkeys(df.columns)
            row["ID"] = 100 + int(person_num)
            row["Trial_no"] = idx
            row["Trial_quad"] = data_pair[0]
            row["Str_Code"] = trial_no_to_code[idx]


            for i in range(BLINK_SPLIT_COUNT) :
                row[f"Blink_s{i}"] = 0
            for i in range(VELO_SPLIT_COUNT) :
                row[f"Velo_s{i}"] = 0


            # First velocities
            velocities = np.array(data_pair[1]["velocity"])
            durations = np.array(data_pair[1]["duration"])
            total_time = np.sum(durations)

            # Mask NaN values
            mask = ~np.isnan(velocities)
            valid_velo = velocities[mask]
            valid_dur = durations[mask]

            # Get which one it belongs to
            bin_idx = np.digitize(valid_velo, cutoffs_velo[person_num])
            for i in range(len(bin_idx)) :
                row[f"Velo_s{bin_idx[i]}"] += valid_dur[i]


            # Second blinks
            blinks = np.array(data_pair[1]["blink"])

            # Mask non-NaN values
            mask = np.isnan(velocities)
            valid_blink = blinks[mask]
            valid_dur = durations[mask]

            # Get which one it belongs to
            bin_idx = np.digitize(valid_dur, cutoffs_blink[person_num])
            for i in range(len(bin_idx)) :
                row[f"Blink_s{bin_idx[i]}"] += valid_dur[i]

            # Normalize
            for i in range(BLINK_SPLIT_COUNT) :
                row[f"Blink_s{i}"] = row[f"Blink_s{i}"]/total_time
            for i in range(VELO_SPLIT_COUNT) :
                row[f"Velo_s{i}"] = row[f"Velo_s{i}"]/total_time
            df.loc[len(df)] = row

    with open('post_emotions_normalized', "rb") as f:
        emotions = pickle.load(f)

    df.to_csv("./featurized.csv", index = False)

    
    trial_no_to_code = {0 : "BNS", 1: "BRZ", 2: "DST", 3: "EXR", 4: "JNG", 5: "PRS", 6: "BOT", 7: "RFS", 8: "RPW", 9: "RST", 10: "TNT", 11: "ZMZ"}

    full_data = pd.merge(df, emotions, on=['ID', 'Str_Code'], how='inner')

    train, test = train_test_split(full_data, test_size=0.2, random_state=42)

    train.to_csv("../../data/processed/train.csv", index = False)
    test.to_csv("../../data/processed/test.csv", index = False)

cutoffs_velo = get_velo_cutoffs(raw_velo)
cutoffs_blink = get_blink_cutoffs(raw_velo)
create_array(cutoffs_velo, cutoffs_blink, raw_velo)




