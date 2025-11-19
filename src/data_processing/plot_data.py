# Note: this calculates percentages based on each segment, not time. Segments should be good enough as a proxy

import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


with open('velo.pkl', "rb") as f:
    raw_velo = pickle.load(f)



#####################################################
# SETTINGS

CUTOFF_VELO = 98
CUTOFF_BLINK = 98
#######################################################


def get_velos(raw_velo) :

    # {person: [velo1, velo2, ...]}
    velo_person = defaultdict(list)

    for person_num in raw_velo :
        for data_pair in raw_velo[person_num] :
            data = data_pair[1]
            # get rid of blinks
            vel_no_blink = data["velocity"]
            vel_no_blink = [v for v in vel_no_blink if not np.isnan(v)]

            velo_person[person_num].extend(vel_no_blink)
        

    all_velo = []
    for key, value in velo_person.items() :
        all_velo.extend(value)


    return all_velo


def get_blinks(raw_velo) :

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

    all_blink = []
    for key, value in blink_person.items() :
        all_blink.extend(value)

    return all_blink


all_velo = get_velos(raw_velo)
all_blink = get_blinks(raw_velo)
all_blink = np.array(all_blink, dtype=np.float64)


threshold = np.percentile(all_blink, CUTOFF_BLINK)

# Step 2: Filter the array to remove values >= the threshold
filtered_arr = all_blink[all_blink < threshold]

plt.hist(filtered_arr, bins = 500)
plt.xlabel("Blink Time")
plt.ylabel("Count")
plt.show()

all_velo = np.array(all_velo, dtype=np.float64)

threshold = np.percentile(all_velo, CUTOFF_VELO)

# Step 2: Filter the array to remove values >= the threshold
filtered_arr = all_velo[all_velo < threshold]

plt.hist(filtered_arr, bins = 500)
plt.xlabel("Velocity")
plt.ylabel("Count")
plt.show()