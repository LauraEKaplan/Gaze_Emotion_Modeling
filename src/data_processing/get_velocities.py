import numpy as np
import pandas as pd
import os
import pickle
import math

OUTPUT = "velo.pkl"
folder_path = "../04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)/"


def angular_distance(x1, y1, x2, y2):
    # convert to radians
    x1, y1, x2, y2 = map(math.radians, [x1, y1, x2, y2])

    # compute spherical (great-circle) distance
    d = math.acos(
        math.sin(y1)*math.sin(y2) + math.cos(y1)*math.cos(y2)*math.cos(x1 - x2)
    )
    
    # convert back to degrees if desired
    return math.degrees(d)

def get_velocity_data(data) :
    vel_data = {"duration": [], "velocity":[], "blink":[]}

    curr_blink_len = 0
    curr_blink = False
    for idx in range(1, len(data[0])) :
        # Not sure how to deal with blinks, for now all sections that contain a start or end with blink are blinks? I think with 60hz this should be ok?
        # Not sure about winks either
        # Start simple: just a blink if either eye blinks

        # Get time difference
        timestamp_diff = data[0][idx] - data[0][idx-1]

        blink = data[3][idx] > 0 or data[6][idx] > 0

        # if we are finishing a blink
        if curr_blink and not blink :
            curr_blink_len += timestamp_diff / 2
            vel_data["blink"].append(1)
            vel_data["duration"].append(curr_blink_len)
            vel_data["velocity"].append(np.nan)

            curr_blink = False
            curr_blink_len = 0

        # extend a blink
        if blink and curr_blink :
            curr_blink_len += timestamp_diff

        # Start a blink
        if blink and not curr_blink :
            curr_blink_len = timestamp_diff/2
            curr_blink = True

        # Not blinkin
        if not blink and not curr_blink :
            vel_data["blink"].append(0)

            vel_data["duration"].append(timestamp_diff)
            ang_dist_left_eye = angular_distance(data[1][idx], data[2][idx], data[1][idx-1], data[2][idx-1])
            ang_dist_right_eye = angular_distance(data[4][idx], data[5][idx], data[4][idx-1], data[5][idx-1])

            # Average of both
            vel_data["velocity"].append((ang_dist_left_eye + ang_dist_right_eye) / (2 * timestamp_diff))

    return vel_data


if __name__ == '__main__':

    # {patient_no: [(label, {duration: [], velocity: [], blink: []})]}
    data_dict = {}

    for filename in os.listdir(folder_path):
        # Ignore other files in case
        if filename.startswith("1") and filename.endswith(".dat"):
            file_path = os.path.join(folder_path, filename)

            # Get patient number
            patient_no = filename.split('_')[0]
            patient_no = patient_no[1:]
            
            # load the pickle data
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            labels = data['Labels'].astype(int).ravel()
            trials = data['Data']

            if len(trials) == len(labels) :

                data_dict[patient_no] = []

                for i, lab in enumerate(labels) :
                    data_dict[patient_no].append((lab, get_velocity_data(trials[i])))

    with open(OUTPUT, "wb") as f:
        pickle.dump(data_dict, f)

