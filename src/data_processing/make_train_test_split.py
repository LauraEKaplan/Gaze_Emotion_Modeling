import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split



with open('post_emotions_normalized', "rb") as f:
    emotions = pickle.load(f)

gaze_data = pd.read_csv("featurized.csv")

trial_no_to_code = {0 : "BNS", 1: "BRZ", 2: "DST", 3: "EXR", 4: "JNG", 5: "PRS", 6: "BOT", 7: "RFS", 8: "RPW", 9: "RST", 10: "TNT", 11: "ZMZ"}

full_data = pd.merge(gaze_data, emotions, on=['ID', 'Str_Code'], how='inner')

train, test = train_test_split(full_data, test_size=0.2, random_state=10000)

train.to_csv("../../data/processed/train.csv", index = False)
test.to_csv("../../data/processed/test.csv", index = False)



