import numpy as np
import pandas as pd
import os
import pickle
import math
import matplotlib.pyplot as plt

folder_path = "../04 Eye Tracking Data/01 Eye Tracking Data (Pre-Processed)/"



def get_blinks(data) :
    blink_lens = []
    blink_frames = []

    curr_blink_frames = 0
    curr_blink_lens = 0
    for idx in range(1, len(data[0])) :
        # Not sure how to deal with blinks, for now all sections that contain a start or end with blink are blinks? I think with 60hz this should be ok?
        # Not sure about winks either
        # Start simple: just a blink if either eye blinks
        blink = data[3][idx] > 0 or data[6][idx] > 0 or data[3][idx-1] > 0 or data[6][idx - 1] > 0 
        timestamp_diff = data[0][idx] - data[0][idx-1]
        
        if blink :
            curr_blink_frames += 1
            curr_blink_lens += timestamp_diff
        
        # Reset
        else :
            if curr_blink_frames > 0:
                blink_lens.append(curr_blink_lens)
                blink_frames.append(curr_blink_frames)
            
            curr_blink_frames = 0
            curr_blink_lens = 0
            

    return blink_lens, blink_frames


if __name__ == '__main__':

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

            



          

            if len(trials) == len(labels):

                n_plots = min(len(labels), 12)  # up to 12 plots

                # Create grid (e.g. 3x4 if 12 plots)
                n_cols = 4
                n_rows = int(np.ceil(n_plots / n_cols))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
                axes = axes.flatten()

                for ax_idx, lab in enumerate(labels[:12]):


                    blink_lens, blink_frames = get_blinks(trials[ax_idx])

                    blink_frames = np.array(blink_frames)
                    bins = np.arange(blink_frames.min(), blink_frames.max() + 2) - 0.5

                    ax = axes[ax_idx]
                    ax.hist(blink_frames, bins=bins)
                    ax.set_title(f"Test {ax_idx}: QuadCat {lab}")
                    ax.set_xlabel("Frames")
                    ax.set_ylabel("Blink Count")

                # Remove any empty subplots
                for ax in axes[len(labels):]:
                    ax.set_visible(False)

                plt.tight_layout()
                plt.show()


                fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 8))
                axes = axes.flatten()

                for ax_idx, lab in enumerate(labels[:12]):
                    # Get the index for this label
                    blink_lens, blink_frames = get_blinks(trials[ax_idx])

                    blink_frames = np.array(blink_frames)

                    ax = axes[ax_idx]
                    ax.hist(blink_lens)
                    ax.set_title(f"Test {ax_idx}: QuadCat {lab}")
                    ax.set_xlabel("Length (Time)")
                    ax.set_ylabel("Blink Count")

                # Remove any empty subplots
                for ax in axes[len(labels):]:
                    ax.set_visible(False)

                plt.tight_layout()
                plt.show()
