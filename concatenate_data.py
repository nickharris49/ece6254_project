## KNEE BIOIMPEDANCE DEVICE WORN ON LEFT LEG

import numpy as np
import matplotlib.pyplot as plt
import os
from open_dataset import *

## README
# this file gets all of the data that was cleaned and organized in reorganize_dataset.py
# as of right now (march 25) that data is just the base knee data

def main():
    # pathing
    base_path = get_path_to_dataset()
    full_path = base_path + '/nicks_reworked_dataset/'
    
    # get all the data file names
    data_files = os.listdir(full_path)
    data_files.remove('.DS_Store')
    for data_file in data_files:
        subj_num = data_file.split("_")[-1].split(".")[0]

        # instantiate big data files for training
        y = []
        r5k = []
        x5k = []
        r100k = []
        x100k = []
        data = np.load(full_path + data_file)
        for window in data:
            y.append((window['knee_angle_l'] - np.mean(window['knee_angle_l'])) / np.std(window['knee_angle_l']))
            r5k.append((window['knee_bioz_5k_resistance'] - np.mean(window['knee_bioz_5k_resistance'])) / np.std(window['knee_bioz_5k_resistance']))
            x5k.append((window['knee_bioz_5k_reactance'] - np.mean(window['knee_bioz_5k_reactance'])) / np.std(window['knee_bioz_5k_reactance']))
            r100k.append((window['knee_bioz_100k_resistance'] - np.mean(window['knee_bioz_100k_resistance'])) / np.std(window['knee_bioz_100k_resistance']))
            x100k.append((window['knee_bioz_100k_reactance'] - np.mean(window['knee_bioz_100k_reactance'])) / np.std(window['knee_bioz_100k_reactance']))

        y = np.expand_dims(np.concatenate(y, axis=0), axis=1)
        r5k = np.expand_dims(np.concatenate(r5k, axis=0), axis=1)
        x5k = np.expand_dims(np.concatenate(x5k, axis=0), axis=1)
        r100k = np.expand_dims(np.concatenate(r100k, axis=0), axis=1)
        x100k = np.expand_dims(np.concatenate(x100k, axis=0), axis=1)

        feat_vec_5k = np.concatenate((r5k, x5k), axis=1)
        feat_vec_100k = np.concatenate((r100k, x100k), axis=1)
        feat_vec_full = np.concatenate((r5k, x5k, r100k, x100k), axis=1)

        np.save(full_path + "feature_vector_5k_normalized_" + subj_num + ".npy", feat_vec_5k)
        np.save(full_path + "feature_vector_100k_normalized_" + subj_num + ".npy", feat_vec_100k)
        np.save(full_path + "feature_vector_full_normalized_" + subj_num + ".npy", feat_vec_full)
        np.save(full_path + "y_normalized_" + subj_num + ".npy", y)

if __name__ == '__main__':
    main()