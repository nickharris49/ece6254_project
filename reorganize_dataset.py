import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import matlab.engine as meng
import pickle
from open_dataset import *

## README
# This list of subjects is all of those which have NO missing data for their knee bioimpedance, with no exoskeleton
# This file loads the messy data for these subjects and organizes it into a numpy record array
# The new dataset looks as follows:
# Each subject file contains 16 timeseries signals which are already aligned.
# Each signal is chunked into 60second windows, sampled at 100Hz. Each window is thus 6000 samples.
# A perfect subject would have 30 windows (30 minutes of data), 
# but many subjects have erroneous windows where either the knee bioz, ankle bioz, or biomech recording failed.
# Thus, each subject has a .npz file with anywhere from 20 to 30 windows of data, 6000 samples each.
# If you call np.shape on one of the subject's .npz files, it will return (~25, 6000) - this is misleading.
# record arrays are sort of like dictionaries, and if you access the recarray by saying recarray['hip_flexion_r'], you'll get a 25x6000 array.
# so really, the size of the array is 16x25x6000, but that detail is abstracted away.

def main():
    # pathing and important globals
    # for some reason, 9 and 10 missing from the biomech, so they can't be included in the subjects list
    base_path = get_path_to_dataset()
    subjects_knee_base = [1, 2, 3, 5, 6, 7, 8, 11]
    subjects_ankle_base = [3, 4, 5, 6, 7, 8, 11]
    subjects_knee_exo = [1, 2, 6, 7, 8, 11]
    subjects_ankle_exo = [1, 3, 4, 5, 6, 7, 8, 11]

    # these are the 'keys' for the numpy record array which stores the (in my opinion) more organized data
    ankle_bioz_keys = ['ankle_bioz_5k_resistance', 'ankle_bioz_5k_reactance', 'ankle_bioz_100k_resistance', 'ankle_bioz_100k_reactance']
    knee_bioz_keys = ['knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance']
    mech_keys = ['hip_flexion_r', 'hip_flexion_l', 'hip_adduction_r', 'hip_adduction_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l']

    # create the 'dtype' for the recarray
    # this is similar to statically instantiating the keys of a python dictionary

    # iterate over the valid subjects
    for joint in ['knee', 'ankle']:
        for exo in ['exo', 'base']:
            dtype = []
            for key in knee_bioz_keys:
                dtype.append((key, 'f8'))

            for key in mech_keys:
                dtype.append((key, 'f8'))
            for subject in subjects_knee_base:
                save_path = base_path + '/nicks_reworked_dataset/knee_base_' + str(subject) + '.npy'
                if 'knee_base_' + str(subject) in os.listdir(base_path + '/nicks_reworked_dataset/'):
                    continue

                # get the indices of data (60sec windows) which are NOT missing across all data types
                # e.g. if window number 10 is missing in any of the biomech recording, the ankle bioz, or the knee bioz, it will NOT be included
                clean_inds = get_clean_inds(subject, bioz_to_include='kn', exo=False)
                mech_vals = get_biomech_from_subject(subject, clean_inds, exo=False)

                ankle_bioz = get_bioz_from_subject(subject, clean_inds, ak_or_kn=True, exo=False)
                knee_bioz = get_bioz_from_subject(subject, clean_inds, ak_or_kn=False, exo=False)
                recarray = np.ones(np.shape(np.squeeze(mech_vals[0])), dtype=dtype)

                # populate record array with joint angles
                for i, mech_val in enumerate(mech_vals):
                    recarray[mech_keys[i]] = np.squeeze(mech_val)

                # populate record array with ankle bioimpedance values
                for i, bioz_val in enumerate(ankle_bioz):
                    recarray[ankle_bioz_keys[i]] = bioz_val
                
                # populate record array with knee bioimpedance values
                for i, bioz_val in enumerate(knee_bioz):
                    recarray[knee_bioz_keys[i]] = bioz_val
                
                # save the new data
                save_path = base_path + '/nicks_reworked_dataset/knee_base_' + str(subject) + '.npy'
                np.save(save_path, recarray)




if __name__ == '__main__':
    main()