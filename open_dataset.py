import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import matlab.engine as meng
import pickle

from Utilities.Dataset import parse_file_for_data_sets
from Utilities.FileParser import parse_file
from DataAnalysis.KneeDataAnalysis import *
from Utilities.AdvancedLoggingParser import *
from Utilities.Dataset import parse_file_for_data_sets
from Utilities.AdvancedLoggingParser import *
from Utilities.Dataset import parse_file_for_data_sets

def get_path_to_dataset():

    if os.getlogin() == 'nicholasharris':
        base_path = '/Users/nicholasharris/OneDrive - Georgia Institute of Technology/BioZ Dataset'
    
    return base_path

def get_bioz_from_subject(subject: int, missing_inds=[], exo=False, ak_or_kn=False):

    ## pathing stuff
    base_path = get_path_to_dataset()
    if exo:
        exo_path = 'exo'
    else:
        exo_path = 'base'

    if ak_or_kn:
        akkn_path = 'ak'
    else:
        akkn_path = 'kn'
    bioz_path = base_path + '/bioimpedance/s' + str(subject) + '_' + akkn_path + '_' + exo_path + '.obj'
    print(bioz_path)

    ## open data file (.obj)
    filehandler = open(bioz_path, 'rb')
    data = pickle.load(filehandler)
    filehandler.close()

    walking_data = data[4] # the 30min walking will always be at index 4

    bioz_5k_resistance = []
    bioz_5k_reactance = []
    bioz_100k_resistance = []
    bioz_100k_reactance = []
    for i in range(30):
        print(np.shape(walking_data[i].BioZ1.Data[0]))
    for i in range(30):
        if np.shape(walking_data[i].BioZ1.Data[0])[0] == 0 or i + 1 in missing_inds:
            print("MISSING BIOZ DATA FROM SUBJECT " + str(subject) + " AT TRIAL " + str(i+1) + ". " + ("EXO" if exo else "NO EXO"))
            continue
        bioz_5k_resistance.append(scipy.signal.resample(walking_data[i].BioZ1.Data[0][:,1], 6000))
        bioz_5k_reactance.append(scipy.signal.resample(walking_data[i].BioZ1.Data[0][:,2], 6000))
        bioz_100k_resistance.append(scipy.signal.resample(walking_data[i].BioZ1.Data[1][:,1],6000))
        bioz_100k_reactance.append(scipy.signal.resample(walking_data[i].BioZ1.Data[1][:,2],6000))

    return bioz_5k_resistance, bioz_5k_reactance, bioz_100k_resistance, bioz_100k_reactance


def get_biomech_from_subject(subject: int, exo=False):

    ## pathing stuff
    base_path = get_path_to_dataset()
    if exo:
        exo_path = '/Exo'
    else:
        exo_path = '/No Exo'
    biomech_path = base_path + '/biomechanics/fatigue_' + str(subject) + exo_path + '/pre_dat.mat'
    print(biomech_path)

    eng = meng.start_matlab()
    data = eng.load(biomech_path)
    walking_data = data['angles']['main']['walk']
    hip_flexion_r = []
    hip_flexion_l = []
    hip_adduction_r = []
    hip_adduction_l = []
    knee_angle_r = []
    knee_angle_l = []
    ankle_angle_r = []
    ankle_angle_l = []

    missing_inds = []
    for i in range(1,31):
        try:
            angles = walking_data['g' + str(i)]
        except:
            print("MISSING MECH DATA FROM SUBJECT " + str(subject) + " AT g" + str(i) + ". " + ("EXO" if exo else "NO EXO"))
            missing_inds.append(i)
            continue
        hip_flexion_r.append(angles['hip_flexion_r'])
        hip_flexion_l.append(angles['hip_flexion_l'])
        hip_adduction_r.append(angles['hip_adduction_r'])
        hip_adduction_l.append(angles['hip_adduction_l'])
        knee_angle_r.append(angles['knee_angle_r'])
        knee_angle_l.append(angles['knee_angle_l'])
        ankle_angle_r.append(angles['ankle_angle_r'])
        ankle_angle_l.append(angles['ankle_angle_l'])

    return (hip_flexion_r, hip_flexion_l, hip_adduction_r, hip_adduction_l, knee_angle_r, knee_angle_l, ankle_angle_r, ankle_angle_l), missing_inds






def main():
    base_path = get_path_to_dataset()
    subject = 11
    ## ANKLE WORN ON RIGHT LEG
    ## KNEE WORN ON LEFT LEG

    bioz_keys = ['ankle_bioz_5k_resistance', 'ankle_bioz_5k_reactance', 'ankle_bioz_100k_resistance', 'ankle_bioz_100k_reactance', 'knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance']
    mech_keys = ['hip_flexion_r', 'hip_flexion_l', 'hip_adduction_r', 'hip_adduction_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l']

    dtype = []
    for key in bioz_keys:
        dtype.append((key, 'f8'))

    for key in mech_keys:
        dtype.append((key, 'f8'))

    #get_biomech_from_subject(subject)
    mech_vals, missing_inds = get_biomech_from_subject(subject)
    ankle_bioz = get_bioz_from_subject(subject, missing_inds, ak_or_kn=True)
    knee_bioz = get_bioz_from_subject(subject, missing_inds, ak_or_kn=False)
    recarray = np.ones(np.shape(np.squeeze(mech_vals[0])), dtype=dtype)

    for i, mech_val in enumerate(mech_vals):
        recarray[mech_keys[i]] = np.squeeze(mech_val)

    #hip_flexion_r, hip_flexion_l, hip_adduction_r, hip_adduction_l, knee_angle_r, knee_angle_l, ankle_angle_r, ankle_angle_l = get_biomech_from_subject(subject)
    # for i in range(30):
    #     fig, axs = plt.subplots(nrows=3)
    #     axs[0].plot(hip_flexion_l[i])
    #     axs[1].plot(knee_angle_l[i])
    #     axs[2].plot(ankle_angle_l[i])
    #     plt.show(block=False)
    # plt.show(block=True)
    
    ## ankle bioz
    for i, bioz_val in enumerate(ankle_bioz):
        recarray[bioz_keys[i]] = bioz_val
    # ankle_bioz_5k_resistance, ankle_bioz_5k_reactance, ankle_bioz_100k_resistance, ankle_bioz_100k_reactance = get_bioz_from_subject(subject, ak_or_kn=True)

    # for i in range(30):
    #     fig, axs = plt.subplots(nrows=2)
    #     axs[0].plot(ankle_bioz_5k_resistance[i])
    #     axs[0].plot(ankle_bioz_100k_resistance[i])
    #     axs[1].plot(ankle_bioz_5k_reactance[i])
    #     axs[1].plot(ankle_bioz_100k_reactance[i])
    #     plt.show(block=False)
    # plt.show(block=True)
    ## knee bioz
    for i, bioz_val in enumerate(knee_bioz):
        recarray[bioz_keys[i+4]] = bioz_val
    #knee_bioz_5k_resistance, knee_bioz_5k_reactance, knee_bioz_100k_resistance, knee_bioz_100k_reactance = get_bioz_from_subject(subject, ak_or_kn=False)
    # for i in range(30):
    #     fig, axs = plt.subplots(nrows=2)
    #     axs[0].plot(bioz_5k_resistance[i])
    #     axs[0].plot(bioz_100k_resistance[i])
    #     axs[1].plot(bioz_5k_reactance[i])
    #     axs[1].plot(bioz_100k_reactance[i])
    #     plt.show(block=True)
    ## biomech

    # bioz_keys = ['ankle_bioz_5k_resistance', 'ankle_bioz_5k_reactance', 'ankle_bioz_100k_resistance', 'ankle_bioz_100k_reactance', 'knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance']
    # mech_keys = ['hip_flexion_r', 'hip_flexion_l', 'hip_adduction_r', 'hip_adduction_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l']

    # dtype = []
    # for key in bioz_keys:
    #     dtype.append((key, 'f8'))

    ## this is just to visually check for drift (there is none)
    fig, axs = plt.subplots(nrows=8)
    axs[0].plot(recarray['ankle_bioz_5k_resistance'][0][:500])
    axs[1].plot(recarray['ankle_angle_r'][0][:500])
    axs[2].plot(recarray['ankle_bioz_5k_resistance'][0][1000:1500])
    axs[3].plot(recarray['ankle_angle_r'][0][1000:1500])
    axs[4].plot(recarray['ankle_bioz_5k_resistance'][0][2000:2500])
    axs[5].plot(recarray['ankle_angle_r'][0][2000:2500])
    axs[6].plot(recarray['ankle_bioz_5k_resistance'][0][3000:3500])
    axs[7].plot(recarray['ankle_angle_r'][0][3000:3500])
    plt.show(block=True)

    np.save(base_path + '/resampled_' + str(subject) + '.npy', recarray)
    # for key in mech_keys:
    #     dtype.append((key, 'f8'))
    dumy=1



if __name__ == '__main__':
    main()