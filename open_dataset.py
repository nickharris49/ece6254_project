## ANKLE BIOIMPEDANCE DEVICE WORN ON RIGHT LEG
## KNEE BIOIMPEDANCE DEVICE WORN ON LEFT LEG

import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import matlab.engine as meng
import pickle

## README
# If you want to use / play around with this function on your local machine, you'll need to update this function to include your path as an option
# i.e. add an elif os.getlogin() == 'your_login'
#                  base_path = 'your_path_to_dataset'
def get_path_to_dataset():

    if os.getlogin() == 'nicholasharris':
        base_path = '/Users/nicholasharris/OneDrive - Georgia Institute of Technology/BioZ Dataset'
    return base_path

def get_clean_inds(subject: int, exo=False, ak_or_kn=False, bioz_to_include='both'):
    clean_inds = []
    ## BIOMECH INDEX CHECKING
    ## pathing to the dataset / subject
    base_path = get_path_to_dataset()
    if exo:
        exo_path = '/Exo'
    else:
        exo_path = '/No Exo'
    biomech_path = base_path + '/biomechanics/fatigue_' + str(subject) + exo_path + '/pre_dat.mat'
    print(biomech_path)

    ## mat file relies on matlab engine to open
    eng = meng.start_matlab()
    data = eng.load(biomech_path)
    walking_data = data['angles']['main']['walk']

    ## get the clean indices for biomech
    for i in range(1,31):
        try:
            angles = walking_data['g' + str(i)]
            clean_inds.append(i)
        except:
            print("MISSING MECH DATA FROM SUBJECT " + str(subject) + " AT g" + str(i) + ". " + ("EXO" if exo else "NO EXO"))
            continue

    ## BIOIMPEDANCE INDEX CHECKING (ANKLE)
    ## pathing to get to the dataset / subject
    base_path = get_path_to_dataset()
    if exo:
        exo_path = 'exo'
    else:
        exo_path = 'base'

    if bioz_to_include == 'both' or bioz_to_include == 'ak':
        bioz_path = base_path + '/bioimpedance/s' + str(subject) + '_' + 'ak' + '_' + exo_path + '.obj'
        print(bioz_path)

        ## opening the data file
        filehandler = open(bioz_path, 'rb')
        data = pickle.load(filehandler)
        filehandler.close()

        ## selecting only the walking data
        # it will always be at index 4
        walking_data = data[4]

        ## Check bioz data for missing entries!!
        for i in range(0,30):
            if (i >= len(walking_data) or np.shape(walking_data[i].BioZ1.Data[0])[0] == 0) and i + 1 in clean_inds:
                clean_inds.remove(i+1)
                print("MISSING BIOZ DATA FROM SUBJECT " + str(subject) + " AT TRIAL " + str(i+1) + ". " + ("EXO" if exo else "NO EXO"))

    ## BIOIMPEDANCE INDEX CHECKING (KNEE)
    if bioz_to_include == 'both' or bioz_to_include == 'kn':
        bioz_path = base_path + '/bioimpedance/s' + str(subject) + '_' + 'kn' + '_' + exo_path + '.obj'
        print(bioz_path)

        ## opening the data file
        filehandler = open(bioz_path, 'rb')
        data = pickle.load(filehandler)
        filehandler.close()

        ## selecting only the walking data
        # it will always be at index 4
        walking_data = data[4]

        ## Check knee bioz data for missing entries!!
        for i in range(0,30):
            if (i >= len(walking_data) or np.shape(walking_data[i].BioZ1.Data[0])[0] == 0) and i + 1 in clean_inds:
                clean_inds.remove(i+1)
                print("MISSING BIOZ DATA FROM SUBJECT " + str(subject) + " AT TRIAL " + str(i+1) + ". " + ("EXO" if exo else "NO EXO"))
    
    return clean_inds



## ak_or_kn False -> use knee
## ak_or_kn True -> use ankle
def get_bioz_from_subject(subject: int, clean_inds=[], exo=False, ak_or_kn=False):

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
        if i+1 not in clean_inds:
            continue
        print(np.shape(walking_data[i].BioZ1.Data[0]))
    for i in range(30):
        if i+1 not in clean_inds:
            continue
        bioz_5k_resistance.append(scipy.signal.resample(walking_data[i].BioZ1.Data[0][:,1], 6000))
        bioz_5k_reactance.append(scipy.signal.resample(walking_data[i].BioZ1.Data[0][:,2], 6000))
        bioz_100k_resistance.append(scipy.signal.resample(walking_data[i].BioZ1.Data[1][:,1], 6000))
        bioz_100k_reactance.append(scipy.signal.resample(walking_data[i].BioZ1.Data[1][:,2], 6000))

    return bioz_5k_resistance, bioz_5k_reactance, bioz_100k_resistance, bioz_100k_reactance


def get_biomech_from_subject(subject: int, clean_inds=[], exo=False):

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

    for i in range(1,31):
        if i not in clean_inds:
            continue
        angles = walking_data['g' + str(i)]
        hip_flexion_r.append(angles['hip_flexion_r'])
        hip_flexion_l.append(angles['hip_flexion_l'])
        hip_adduction_r.append(angles['hip_adduction_r'])
        hip_adduction_l.append(angles['hip_adduction_l'])
        knee_angle_r.append(angles['knee_angle_r'])
        knee_angle_l.append(angles['knee_angle_l'])
        ankle_angle_r.append(angles['ankle_angle_r'])
        ankle_angle_l.append(angles['ankle_angle_l'])

    biggie = np.concatenate(ankle_angle_r)
    plt.plot(biggie)
    plt.show(block=True)
    1
    return (hip_flexion_r, hip_flexion_l, hip_adduction_r, hip_adduction_l, knee_angle_r, knee_angle_l, ankle_angle_r, ankle_angle_l)






def main():
    base_path = get_path_to_dataset()
    subject = 11
    ## ANKLE BIOIMPEDANCE DEVICE WORN ON RIGHT LEG
    ## KNEE BIOIMPEDANCE DEVICE WORN ON LEFT LEG

    bioz_keys = ['ankle_bioz_5k_resistance', 'ankle_bioz_5k_reactance', 'ankle_bioz_100k_resistance', 'ankle_bioz_100k_reactance', 'knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance']
    mech_keys = ['hip_flexion_r', 'hip_flexion_l', 'hip_adduction_r', 'hip_adduction_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l']

    dtype = []
    for key in bioz_keys:
        dtype.append((key, 'f8'))

    for key in mech_keys:
        dtype.append((key, 'f8'))

    #get_biomech_from_subject(subject)
    clean_inds = get_clean_inds(subject, exo=True)
    mech_vals = get_biomech_from_subject(subject, clean_inds, exo=True)
    ankle_bioz = get_bioz_from_subject(subject, clean_inds, ak_or_kn=True, exo=True)
    knee_bioz = get_bioz_from_subject(subject, clean_inds, ak_or_kn=False, exo=True)
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
    # fig, axs = plt.subplots(nrows=8)
    # axs[0].plot(recarray['ankle_bioz_5k_resistance'][0][:500])
    # axs[1].plot(recarray['ankle_angle_r'][0][:500])
    # axs[2].plot(recarray['ankle_bioz_5k_resistance'][0][1000:1500])
    # axs[3].plot(recarray['ankle_angle_r'][0][1000:1500])
    # axs[4].plot(recarray['ankle_bioz_5k_resistance'][0][2000:2500])
    # axs[5].plot(recarray['ankle_angle_r'][0][2000:2500])
    # axs[6].plot(recarray['ankle_bioz_5k_resistance'][0][3000:3500])
    # axs[7].plot(recarray['ankle_angle_r'][0][3000:3500])
    # plt.show(block=True)
    ## ANKLE BIOIMPEDANCE DEVICE WORN ON RIGHT LEG
    ## KNEE BIOIMPEDANCE DEVICE WORN ON LEFT LEG
    # fig, axs = plt.subplots(nrows=2)
    # axs[0].plot(recarray['ankle_bioz_5k_resistance'][0])
    # axs[1].plot(recarray['ankle_angle_r'][0])
    # plt.show(block=True)

    # plt.scatter(recarray['ankle_angle_r'][0], recarray['ankle_bioz_5k_resistance'][1], alpha=0.01)
    # plt.show(block=True)

    np.save(base_path + '/resampled_exo_' + str(subject) + '.npy', recarray)
    # for key in mech_keys:
    #     dtype.append((key, 'f8'))
    dumy=1

    ## ANKLE BIOIMPEDANCE DEVICE WORN ON RIGHT LEG
    ## KNEE BIOIMPEDANCE DEVICE WORN ON LEFT LEG

if __name__ == '__main__':
    main()