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



def main():
    ## pathing stuff
    ## UPDATE THIS TO REFLECT YOUR OS
    if os.getlogin() == 'nicholasharris':
        base_path = '/Users/nicholasharris/OneDrive - Georgia Institute of Technology/BioZ Dataset'
    # elif -> add here
    
    path_end = '/biomechanics/fatigue_1/No Exo/pre_dat.mat'
    full_path_mech = base_path + path_end

    path_end = '/bioimpedance/s1_ak_base.obj'
    full_path_bioz = base_path + path_end

    eng = meng.start_matlab()
    data = eng.load(full_path_mech)
    ankle_angle = data['angles']['main']['walk']

    plt.plot(ankle_angle['g' + str(1)]['ankle_angle_r'])
    plt.show(block=True)

    filehandler = open(full_path_bioz, 'rb')
    s1_ak_base = pickle.load(filehandler)
    filehandler.close()

    ## IDEAL DATASET
    # bioimpedance
    # biomechanics
    # just the 30 minutes, chunked into 60s windows
    # 
    
    bioz_walking = s1_ak_base[4]
    for i in range(30):
        fig, axs = plt.subplots(nrows=2)
        frame_1 = bioz_walking[i]
        frame_1_BioZ = np.transpose(frame_1.BioZ1.Data[0])
        ## I want to figure out the sampling freq of bioz and of biomech and compare them
        # biomech_freq = 100Hz
        # bioimp_freq = 31.25 Hz
        biomech_frames = len(ankle_angle['g' + str(i+1)]['ankle_angle_r'])
        biomech_freq = biomech_frames / 60
        bioimp_frames = len(frame_1_BioZ[1])
        bioimp_freq = bioimp_frames / 60
        axs[0].plot(ankle_angle['g' + str(i+1)]['ankle_angle_r'])
        axs[1].plot(frame_1_BioZ[1])
        plt.show(block=True)


    bioz_walking = s1_ak_base[4]
    frame_1 = bioz_walking[0]
    frame_1_BioZ = np.transpose(frame_1.BioZ1.Data[0])
    plt.plot(frame_1_BioZ[0])
    plt.show()
    plt.plot(frame_1_BioZ[1])
    plt.show()
    plt.plot(frame_1_BioZ[2])
    plt.show()

    StepsPerWindow = 30
    Windows = 15
    SubjectToUse = 0
    AverageStepsPerWindow = []
    for i in range(Windows):
        NormalizedSteps = (SnapshotsToUse[SubjectToUse].ResampledBioZStepData[i*StepsPerWindow :(i+1)*StepsPerWindow,:,0].transpose() - SnapshotsToUse[SubjectToUse].ResampledBioZStepData[i*StepsPerWindow :(i+1)*StepsPerWindow,0,0]).transpose()
        AverageStepsPerWindow.append(NormalizedSteps.mean(axis=0))

    weights = np.arange(1, Windows+1)

    AverageStepsPerWindow = np.array(AverageStepsPerWindow)
    fig, ax = plt.subplots()
    for i in range(Windows):
        ax.plot(AverageStepsPerWindow[i], c="blue", alpha=(i/Windows))
    plt.tight_layout()
    plt.show()
    bitch = 1
    



if __name__ == '__main__':
    main()