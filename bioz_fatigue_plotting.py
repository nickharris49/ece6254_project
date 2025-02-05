import matplotlib.pyplot as plt
import datetime
import pickle
import matplotlib
import matplotlib.colors as mcolors

import pandas as pd
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import os
from scipy import signal
from scipy import stats
from Utilities.Dataset import parse_file_for_data_sets
from Utilities.FileParser import parse_file
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from Utilities.Dataset import parse_file_for_data_sets
from sklearn import decomposition
from DataAnalysis.KneeDataAnalysis import *
import pickle
import copy
from Utilities.AdvancedLoggingParser import *
from Utilities.Dataset import parse_file_for_data_sets
from Utilities.AdvancedLoggingParser import *
import matplotlib as mpl
import pandas as pd

def list_files(directory, search_string):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if search_string in filename:
                files.append(os.path.join(root, filename))
    return files

if "chjnichols" in os.getcwd():
    RootDir="/Users/chjnichols/Library/Mobile Documents/com~apple~CloudDocs/"
else:
    RootDir="/Users/samer/Library/Mobile Documents/com~apple~CloudDocs/Work/"

### use this if want to generate for individual subject ###
# obj_files = RootDir + "MotionCaptureSharedFolder/pickles/" + "s1_ak_base"

# for study_char in ['s','q']:
#     for sub_number in np.arange(1,12):
#         sub_study_identifier = study_char + str(sub_number) + "_"

#         obj_files = list_files(RootDir+"MotionCaptureSharedFolder/pickles/", sub_study_identifier)
#         fig, ax = plt.subplots(2,figsize=(9,7)) # this is plotting 5k bioz from ankle and knee wiht and without Exo
#         fig1, ax1 = plt.subplots(2, figsize=(9,7))
#         for obj_file in obj_files:
#             filehandler = open(obj_file, 'rb')
#             SSdata = pickle.load(filehandler)
#             filehandler.close()

#             # specify which axis to plot in
#             if "kn" in obj_file: # knee data
#                 joint_ax = 1
#             else: # ankle data
#                 joint_ax = 0

#             # specify plot color
#             if "exo" in obj_file: # exo data
#                 class_color = 'tab:green'
#                 class_label = 'Exo Assistance'
#             elif "vest" in obj_file: # vest data
#                 class_color = 'tab:orange'
#                 class_label = 'Weighted Vest'
#             else: # baseline data
#                 class_color = 'tab:blue'
#                 class_label = 'Unassisted'

#             # catch for s1 device falling off
#             if "s1_kn_base" in obj_file: 
#                 plot_length = 15
#             else:
#                 plot_length = len(SSdata[4])

#             for i in range(plot_length):
#                 try:
#                     if i == 0:
#                         ax[joint_ax].scatter(i, 100.0*(SSdata[4][i].ResampledBioZStepData[:, :, 0].mean(axis=0)[0]-SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0])/SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0],color=class_color, label=class_label)
#                     else:
#                         ax[joint_ax].scatter(i, 100.0*(SSdata[4][i].ResampledBioZStepData[:, :, 0].mean(axis=0)[0]-SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0])/SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0],color=class_color)
#                 except Exception as e:
#                     print(e)

#             ax[0].set_ylabel('Ankle Resistance 5 kHz')  
#             ax[1].set_ylabel('Knee Resistance 5 kHz')
#             ax[1].legend()
#             ax[0].set_title(sub_study_identifier[:-1] + ' percent change in bioimpedance')
#             ax[1].set_xlabel('minute')
#             plt.tight_layout()
#             fig.savefig(RootDir+"MotionCaptureSharedFolder/figures/" + sub_study_identifier + "hs_5.png", format="png")


#             for i in range(plot_length):
#                 try: # [which step, 50 samples of resampeld step, 0 or 1 frequency ]
#                     if i == 0:
#                         ax1[joint_ax].scatter(i, 100.0*(SSdata[4][i].ResampledBioZStepData[:, :, 1].mean(axis=0)[0]-SSdata[4][0].ResampledBioZStepData[:, :, 1].mean(axis=0)[0])/SSdata[4][0].ResampledBioZStepData[:, :, 1].mean(axis=0)[0],color=class_color, label=class_label)
#                     else:
#                         ax1[joint_ax].scatter(i, 100.0*(SSdata[4][i].ResampledBioZStepData[:, :, 1].mean(axis=0)[0]-SSdata[4][0].ResampledBioZStepData[:, :, 1].mean(axis=0)[0])/SSdata[4][0].ResampledBioZStepData[:, :, 1].mean(axis=0)[0],color=class_color)
               
#                 except Exception as e:
#                     print(e)

#             ax1[0].set_ylabel('Ankle Resistance 100 kHz')
#             ax1[1].set_ylabel('Knee Resistance 100 kHz')
#             ax1[1].legend()
#             ax1[0].set_title(sub_study_identifier[:-1] + ' percent change in bioimpedance')
#             ax1[1].set_xlabel('minute')
#             plt.tight_layout()
#             fig1.savefig(RootDir+"MotionCaptureSharedFolder/figures/" + sub_study_identifier + "hs_100.png", format="png")
#     plt.show()


### 5 by 1 plot of 5k waveform, 100k waveform change over time with angle, torque and emg change in time ###


for study_char in ['s','q']:
    for sub_number in np.arange(8,12):
        sub_study_identifier = study_char + str(sub_number) + "_"

        obj_files = list_files(RootDir+"MotionCaptureSharedFolder/pickles/", sub_study_identifier)

        fig, ax = plt.subplots(6,1,figsize=(9,7)) # ankle
        # plt.tight_layout()
        fig1, ax1 = plt.subplots(6,1,figsize=(9,7)) # knee
        # plt.tight_layout()
        for obj_file in obj_files:
            filehandler = open(obj_file, 'rb')
            SSdata = pickle.load(filehandler)
            filehandler.close()

            # specify plot color
            if "exo" in obj_file: # exo data
                class_color = 'tab:green'
                # cmap = plt.get_cmap("")
                class_label = 'Exo Assistance'
            elif "vest" in obj_file: # vest data
                class_color = 'tab:orange'
                class_label = 'Weighted Vest'
            else: # baseline data
                class_color = 'tab:blue'
                class_label = 'Unassisted'

            if study_char == 's':
                minute_plot_indices = [0,4,9,14,19,24,29]
            else:
                minute_plot_indices = [0,2,4,6,8,10,12]

            # catch for s1 device falling off
            if "s1_kn_base" in obj_file: 
                plot_length = 4
            else:
                plot_length = len(minute_plot_indices)

            # specify which axis to plot in
            if "ak" in obj_file: # ankle data
                joint_ax = 0

                for i in range(plot_length):
                    try:
                        if i == plot_length-1:
                            ax[0].plot(SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,0].mean(axis=0)-SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,0].mean(axis=0)[0],color=class_color, alpha=(i+1)/8, label = class_label)
                        else:
                            ax[0].plot(SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,0].mean(axis=0)-SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,0].mean(axis=0)[0],color=class_color, alpha=(i+1)/8)
                        ax[0].set_ylabel("R 5kHz")
                        ax[0].set_title(sub_study_identifier[:-1] + ' Ankle Waveforms')
                        ax[1].plot(SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,1].mean(axis=0)-SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,1].mean(axis=0)[0],color=class_color, alpha=(i+1)/8)
                        ax[1].set_ylabel("R 100kHz")
                        ax[2].plot(-SSdata[4][minute_plot_indices[i]].MotionCapData['angle_ankle_mean'],color=class_color, alpha=(i+1)/8)
                        ax[2].set_ylabel("Joint\nAngle")
                        if "exo" not in obj_file:
                            ax[3].plot(SSdata[4][minute_plot_indices[i]].MotionCapData['torque_ankle_mean'],color=class_color, alpha=(i+1)/8)
                        else:
                            ax[3].plot(SSdata[4][minute_plot_indices[i]].MotionCapData['torque_ankle_mean']['bio'],color=class_color, alpha=(i+1)/8)
                        ax[3].set_ylabel("Joint\nTorque")            
                        ax[4].plot(SSdata[4][minute_plot_indices[i]].MotionCapData['mvc']['EMG_waveform_mean'][:,1],color=class_color, alpha=(i+1)/8)
                        ax[4].set_ylabel("EMG\nSoleus")  
                        ax[5].plot(SSdata[4][minute_plot_indices[i]].MotionCapData['mvc']['EMG_waveform_mean'][:,0],color=class_color, alpha=(i+1)/8)
                        ax[5].set_ylabel("EMG\nTA")    
                        # plt.tight_layout()

                    except Exception as e:
                        print(e)

            else: # knee data
                joint_ax = 1

                for i in range(plot_length):
                    try:
                        if i == plot_length-1:
                            ax1[0].plot(SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,0].mean(axis=0)-SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,0].mean(axis=0)[0],color=class_color, alpha=(i+1)/8, label = class_label)
                        else:
                            ax1[0].plot(SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,0].mean(axis=0)-SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,0].mean(axis=0)[0],color=class_color, alpha=(i+1)/8)
                        ax1[0].set_ylabel("R 5kHz")
                        ax1[0].set_title(sub_study_identifier[:-1] + ' Knee Waveforms')
                        ax1[1].plot(SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,1].mean(axis=0)-SSdata[4][minute_plot_indices[i]].ResampledBioZStepData[:,:,1].mean(axis=0)[0],color=class_color, alpha=(i+1)/8)
                        ax1[1].set_ylabel("R 100kHz")
                        ax1[2].plot(-SSdata[4][minute_plot_indices[i]].MotionCapData['angle_knee_mean'],color=class_color, alpha=(i+1)/8)
                        ax1[2].set_ylabel("Joint\nAngle")
                        ax1[3].plot(SSdata[4][minute_plot_indices[i]].MotionCapData['torque_knee_mean'],color=class_color, alpha=(i+1)/8)
                        ax1[3].set_ylabel("Joint\nTorque")            
                        ax1[4].plot(SSdata[4][minute_plot_indices[i]].MotionCapData['mvc']['EMG_waveform_mean'][:,3],color=class_color,alpha=(i+1)/8)
                        ax1[4].set_ylabel("EMG\nVastus Med")  
                        ax1[5].plot(SSdata[4][minute_plot_indices[i]].MotionCapData['mvc']['EMG_waveform_mean'][:,4],color=class_color, alpha=(i+1)/8)
                        ax1[5].set_ylabel("EMG amp\nRectus Fem")
                        
                        

                    except Exception as e:
                        print(e)  
        # ax[0].legend()                        
        # ax1[0].legend()
        fig.savefig(RootDir+"MotionCaptureSharedFolder/figures/" + sub_study_identifier + "ankle_waveforms.png", format="png")
        fig1.savefig(RootDir+"MotionCaptureSharedFolder/figures/" + sub_study_identifier + "knee_waveforms.png", format="png")

        plt.show()
    

# for creating similar plot to matlab plot of ankle over time compared to knee over time
# exo_30m = []
# base_30m = []
# vest_30m = []
# # fig, ax = plt.subplots(2,figsize=(9,7))

# for i in range(30):
#     exo_30m.append([])
#     base_30m.append([])
#     vest_30m.append([])

# for study_char in ['s','q']:
#     for sub_number in np.arange(1,12):
#         sub_study_identifier = study_char + str(sub_number) + "_kn"

#         obj_files = list_files(RootDir+"MotionCaptureSharedFolder/pickles/", sub_study_identifier)
#         for obj_file in obj_files:
#             filehandler = open(obj_file, 'rb')
#             SSdata = pickle.load(filehandler)
#             filehandler.close()

#             # catch for s1 device falling off
#             if "s1_kn_base" in obj_file: 
#                 plot_length = 15
#             else:
#                 plot_length = len(SSdata[4])

#             for i in range(plot_length):
#                 try:
#                     # specify class
#                     if "exo" in obj_file: # exo data
#                         class_color = 'tab:green'
#                         class_label = 'Exo Assistance'
#                         exo_30m[SSdata[4][i].MotionCapIndex].append(100.0*(SSdata[4][i].ResampledBioZStepData[:, :, 1].mean(axis=0)[0]-SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0])/SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0])

#                     elif "vest" in obj_file: # vest data
#                         class_color = 'tab:orange'
#                         class_label = 'Weighted Vest'
#                         vest_30m[SSdata[4][i].MotionCapIndex].append(100.0*(SSdata[4][i].ResampledBioZStepData[:, :, 1].mean(axis=0)[0]-SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0])/SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0])

#                     else: # baseline data
#                         class_color = 'tab:blue'
#                         class_label = 'Unassisted'
#                         base_30m[SSdata[4][i].MotionCapIndex].append(100.0*(SSdata[4][i].ResampledBioZStepData[:, :, 1].mean(axis=0)[0]-SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0])/SSdata[4][0].ResampledBioZStepData[:, :, 0].mean(axis=0)[0])

#                 except Exception as e:
#                     print(e)


# # Remove NaN values from the array using the mask

# #  Function to replace NaN values with linear interpolation
# def interpolate_nan(arr):
#     nan_indices = np.isnan(arr)  # Find NaN indices
#     non_nan_indices = ~nan_indices  # Find non-NaN indices
#     arr[nan_indices] = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(non_nan_indices), arr[non_nan_indices])  # Interpolate NaN values
#     return arr

# # Apply interpolation
# # interpolate_nan(data)

# base_mean = np.zeros([30])
# base_std = np.zeros([30])
# vest_mean = np.zeros([30])
# vest_std = np.zeros([30])
# exo_mean = np.zeros([30])
# exo_std = np.zeros([30])
# for ii in range(30):
#     base_mean[ii] = np.nanmean(base_30m[ii])
#     base_std[ii] = np.nanstd(base_30m[ii])
#     vest_mean[ii] = np.nanmean(vest_30m[ii])
#     vest_std[ii] = np.nanstd(vest_30m[ii])
#     exo_mean[ii] = np.nanmean(exo_30m[ii])
#     exo_std[ii] = np.nanstd(exo_30m[ii])

# vest_mean_full = interpolate_nan(vest_mean)
# vest_std_full = interpolate_nan(vest_std)

# fig, ax = plt.subplots(figsize=(9,7))
# ax.plot(np.arange(0,30), base_mean + base_std, linestyle= "--",color = 'tab:blue',alpha=0.25)
# ax.plot(np.arange(0,30), base_mean - base_std, linestyle= "--", color = 'tab:blue',alpha=0.25)
# plt.fill_between(np.arange(0,30), base_mean + base_std, base_mean - base_std, color='tab:blue', alpha=0.25)
# ax.plot(np.arange(0,30), base_mean, color = 'tab:blue', label = 'unassisted')


# ax.plot(np.arange(0,30), exo_mean + exo_std, linestyle= "--",color = 'tab:green',alpha=0.25)
# ax.plot(np.arange(0,30), exo_mean - exo_std, linestyle= "--", color = 'tab:green',alpha=0.25)
# plt.fill_between(np.arange(0,30), exo_mean + exo_std, exo_mean - exo_std, color='tab:green', alpha=0.25)
# ax.plot(np.arange(0,30), exo_mean, color = 'tab:green', label = 'exo')

# ax.plot(np.arange(0,30), vest_mean + vest_std, linestyle= "--",color = 'tab:orange',alpha=0.25)
# ax.plot(np.arange(0,30), vest_mean - vest_std, linestyle= "--", color = 'tab:orange',alpha=0.25)
# plt.fill_between(np.arange(0,30), vest_mean + vest_std, vest_mean - vest_std, color='tab:orange', alpha=0.25)
# ax.plot(np.arange(0,30), vest_mean, color = 'tab:orange', label = 'vest')

# ax.set_ylabel('percent change in Resistance 100kHz')
# ax.set_xlabel('minutes')
# fig.savefig(RootDir+"MotionCaptureSharedFolder/figures/averaged_by_minute_100k.png", format="png")

# plt.show()
5
    

    