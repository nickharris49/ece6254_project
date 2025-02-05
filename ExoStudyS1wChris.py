import matplotlib.pyplot as plt
import datetime
import pickle
import matplotlib
import re
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
from scipy.io import loadmat
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
from Utilities.Dataset import parse_file_for_data_sets
import matplotlib as mpl
import pandas as pd

if "chjnichols" in os.getcwd():
    RootDir="/Users/chjnichols/Library/Mobile Documents/com~apple~CloudDocs/"
    Amro_data_path = "/Users/chjnichols/Dropbox (GaTech)/Final Participant Data/"
else:
    RootDir="/Users/samer/Library/Mobile Documents/com~apple~CloudDocs/Work/"
    Amro_data_path = RootDir+"MotionCaptureSharedFolder/Final Participant Data/" # samer edit this if incorrect


#### define which timings file to pull from

file_choice = "s8_ak_base"

dataframe1 = pd.read_csv(RootDir + "MotionCaptureSharedFolder/Vicon_sync_timings/" + file_choice + "_timings.csv")
####

if file_choice[0] == 's':
    # load Amro file structure
    subj_number = re.findall(r'\d+', file_choice)
    if dataframe1["class"][0] == 0: #base day
        fatigue_class = "No Exo"
        fatigue_char = 'NoExo'
        Exo = False
    else:# exo
        fatigue_class = "Exo"
        fatigue_char = 'Exo'
        Exo = True

    if os.path.exists(Amro_data_path + "fatigue_" + str(subj_number[0]) + "/" + fatigue_class + "/final_dat.mat"): 
        biomech_data = loadmat(Amro_data_path + "fatigue_" + str(subj_number[0]) + "/" + fatigue_class + "/final_dat.mat")
        #indexing this is a pain in the ass, very large matlab struct.
        # the following [0,0] to each dict entry specifies the contents of that dict.
        # follow last [0,0] with another [x,y] for position in 6000 by 30 array

        # processing stride percentage kept
        stride_perc = biomech_data['final_dat'][fatigue_char][0,0]['strideperc'][0,0][0]

        # mean power frequency from EMG, and FI.
        # unsure where these go, most likely paired with MVCs. just saving all 8 to each snapshot
        MPF = biomech_data['final_dat'][fatigue_char][0,0]['MPF'][0,0][0]
        MPFperc = biomech_data['final_dat'][fatigue_char][0,0]['MPFperc'][0,0][0]

        FI = biomech_data['final_dat'][fatigue_char][0,0]['FI'][0,0][0]
        FIperc = biomech_data['final_dat'][fatigue_char][0,0]['FIperc'][0,0][0]

        # Rate of percieved exertion. recorded at minutes 1,5,10,15,20,25, and 30
        RPE = biomech_data['final_dat'][fatigue_char][0,0]['RPE'][0,0][0]

        # angle, velocity, torque, power data. inside is 6000 x 30 array (6000 datapoints, 30 minutes)
        #   hip_flexion_r # dont really care about
        #   knee_angle_r
        #   ankle_angle_r
        #   hip_adduction_r # dont really care about

        # angle data
        angle_mean = biomech_data['final_dat'][fatigue_char][0,0]['angles'][0,0]['mean'][0,0]
        angle_sd = biomech_data['final_dat'][fatigue_char][0,0]['angles'][0,0]['sd'][0,0]

        # angular velocity data
        vel_mean = biomech_data['final_dat'][fatigue_char][0,0]['velocity'][0,0]['mean'][0,0]
        vel_sd = biomech_data['final_dat'][fatigue_char][0,0]['velocity'][0,0]['sd'][0,0]

        # torque data
        torque_mean = biomech_data['final_dat'][fatigue_char][0,0]['torque'][0,0]['mean'][0,0]
        torque_sd = biomech_data['final_dat'][fatigue_char][0,0]['torque'][0,0]['sd'][0,0]

        # power data
        power_mean = biomech_data['final_dat'][fatigue_char][0,0]['power'][0,0]['mean'][0,0]
        power_sd = biomech_data['final_dat'][fatigue_char][0,0]['power'][0,0]['sd'][0,0]

        # emg data. broken up into emg and ratios
        #   ratios has
        #       mvc 7x30 double, 7 emgs and 30 minutes
        #       mvcsd
        #       walk
        #       walksd
        #       powermvc
        #       powermvcsd
        #       powerwalk
        #       powerwalksd
        #   emg has mvc and walk - what emg was normalized against, the first min of walking or mvcs.
        #       broken into waveforms of mean and sd as 6000 x 8 x 30 arrays,
        #       6000 datapoints, 8 emgs, and 30 minutes

        #emg waveforms
        emg_waveform_mvc_mean = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['emg'][0,0]['mvc'][0,0]['mean'][0,0]
        emg_waveform_mvc_sd = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['emg'][0,0]['mvc'][0,0]['sd'][0,0]
        emg_waveform_walk_mean = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['emg'][0,0]['walk'][0,0]['mean'][0,0]
        emg_waveform_walk_sd = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['emg'][0,0]['walk'][0,0]['sd'][0,0]

        # emg ratios
        emg_ratios_mvc_mean = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['ratios'][0,0]['mvc'][0,0]
        emg_ratios_mvc_sd = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['ratios'][0,0]['mvcsd'][0,0]
        emg_ratios_walk_mean = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['ratios'][0,0]['walk'][0,0]
        emg_ratios_walk_sd = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['ratios'][0,0]['walksd'][0,0]
        emg_power_mvc_mean = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['ratios'][0,0]['powermvc'][0,0]
        emg_power_mvc_sd = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['ratios'][0,0]['powermvcsd'][0,0]
        emg_power_walk_mean = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['ratios'][0,0]['powerwalk'][0,0]
        emg_power_walk_sd = biomech_data['final_dat'][fatigue_char][0,0]['emg'][0,0]['ratios'][0,0]['powerwalksd'][0,0]

else:
    # load Qingyi file structure 
    biomech_data = loadmat(RootDir + "MotionCaptureSharedFolder/Q_Vest_Study_Biomech_Data_Averaged.mat")
    subj_number = re.findall(r'\d+', file_choice)
    if dataframe1["class"][0] == 0: #base day
        fatigue_class = "base"
        fatigue_char = 'base'
        Exo = False
    else:# vest
        fatigue_class = "vest"
        fatigue_char = 'vest'
        Exo = False
    biomech_data = biomech_data['Q_Vest_Study_Data_Averaged'][0,0]['q'+str(subj_number[0])][0,0][fatigue_class][0,0]
        ### what is in Amro data but not q datasets
        # power
        # velocity
        # EMG mvc normed
        # RPE -> Q has this data, but will need to be added


# SS[4][0].

FilePath = RootDir+dataframe1["file_path"][0]
ParserType = dataframe1["device_type"][0]
FileIndexs = dataframe1["file_index"].to_numpy()
DataTypes = dataframe1["DataType"].to_numpy()
NumSnapshots = len(dataframe1["file_path"])
StartTime = dataframe1["start_time_ms"].to_numpy()
StopTime = dataframe1["end_time_ms"].to_numpy()
IndexFromCSV = dataframe1["index"].to_numpy()

if ParserType == 0:
    Datasets = parse_file_for_data_sets(FilePath)
elif ParserType == 1:
    Datasets = AdvancedLoggingDataset(FilePath)

SS = []
for i in range(np.max(DataTypes)+1):
    SS.append([])


for i in range(NumSnapshots):
    if ParserType == 0:
        Dataset = Datasets[FileIndexs[i]]
        BioZ1 = copy.deepcopy(Dataset.BioZ)
    elif ParserType == 1:
        Dataset = Datasets
        BioZ1 = copy.deepcopy(Dataset.BioZ1)
    BioZ0 = None
    IMU = copy.deepcopy(Dataset.IMU)
    # BioZ1.plot()
    DeviceTurnedOn = datetime.datetime(2023,2,9,15,0,0)
    TempSS = KneeDataSnapShot(BioZ0,BioZ1, IMU, DeviceTurnedOn,StartTime[i]/3600000 ,StopTime[i]/3600000,0,0,0, "Right", "")
    if (DataTypes[i] != 0):
        print("Segmenting Datatype:",DataTypes[i],"Index:", i)
        try:
            # TempSS.GetStepsFromTimeWindowStartingHeelStrike(StartTime=TempSS.IMU.Data[0,0],
            #                                                 StopTime=TempSS.IMU.Data[-1,0],
            #                                                 MidSwingGyroIndex=np.abs(dataframe1['gyro_select'][i]),
            #                                                 MidSwingGyroSign=np.sign(dataframe1['gyro_select'][i]),
            #                                                 mid_swing_peak_height=0.1,
            #                                                 mid_distance_between_peaks=50,
            #                                                 HeelStrikeAcclIndex=8,
            #                                                 HeelStrikeSign=1,
            #                                                 HeelStrikeHeight= 1.0)
            # TempSS.GetStepsFromWindow(np.abs(dataframe1['gyro_select'][i]),np.sign(dataframe1['gyro_select'][i]), TempSS.IMU.Data[0,0], TempSS.IMU.Data[-1,0],
            #                                  peak_height=0.1, distance_between_peaks=50)
            TempSS.GetStepsFromWindowStartAtEndOfForwardSwing(StartTime=TempSS.IMU.Data[0,0],
                                                            StopTime=TempSS.IMU.Data[-1,0],
                                                            MidSwingGyroIndex=np.abs(dataframe1['gyro_select'][i]),
                                                            MidSwingGyroSign=np.sign(dataframe1['gyro_select'][i]),
                                                            mid_swing_peak_height=0.1,
                                                            mid_distance_between_peaks=40,
                                                            heel_strike_peak_height=0,
                                                            max_step_samples = 100)
            # for knee device make Heelstrike sign -1 and axis 8 and distance between peaks 50 HeelStrikeHeigh 1.0
            # for ankle device make Heelstrike sign 1 and axis 1 and distance between peaks 30 Heel StrikeHeight 0.5
#           for knee device with one IMU sign is -9, axis 9 and .5 height
            TempSS.ResampleStepData(50)
        except Exception as e:
            print(e)
    TempSS.MotionCapIndex = IndexFromCSV[i]
    TempSS.DataType = dataframe1["class"][i]
    SS[DataTypes[i]].append(copy.deepcopy(TempSS))

RPEIndexArray = np.array([0,4,9,14,19,24,29])

if file_choice[0] == 's':
    if os.path.exists(Amro_data_path + "fatigue_" + str(subj_number[0]) + "/" + fatigue_class + "/final_dat.mat"):
        # process Amro Data
        for ii in np.arange(0,len(SS[4])):
                MotionCapData = dict()
                # angle, velocity, torque, power data. inside is 6000 x 30 array (6000 datapoints, 30 minutes)
                #   hip_flexion_r # dont really care about
                #   knee_angle_r
                #   ankle_angle_r
                #   hip_adduction_r # dont really care about
                if Exo:
                    Torque = dict()
                    Torque['tot'] = torque_mean["ankle_angle_r"][0,0]['tot'][0,0][:,ii]
                    Torque['bio'] = torque_mean["ankle_angle_r"][0,0]['bio'][0,0][:,ii]
                    Torque['exo'] = torque_mean["ankle_angle_r"][0,0]['exo'][0,0][:,ii]

                    MotionCapData["torque_ankle_mean"] = copy.deepcopy(Torque)
                else:
                    MotionCapData["torque_ankle_mean"] = torque_mean["ankle_angle_r"][0,0][:,ii]

                MotionCapData["torque_knee_mean"] =  torque_mean["knee_angle_r"][0,0][:,ii]
                MotionCapData["torque_hip_mean"] = torque_mean["hip_flexion_r"][0, 0][:, ii]

                if Exo:
                    Torque = dict()
                    Torque['tot'] = torque_sd["ankle_angle_r"][0,0]['tot'][0,0][:,ii]
                    Torque['bio'] = torque_sd["ankle_angle_r"][0,0]['bio'][0,0][:,ii]
                    Torque['exo'] = torque_sd["ankle_angle_r"][0,0]['exo'][0,0][:,ii]

                    MotionCapData["torque_ankle_std"] = copy.deepcopy(Torque)
                else:
                    MotionCapData["torque_ankle_std"] = torque_sd["ankle_angle_r"][0, 0][:, ii]

                MotionCapData["torque_knee_std"] =  torque_sd["knee_angle_r"][0,0][:,ii]
                MotionCapData["torque_hip_std"] = torque_sd["hip_flexion_r"][0, 0][:, ii]

                MotionCapData["angle_ankle_mean"] = angle_mean["ankle_angle_r"][0,0][:,ii]
                MotionCapData["angle_knee_mean"] =  angle_mean["knee_angle_r"][0,0][:,ii]
                MotionCapData["angle_hip_mean"] = angle_mean["hip_flexion_r"][0, 0][:, ii]

                MotionCapData["angle_ankle_std"] = angle_sd["ankle_angle_r"][0,0][:,ii]
                MotionCapData["angle_knee_std"] =  angle_sd["knee_angle_r"][0,0][:,ii]
                MotionCapData["angle_hip_std"] =  angle_sd["hip_flexion_r"][0,0][:,ii]

                if Exo:
                    Power = dict()
                    Power['tot'] = power_mean["ankle_angle_r"][0,0]['tot'][0,0][:,ii]
                    Power['bio'] = power_mean["ankle_angle_r"][0,0]['bio'][0,0][:,ii]
                    Power['exo'] = power_mean["ankle_angle_r"][0,0]['exo'][0,0][:,ii]

                    MotionCapData["power_ankle_mean"] = copy.deepcopy(Power)
                else:
                    MotionCapData["power_ankle_mean"] = power_mean["ankle_angle_r"][0, 0][:, ii]

                MotionCapData["power_knee_mean"] =  power_mean["knee_angle_r"][0,0][:,ii]
                MotionCapData["power_hip_mean"] = power_mean["hip_flexion_r"][0, 0][:, ii]


                if Exo:
                    Power = dict()
                    Power['tot'] = power_sd["ankle_angle_r"][0,0]['tot'][0,0][:,ii]
                    Power['bio'] = power_sd["ankle_angle_r"][0,0]['bio'][0,0][:,ii]
                    Power['exo'] = power_sd["ankle_angle_r"][0,0]['exo'][0,0][:,ii]

                    MotionCapData["power_ankle_std"] = copy.deepcopy(Power)
                else:
                    MotionCapData["power_ankle_std"] = power_sd["ankle_angle_r"][0, 0][:, ii]

                MotionCapData["power_knee_std"] =  power_sd["knee_angle_r"][0,0][:,ii]
                MotionCapData["power_hip_std"] = power_sd["hip_flexion_r"][0, 0][:, ii]

                MotionCapData["velocity_ankle_mean"] = vel_mean["ankle_angle_r"][0,0][:,ii]
                MotionCapData["velocity_knee_mean"] =  vel_mean["knee_angle_r"][0,0][:,ii]
                MotionCapData["velocity_hip_mean"] = vel_mean["hip_flexion_r"][0,0][:, ii]

                MotionCapData["velocity_ankle_std"] = vel_sd["ankle_angle_r"][0,0][:,ii]
                MotionCapData["velocity_knee_std"] =  vel_sd["knee_angle_r"][0,0][:,ii]
                MotionCapData["velocity_hip_std"] = vel_sd["hip_flexion_r"][0,0][:, ii]

                if ii in RPEIndexArray:
                    MotionCapData["rpe"] = RPE[np.where(ii==RPEIndexArray)[0]]
                    MotionCapData["mpf"] = MPF[np.where(ii==RPEIndexArray)[0]]
                else:
                    MotionCapData["rpe"] = np.nan
                    MotionCapData["mpf"] = np.nan

                MVC = dict()
                MVC["EMG_waveform_mean"] = emg_waveform_mvc_mean[:,:,ii]
                MVC["EMG_waveform_std"] = emg_waveform_mvc_sd[:,:,ii]
                MVC["EMG_ratio_mean"] = emg_ratios_mvc_mean[:,ii]
                MVC["EMG_ratio_std"] = emg_ratios_mvc_sd[:,ii]
                MVC["EMG_power_mean"] = emg_power_mvc_mean[:,ii]
                MVC['EMG_power_std'] = emg_power_mvc_sd[:,ii]


                Walk = dict()
                Walk["EMG_waveform_mean"] = emg_waveform_walk_mean[:,:,ii]
                Walk["EMG_waveform_std"] = emg_waveform_walk_sd[:,:,ii]
                Walk["EMG_ratio_mean"] = emg_ratios_walk_mean[:,ii]
                Walk["EMG_ratio_std"] = emg_ratios_walk_sd[:,ii]
                Walk["EMG_power_mean"] = emg_power_walk_mean[:,ii]
                Walk['EMG_power_std'] = emg_power_walk_sd[:,ii]

                MotionCapData["walk"] = Walk
                MotionCapData["mvc"] = MVC

                SS[4][ii].MotionCapData = copy.deepcopy(MotionCapData)


elif file_choice[0] == 'q':
    QIndexArray = np.array([0,2,4,6,8,10,12])
    # process Q data
    # process Amro Data
    for ii in np.arange(0,7):
        MotionCapData = dict()
        # angle, velocity, torque, power data. inside is 6000 x 30 array (6000 datapoints, 30 minutes)
        #   hip_flexion_r # dont really care about
        #   knee_angle_r
        #   ankle_angle_r
        #   hip_adduction_r # dont really care about

        MotionCapData["torque_ankle_mean"] = biomech_data["torque"][0,0]["ankle"][:,ii]
        MotionCapData["torque_knee_mean"] =  biomech_data["torque"][0,0]["knee"][:,ii]
        MotionCapData["torque_hip_mean"] = biomech_data["torque"][0,0]["hip"][:,ii]

        MotionCapData["torque_ankle_std"] = biomech_data["torque"][0,0]["ankle_sd"][:,ii]
        MotionCapData["torque_knee_std"] =  biomech_data["torque"][0,0]["knee_sd"][:,ii]
        MotionCapData["torque_hip_std"] = biomech_data["torque"][0,0]["hip_sd"][:,ii]

        MotionCapData["angle_ankle_mean"] =  biomech_data["angle"][0,0]["ankle"][:,ii]
        MotionCapData["angle_knee_mean"] =  biomech_data["angle"][0,0]["knee"][:,ii]
        MotionCapData["angle_hip_mean"] = biomech_data["angle"][0,0]["hip"][:, ii]

        MotionCapData["angle_ankle_std"] = biomech_data["angle"][0,0]["ankle_sd"][:,ii]
        MotionCapData["angle_knee_std"] =  biomech_data["angle"][0,0]["knee_sd"][:,ii]
        MotionCapData["angle_hip_std"] =  biomech_data["angle"][0,0]["hip_sd"][:,ii]



        Walk = dict()
        Walk["EMG_waveform_mean"] = biomech_data["emg"][:,:,ii]
        Walk["EMG_waveform_std"] = biomech_data["emg_sd"][:,:,ii]

        MotionCapData["walk"] = Walk

        SS[4][QIndexArray[ii]].MotionCapData = copy.deepcopy(MotionCapData)

plt.close('all')


# if everything looks good, save to pickle file

filehandler = open(RootDir + "MotionCaptureSharedFolder/pickles/" + file_choice + ".obj", 'wb')
pickle.dump(SS, filehandler)
filehandler.close()

# for plotting individual gyro segments to make sure alignment was successful
for i in range(len(SS[4])):
    fig, ax = plt.subplots()
    try:
        for j in range(len(SS[4][i].IMUStepsData)):
            # ax.plot(SS[4][i].IMUStepsData[j][:,0], SS[4][i].IMUStepsData[j][:,np.abs(dataframe1['gyro_select'][j])])
            ax.plot(SS[4][i].IMUStepsData[j][:,0]-SS[4][i].IMUStepsData[j][0,0],SS[4][i].IMUStepsData[j][:,np.abs(dataframe1['gyro_select'][j])])
            ax.set_title("minute " + str(i + 1))
    except Exception as e:
        print(e)
plt.show()

fig, ax = plt.subplots(2)
for i in range(len(SS[4])):
    try:
        ax[0].plot(SS[4][i].ResampledBioZStepData[:,:,0].mean(axis=0)-SS[4][i].ResampledBioZStepData[:,:,0].mean(axis=0)[0],color='blue', alpha=i/15)
        ax[1].plot(SS[4][i].ResampledBioZStepData[:,:,1].mean(axis=0)-SS[4][i].ResampledBioZStepData[:,:,1].mean(axis=0)[0],color='blue', alpha=i/15)
        ax[0].set_title('R waveform 5kHz')
        ax[1].set_title('R waveform 100kHz')
    except Exception as e:
        print(e)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(5)
for i in range(len(SS)):
    try:
        a = i/15
        ax[0].plot(SS[4][i].ResampledBioZStepData[:,:,0].mean(axis=0)[:]-SS[4][i].ResampledBioZStepData[:,:,0].mean(axis=0)[0],color='blue', alpha=a)
        ax[1].plot(SS[4][i].ResampledBioZStepData[:,:,1].mean(axis=0)[:]-SS[4][i].ResampledBioZStepData[:,:,1].mean(axis=0)[0],color='blue', alpha=a)
        ax[0].set_title('R waveform 5kHz')
        ax[1].set_title('R waveform 100kHz')
        ax[2].plot(SS[4][i].MotionCapData['torque_knee_mean'][:],color='blue', alpha=a)
        ax[2].set_title('torque')
        ax[3].plot(SS[4][i].MotionCapData['angle_knee_mean'][:],color='blue', alpha=a)
        ax[3].set_title('angle')
        ax[4].plot(SS[4][i].MotionCapData['power_knee_mean'][:],color='blue', alpha=a)
        ax[4].set_title('power')
    except Exception as e:
        print(e)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(2)
for i in range(len(SS[4])):
    try:
        ax[0].scatter(SS[4][i].MotionCapIndex,SS[4][i].ResampledBioZStepData[:,:,0].mean(axis=0)[0])
        ax[1].scatter(SS[4][i].MotionCapIndex,SS[4][i].ResampledBioZStepData[:,:,1].mean(axis=0)[0])
        ax[0].set_title('heelstrike R 5kHz')
        ax[1].set_title('heelstrike R 100kHz')
    except Exception as e:
        print(e)
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots()
# for i in range(len(SS[4])):
#     try:
#         ax.scatter(SSIndexForPlot[4][i],SS[4][i].ResampledBioZStepData[:,:,0].mean(axis=0)[0]/SS[4][i].ResampledBioZStepData[:,:,1].mean(axis=0)[0])
#     except Exception as e:
#         print(e)
# plt.tight_layout()
# plt.show()





# if everything looks good, save to pickle file
#
filehandler = open(RootDir + "MotionCaptureSharedFolder/pickles/" + file_choice + ".obj", 'wb')
pickle.dump(SS, filehandler)
filehandler.close()