import matplotlib
import pandas as pd
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import os
from scipy import signal
from scipy import stats
from Utilities.Dataset import parse_file_for_data_sets
from enum import Enum


def find_nearest_time_index(TimeArray, TimeArrayToFind, threshold=None):
    if threshold:
        d = np.abs(TimeArray - TimeArrayToFind[:, np.newaxis])
        return np.any(d <= threshold, axis=0)
    else:
        return np.abs(TimeArray-TimeArrayToFind[:, np.newaxis]).argmin(axis=0)


class WalkingStatus(Enum):
    NA = 0
    Walking = 1
    Resting = 2

class DataWindow:

    def __init__(self, BioZ, IMU, Temperature, BioZIndex, IMUIndex, TemperatureIndex):
        self.BioZ = BioZ
        self.IMU = IMU
        self.Temperature = Temperature
        self.MovementStatus = WalkingStatus(0)
        self.ZeroCrossingsIMUIndex = None
        self.ZeroCrossingsBioZIndex = None
        self.dR5k = None
        self.dR100k = None
        self.h_alpha = None
        self.h_alpha_zero_crossings = None
        self.BioZIndex = BioZIndex
        self.IMUIndex = IMUIndex
        self.TemperatureIndex = TemperatureIndex
        self.Steps = None
    def check_movement_status(self):
        GyroDataHat = signal.savgol_filter(self.IMU[:, 10], 15, 3)
        EnergyX1 = (GyroDataHat* GyroDataHat).sum() / (10000/(2**14))
        GyroDataHat = signal.savgol_filter(self.IMU[:, 4], 15, 3)
        EnergyX2 = (GyroDataHat* GyroDataHat).sum() / (10000/(2**14))
        # used to be EnergyX1>200000
        if (EnergyX1 > (2000/(2**14)) or EnergyX2>(2000/(2**14))):
            self.MovementStatus = 1
        else:
            self.MovementStatus = 2

    def find_zero_crossings(self, ReferenceAxis, tol=None):
        #First smoothen the data
        GyroDataHat = signal.savgol_filter(self.IMU[:, ReferenceAxis], 15, 3)
        self.ZeroCrossingsIMUIndex = np.where(np.diff(np.sign(GyroDataHat)))[0]
        if tol:
            self.ZeroCrossingsBioZIndex = find_nearest_time_index(self.IMU[self.ZeroCrossingsIMUIndex, 0], self.BioZ[0,:,0], tol)
        else:
            self.ZeroCrossingsBioZIndex = find_nearest_time_index(self.IMU[self.ZeroCrossingsIMUIndex, 0], self.BioZ[0,:,0])

    def get_h_alpa(self):
        self.dR5k = self.BioZ[0,self.ZeroCrossingsBioZIndex, 1].max() - self.BioZ[0,self.ZeroCrossingsBioZIndex, 1].min()
        self.dR100k = self.BioZ[1,self.ZeroCrossingsBioZIndex, 1].max() - self.BioZ[1,self.ZeroCrossingsBioZIndex, 1].min()
        self.h_alpha_zero_crossings = self.dR100k/self.dR5k
        self.h_alpha = (self.BioZ[0,:, 1].max() - self.BioZ[0,:, 1].min())/(self.BioZ[1,:,1].max() - self.BioZ[1,:,1].min())

    def extract_features(self):
        Features = np.zeros(19)
        Features[0] = self.BioZ[0,:,1].max()
        Features[1] = self.BioZ[0,:,1].min()
        Features[2] = self.BioZ[0,:,2].max()
        Features[3] = self.BioZ[0,:,2].min()
        Features[4] = self.BioZ[1,:,1].max()
        Features[5] = self.BioZ[1,:,1].min()
        Features[6] = self.BioZ[1,:,2].max()
        Features[7] = self.BioZ[1,:,2].min()

        Features[8] = self.BioZ[0,:,1].mean()
        Features[9] = self.BioZ[0,:,1].std()
        Features[10] = self.BioZ[1,:,1].mean()
        Features[11] = self.BioZ[1,:,1].std()

        Features[12] = self.BioZ[0,:,2].mean()
        Features[13] = self.BioZ[0,:,2].std()
        Features[14] = self.BioZ[1,:,2].mean()
        Features[15] = self.BioZ[1,:,2].std()
        GyroDataHat = signal.savgol_filter(self.IMU[:, 10], 15, 3)
        Energy = (GyroDataHat* GyroDataHat).sum() / (10000/(2**14))
        Features[16] = Energy
        Features[17] = self.IMU[:, 10].mean()
        Features[18] = self.IMU[:, 10].std()

        return Features
    def get_steps_start(self, Axis = 1):
        self.Steps = signal.find_peaks(-self.IMU[:,Axis], height=[5000/(2**14),], distance=15)[0]
        return 0

def get_data_windows(Dataset, WindowTimeSeconds):
    StartTime = Dataset.IMU.Data[0,0]
    EndTime = Dataset.IMU.Data[-1, 0]
    TimeWindows = np.arange(StartTime, EndTime, WindowTimeSeconds*1000)
    DataWindows = []
    for i in range(TimeWindows.shape[0]-1):
        IMUIndex1 = np.abs(TimeWindows[i] - Dataset.IMU.Data[:,0]).argmin()
        IMUIndex2 = np.abs(TimeWindows[i+1]- Dataset.IMU.Data[:, 0]).argmin()
        BioZIndex1 = np.abs(TimeWindows[i] - Dataset.BioZ.Data[0,:,0]).argmin()
        BioZIndex2 = np.abs(TimeWindows[i+1] - Dataset.BioZ.Data[0, :, 0]).argmin()
        TemperatureIndex1 = np.abs(TimeWindows[i] - Dataset.Temperature.Data[:,0]).argmin()
        TemperatureIndex2 = np.abs(TimeWindows[i+1] - Dataset.Temperature.Data[:, 0]).argmin()
        DataWindows.append(DataWindow(Dataset.BioZ.Data[:, BioZIndex1:BioZIndex2, :], Dataset.IMU.Data[IMUIndex1:IMUIndex2,:], Dataset.IMU.Data[TemperatureIndex1:TemperatureIndex2, :], [BioZIndex1, BioZIndex2], [IMUIndex1, IMUIndex2], [TemperatureIndex1, TemperatureIndex1]))
    return DataWindows

def find_foot_IMU(Dataset):
    Energy1 = (Dataset.IMU.Data[:,4] * Dataset.IMU.Data[:, 4]).sum()
    Energy2 = (Dataset.IMU.Data[:, 10] * Dataset.IMU.Data[:, 10]).sum()
    if Energy1>Energy2:
        return 4
    else:
        return 10

def find_walking_times(GyroData, SamplingRate, WindowSizeSeconds):
    WindowSizeSamples = int(WindowSizeSeconds*SamplingRate)
    DataLength = GyroData.shape[0]
    GyroDataWindows =  GyroData[:DataLength - int(DataLength%WindowSizeSamples)].reshape(-1, WindowSizeSamples)
    Energy = (GyroDataWindows*GyroDataWindows).sum(axis=1)/(10000/(2**14))
    WalkingWindowIndex = Energy>(10000/(2**14))
    temp = np.zeros(DataLength, dtype=bool)
    temp[:DataLength-int(DataLength%WindowSizeSamples)] = np.repeat(WalkingWindowIndex, WindowSizeSamples)
    return temp

def get_zero_crossings(GyroData):
    return np.where(np.diff(np.sign(GyroData)))[0]

def fir_filter(Data, FilterLength):
    FilteredData = np.zeros([Data.shape[0]-FilterLength])
    for i in range(FilteredData.shape[0]-FilterLength):
        FilteredData[i] = Data[i:i+FilterLength].mean()
    return FilteredData

class StepData:
    def __init__(self, BioZ, IMU, Temperature, StartTime, EndTime):
        self.BioZ = BioZ
        self.IMU = IMU
        self.Temperature = Temperature
        self.StartTime = StartTime
        self.EndTime = EndTime

