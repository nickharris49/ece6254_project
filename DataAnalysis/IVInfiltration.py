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
from Utilities.FileParser import parse_file
import matplotlib.cm as cm
import struct

def find_nearest_time_index(TimeArray, TimeArrayToFind, threshold=None):
    if threshold:
        d = np.abs(TimeArray - TimeArrayToFind[:, np.newaxis])
        return np.any(d <= threshold, axis=0)
    else:
        return np.abs(TimeArray-TimeArrayToFind[:, np.newaxis]).argmin(axis=0)

class PatientData:
    def __init__(self, PatientID):
        self.PatientID = PatientID
        self.BioZ = np.load('Data/IVInfiltration/%s/BioZ.npy'%PatientID)
        self.IMU = np.load('Data/IVInfiltration/%s/IMU.npy'%PatientID)
        self.Temperature = np.load('Data/IVInfiltration/%s/Temperature.npy'%PatientID)
        self.CurrentVoltage = np.load('Data/IVInfiltration/%s/BioZRaw.npy'%PatientID)
        StartOfDataCollectection = np.append(np.insert(np.where(self.IMU[1:, 0] - self.IMU[:-1, 0] > 50000)[0]+1, 0,0),-1)
        TemperatureStartofDataCollection = np.append(np.insert(np.where(self.Temperature[1:, 0] - self.Temperature[:-1, 0] > 50000)[0]+1, 0,0),-1)
        self.IMUMean = np.zeros([self.BioZ.shape[1] ,self.IMU.shape[1]])
        self.IMUStd = np.zeros([self.BioZ.shape[1] ,self.IMU.shape[1]])
        self.TemperatureMean = np.zeros([self.BioZ.shape[1],self.Temperature.shape[1]])
        self.TemperatureStd = np.zeros([self.BioZ.shape[1] ,self.Temperature.shape[1]])
        for i in range(self.BioZ.shape[1]):
            IMUIndicies = np.where(np.logical_and(self.IMU[:,0]>self.BioZ[0,i,0], self.IMU[:,0]<self.BioZ[-1,i,0]))[0]
            TMPIndicies = np.where(np.logical_and(self.Temperature[:,0]>self.BioZ[0,i,0], self.Temperature[:,0]<self.BioZ[-1,i,0]))[0]
            self.IMUMean[i] = self.IMU[IMUIndicies].mean(axis=0)
            self.IMUStd[i] = self.IMU[IMUIndicies].std(axis=0)
            self.IMUMean[i, 0] = self.BioZ[0,i,0]
            self.IMUStd[i, 0] = self.BioZ[0,i,0]
            self.TemperatureMean[i] = self.Temperature[TMPIndicies].mean(axis=0)
            self.TemperatureStd[i] = self.Temperature[TMPIndicies].std(axis=0)
            self.TemperatureMean[i, 0] = self.BioZ[0,i,0]
            self.TemperatureStd[i, 0] = self.BioZ[0,i,0]
        Xreal = np.array([-7.17218459e+07, 1.80172259e+08, 4.75042865e+02])
        A = np.ones([self.CurrentVoltage.shape[1], 3])
        A[:, 0] = (1 / self.CurrentVoltage[0, :, 1]).real
        A[:, 1] = (1 / self.CurrentVoltage[0, :, 1]).imag
        Zprotection = (1 / 1j * 2 * np.pi * 5000 * 0.01e-6) + (1 / 1j * 2 * np.pi * 5000 * 0.47e-6) + 2000
        ZtotReal = np.dot(A, Xreal) - Zprotection.real
        self.GoodDataIndex = (ZtotReal > 0) & (ZtotReal < 5000)
        self.IMUAngles = self.calculate_angles()

    def get_consecutive_filtered_diff(self, WindowSize=20, PercentageThreshold=20):
        print(self.PatientID)
        BioZ = self.BioZ[:, self.GoodDataIndex, :]
        self.FilteredBioZDiff = np.zeros([BioZ.shape[0], BioZ.shape[1]-(WindowSize+1), 2])
        for j in range(BioZ.shape[0]):
            DiffReal = 100 * np.diff(BioZ[j,:,1]) / BioZ[j,:-1,1]
            DiffImag = 100 * np.diff(BioZ[j,:,2]) / BioZ[j,:-1,2]
            for i in range(len(DiffReal) - WindowSize):
                TempDiffReal = DiffReal[i:i + WindowSize]
                TempDiffImag = DiffImag[i:i + WindowSize]
                PercentageReal = np.sum(TempDiffReal < 0)
                PercentageImag = np.sum(TempDiffImag < 0)
                if PercentageReal >= PercentageThreshold:
                    self.FilteredBioZDiff[j,i,0] = TempDiffReal[TempDiffReal < 0].sum()
                if PercentageImag >= PercentageThreshold:
                    self.FilteredBioZDiff[j,i,1] = TempDiffImag[TempDiffImag < 0].sum()
    def calculate_angles(self):
        Angles = np.zeros([self.IMUMean.shape[0], 7])
        Angles[:,0] = self.IMUMean[:,0]
        #X axis
        Angles[:,1] = np.arctan(self.IMUMean[:,1]/np.sqrt(self.IMUMean[:,2]**2 + self.IMUMean[:,3]**2))*180/np.pi
        Angles[:,2] = np.arctan(self.IMUMean[:,2]/np.sqrt(self.IMUMean[:,1]**2 + self.IMUMean[:,3]**2))*180/np.pi
        Angles[:,3] = np.arctan(self.IMUMean[:,3]/np.sqrt(self.IMUMean[:,2]**2 + self.IMUMean[:,1]**2))*180/np.pi
        Angles[:,4] = np.arctan(self.IMUMean[:,7]/np.sqrt(self.IMUMean[:,8]**2 + self.IMUMean[:,9]**2))*180/np.pi
        Angles[:,5] = np.arctan(self.IMUMean[:,8]/np.sqrt(self.IMUMean[:,7]**2 + self.IMUMean[:,9]**2))*180/np.pi
        Angles[:,6] = np.arctan(self.IMUMean[:,9]/np.sqrt(self.IMUMean[:,8]**2 + self.IMUMean[:,7]**2))*180/np.pi
        return Angles
    # def get_index_without_outlier(self, OhmThreshold):
    #     MeanPerFrequency = (self.BioZ[0,:,1].mean(axis=1))
    #     for i in range(MeanPerFrequency.shape[0]):



class RealTimeAlgorithmData:
    def __init__(self, bytearray):
        NumDataPoints = int(len(bytearray)/84)
        self.MeasurementIndex = np.zeros(NumDataPoints, dtype=int)
        self.DropSince =  np.zeros(NumDataPoints ,dtype=int)
        self.TimeOfMeasurement = np.zeros(NumDataPoints ,dtype=float)
        self.Current5kHz = np.zeros(NumDataPoints, dtype=complex)
        self.Voltage5kHz = np.zeros(NumDataPoints, dtype=complex)
        self.Z5kHz = np.zeros(NumDataPoints, dtype=complex)

        self.Current100kHz = np.zeros(NumDataPoints, dtype=complex)
        self.Voltage100kHz = np.zeros(NumDataPoints, dtype=complex)
        self.Z100kHz = np.zeros(NumDataPoints, dtype=complex)

        self.R5kHzDropSince =  np.zeros(NumDataPoints, dtype=float)
        self.R100kHzDropSince = np.zeros(NumDataPoints, dtype=float)
        self.InfiltrationRiskR5kHz = np.zeros(NumDataPoints, dtype=int)
        self.InfiltrationRiskR100kHz =np.zeros(NumDataPoints, dtype=int)
        self.IRCummulativeR5kHz =np.zeros(NumDataPoints, dtype=int)
        self.IRCummulativeR100kHz =np.zeros(NumDataPoints, dtype=int)
        self.InfiltrationFlag = np.zeros(NumDataPoints, dtype=int)

        self.ElectrodeImpedance5k = np.zeros(NumDataPoints, dtype=float)
        for i in range(NumDataPoints):
            bytestoparse = bytearray[i*84 : (i+1)*84]
            self.MeasurementIndex[i] = int.from_bytes(bytestoparse[0:2], byteorder='little', signed=False)
            self.DropSince[i] = int.from_bytes(bytestoparse[2:4], byteorder='little', signed=False)
            self.TimeOfMeasurement[i] = int.from_bytes(bytestoparse[4:8], byteorder='little', signed=False)/2.048

            self.Current5kHz[i] =  int.from_bytes(bytestoparse[8:12], byteorder='little', signed=True) -1j*int.from_bytes(bytestoparse[12:16], byteorder='little', signed=True)
            self.Voltage5kHz[i] = int.from_bytes(bytestoparse[16:20], byteorder='little', signed=True) -1j*int.from_bytes(bytestoparse[20:24], byteorder='little', signed=True)
            self.Z5kHz[i] = struct.unpack('<f',bytestoparse[24:28] )[0] -1j*struct.unpack('<f',bytestoparse[28:32] )[0]
            Xreal = np.array([-7.17218459e+07, 1.80172259e+08, 4.75042865e+02])
            Zprotection = (1 / (1j * 2 * np.pi * 5000 * 0.01e-6)) + (1 / (1j * 2 * np.pi * 5000 * 0.47e-6)) + 2000
            self.ElectrodeImpedance5k[i] = (1/self.Current5kHz[i].real)*Xreal[0] + (1/self.Current5kHz[i].imag)*Xreal[1] + Xreal[2] - Zprotection.real

            self.Current100kHz[i] = int.from_bytes(bytestoparse[32:36], byteorder='little', signed=True) -1j*int.from_bytes(bytestoparse[36:40], byteorder='little', signed=True)
            self.Voltage100kHz[i] = int.from_bytes(bytestoparse[40:44], byteorder='little', signed=True) -1j*int.from_bytes(bytestoparse[44:48], byteorder='little', signed=True)
            self.Z100kHz[i] = struct.unpack('<f',bytestoparse[48:52])[0] -1j*struct.unpack('<f',bytestoparse[52:56])[0]

            self.R5kHzDropSince[i] = struct.unpack('<f',bytestoparse[56:60] )[0]
            self.R100kHzDropSince[i] = struct.unpack('<f',bytestoparse[60:64] )[0]
            self.InfiltrationRiskR5kHz[i] =  int.from_bytes(bytestoparse[64:68], byteorder='little', signed=False)
            self.InfiltrationRiskR100kHz[i] =  int.from_bytes(bytestoparse[68:72], byteorder='little', signed=False)
            self.IRCummulativeR5kHz[i] =  int.from_bytes(bytestoparse[72:76], byteorder='little', signed=False)
            self.IRCummulativeR100kHz[i] =  int.from_bytes(bytestoparse[76:80], byteorder='little', signed=False)
            self.InfiltrationFlag[i] = int.from_bytes(bytestoparse[80:81], byteorder='little', signed=False)






class BaselineResetIMU:
    def __init__(self, PatientData, IMUThreshold=1000, IMUStdThreshold=100):
        BaselineBioZ = PatientData.BioZ[:,0,:]
        AccelAxis = [1,2,3,7,8,9]
        BaselinePosition = PatientData.IMUMean[0,AccelAxis]
        self.NormalizedBioZ = np.zeros(PatientData.BioZ.shape)
        for i in range(PatientData.BioZ.shape[1]-2):
            #First check if patient moved during data collection
            if np.all(PatientData.IMUStd[i+1,AccelAxis]>IMUStdThreshold):
                print("Patient Moved during collection")
                self.NormalizedBioZ[:,i+1,:] = self.NormalizedBioZ[:,i,:]
            else:
                ComparisonArray = (BaselinePosition >= PatientData.IMUMean[i+1, AccelAxis] - IMUThreshold) & (
                        BaselinePosition <= PatientData.IMUMean[i+1, AccelAxis] + IMUThreshold)
                print(i+1)
                if np.all(ComparisonArray):
                    print("SamePosition")
                    self.NormalizedBioZ[:, i+1, :] = PatientData.BioZ[:, i+1, :] - BaselineBioZ
                else:
                    print("DifferentPosition")
                    BaselineBioZ = PatientData.BioZ[:, i+1, :]- self.NormalizedBioZ[:,i,:]
                    self.NormalizedBioZ[:, i+1, :] = self.NormalizedBioZ[:, i, :]
                    BaselinePosition = PatientData.IMUMean[i+1, AccelAxis]
        self.NormalizedBioZ[:,:,0] = PatientData.BioZ[:,:,0]




