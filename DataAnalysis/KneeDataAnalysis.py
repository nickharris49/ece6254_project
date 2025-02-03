import datetime

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
import copy


#This file will have different classes and helper functions for knee bioimpedance analysis.
#The main class takes in bioz data (5k100k, fullsweep and IMU data in the commonly used format)
#There will be a step class that has the different features of the step at different frequencies and IMU data
#There will be a flexion extension class that also uses the bioz and imu data, and will need to get joint angle
class KneeDataAnalysis:
    def __init__(self, Dataset, Date, Leg, Pain, Swelling, Stiffnes):
        self.Dataset = Dataset
        self.Date = Date
        self.Leg = Leg
        self.Pain = Pain
        self.Swelling = Swelling
        self.Stiffness = Stiffnes

def GetStepsFromTimeWindow(BioZData, IMUData, AxisIndexToUse,IMUDataSign, StartTime, StopTime, peak_height=0.1, distance_between_peaks=50):
        StartIndex = np.argmin(np.abs(StartTime-IMUData[:,0]))
        StopIndex = np.argmin(np.abs(StopTime-IMUData[:,0]))
        ArrayIndex = signal.find_peaks(IMUDataSign * IMUData[StartIndex:StopIndex, AxisIndexToUse],distance = distance_between_peaks, height=peak_height)[0] + StartIndex
        ZeroCrossing = np.zeros(ArrayIndex.shape[0], dtype=int)
        for i in range(len(ArrayIndex)):
            A = ArrayIndex[i]
            TempArray = -IMUData[(A - 50):A, AxisIndexToUse]
            try:
                ZeroCrossing[i] = A - 50 + np.where(np.diff(np.sign(TempArray)))[0][-1]
            except Exception as e:
                print(e)
        # Filter where 0 xings dont exist
        ZeroCrossingIndex = ZeroCrossing[ZeroCrossing > 0]
        # Find the closest bioz samples to the imu Z-axis gyrp 0 xings/start of forward swing.
        #Remove
        IMUZeroCrossingTimeArray = IMUData[ZeroCrossingIndex, 0]
        BioZTimeArrayToFind = BioZData[0, :, 0]
        BioZ0CrossingIndex = np.abs(IMUZeroCrossingTimeArray - BioZTimeArrayToFind[:, np.newaxis]).argmin(axis=0)
        # Loop through and split bioz and imu data into steps

        BioZSteps = []
        IMUSteps = []
        for i in range(len(BioZ0CrossingIndex) - 1):
            if (BioZ0CrossingIndex[i + 1] - BioZ0CrossingIndex[i]) < 200 and (
                    BioZ0CrossingIndex[i + 1] - BioZ0CrossingIndex[i]) > 0 and (ZeroCrossingIndex[i + 1] - ZeroCrossingIndex[i]) < 200 and (
                    ZeroCrossingIndex[i + 1] - ZeroCrossingIndex[i]) > 0:
                BioZSteps.append(BioZData[:, BioZ0CrossingIndex[i]:BioZ0CrossingIndex[i + 1], :])
                IMUSteps.append( IMUData[ZeroCrossingIndex[i]:ZeroCrossingIndex[i + 1],:])
            # Resample BioZ and IMU steps to ResampleSize
        return BioZSteps, IMUSteps


def GetStepsFromTimeWindowStartingHeelStrike(BioZData, IMUData,  StartTime, StopTime,MidSwingGyroIndex,MidSwingGyroSign,mid_swing_peak_height=0.1, mid_distance_between_peaks=50, HeelStrikeAcclIndex=1, HeelStrikeSign=1):
        StartIndex = np.argmin(np.abs(StartTime-IMUData[:,0]))
        StopIndex = np.argmin(np.abs(StopTime-IMUData[:,0]))
        ArrayIndex = signal.find_peaks(MidSwingGyroSign * IMUData[StartIndex:StopIndex, MidSwingGyroIndex],distance = mid_distance_between_peaks, height=mid_swing_peak_height)[0] + StartIndex
        HeelStrikes = np.zeros(ArrayIndex.shape[0], dtype=int)
        for i in range(len(ArrayIndex)):
            A = ArrayIndex[i]
            TempArray = HeelStrikeSign*IMUData[A:(A+20), HeelStrikeAcclIndex]
            try:
                HeelStrikes[i] = A+ np.argmax(TempArray)
            except Exception as e:
                print(e)
        # Filter where 0 xings dont exist
        HeelStrikeIndex = HeelStrikes[HeelStrikes > 0]
        # Find the closest bioz samples to the imu Z-axis gyrp 0 xings/start of forward swing.
        #Remove
        IMUHeelStrikeTimeArray = IMUData[HeelStrikeIndex, 0]
        BioZTimeArrayToFind = BioZData[0, :, 0]
        BioZHeelStrikeIndex = np.abs(IMUHeelStrikeTimeArray - BioZTimeArrayToFind[:, np.newaxis]).argmin(axis=0)
        # Loop through and split bioz and imu data into steps

        BioZSteps = []
        IMUSteps = []
        for i in range(len(BioZHeelStrikeIndex) - 1):
            if (BioZHeelStrikeIndex[i + 1] - BioZHeelStrikeIndex[i]) < 200 and (
                    BioZHeelStrikeIndex[i + 1] - BioZHeelStrikeIndex[i]) > 0 and (HeelStrikeIndex[i + 1] - HeelStrikeIndex[i]) < 200 and (
                    HeelStrikeIndex[i + 1] - HeelStrikeIndex[i]) > 0:
                BioZSteps.append(BioZData[:, BioZHeelStrikeIndex[i]:BioZHeelStrikeIndex[i + 1], :])
                IMUSteps.append( IMUData[HeelStrikeIndex[i]:HeelStrikeIndex[i + 1],:])
            # Resample BioZ and IMU steps to ResampleSize
        return BioZSteps, IMUSteps
def GetResampledStepsFromTimeWindow(BioZData, IMUData, AxisIndexToUse,IMUDataSign, ResampleSize, StartTime, StopTime):
    BioZSteps, IMUSteps = GetStepsFromTimeWindow(BioZData, IMUData, AxisIndexToUse, IMUDataSign, StartTime, StopTime)
    ResampledBioZSteps = ResampleBioZSteps(BioZSteps, ResampleSize)
    ResampledIMUSteps = ResampleIMUSteps(IMUSteps, ResampleSize)
    #Plot the mean of the resampled steps.
    fig, ax = plt.subplots(figsize=(9,7))
    ax.plot(ResampledBioZSteps[:,:,0].mean(axis=0))
    ax.plot(ResampledBioZSteps[:,:,1].mean(axis=0))
    plt.tight_layout()
    plt.show()
    return ResampledBioZSteps, ResampledIMUSteps

def ResampleBioZSteps(BioZSteps, ResampleSize):
    ResampledBioZSteps = np.zeros([len(BioZSteps), ResampleSize, 2])
    for i in range(len(BioZSteps)):
        ResampledBioZSteps[i, :, 0] = signal.resample(BioZSteps[i][0,:,1], ResampleSize)
        ResampledBioZSteps[i, :, 1] = signal.resample(BioZSteps[i][1,:,1], ResampleSize)
    return ResampledBioZSteps

def ResampleIMUSteps(IMUSteps, ResampleSize):
    ResampledIMUSteps = np.zeros([len(IMUSteps), ResampleSize, 12])
    for i in range(len(IMUSteps)):
        for j in range(12):
            ResampledIMUSteps[i, :, j] = signal.resample(IMUSteps[i][:, j+1], ResampleSize)
    return ResampledIMUSteps


class Step:
    def __init__(self, BioZStep, IMUStep, AxisIndex, Direction, Debug=False, MidSwingPeakHeight=0.1, HeelStrikePeakHeight=0.5, ToeOffPeak = 0.03, IMUHealStrikeSign = 1):
        self.BioZ = BioZStep
        self.IMU = IMUStep
        self.AxisIndex = AxisIndex
        self.Direction = Direction
        self.ForwardSwing = None
        if (np.abs(BioZStep[0,0,0]-IMUStep[0,0]))>100:
            raise Exception("IMU and BioZ Step not the same")
        if np.abs((BioZStep[0,-1,0]-BioZStep[0,0,0]) - (IMUStep[-1,0]-IMUStep[0,0]))>200:
            raise Exception("IMU and BioZ Step sizes are not close ")
        MidSwing = signal.find_peaks(Direction*IMUStep[0:int(IMUStep.shape[0]/3), 6], distance=50, height=MidSwingPeakHeight)[0]
        if len(MidSwing)==0:
            raise Exception("Can't find Mid Swing peak")
        MidSwing = MidSwing[0]
        HeelStrike = signal.find_peaks(IMUHealStrikeSign*IMUStep[MidSwing:int(IMUStep.shape[0]/2), 2], distance=50, height=HeelStrikePeakHeight)[0] + MidSwing
        if len(HeelStrike)==0:
            raise Exception("Can't find Heel Strike peak")
        HeelStrike = HeelStrike[0]
        ToeOff = signal.find_peaks(-1*Direction*IMUStep[int(IMUStep.shape[0]/2):, 6], distance=50, height=ToeOffPeak)[0] + int(IMUStep.shape[0]/2)
        if len(ToeOff)==0:
            raise Exception("Can't find Toe off peak")
        ToeOff = ToeOff[0]

        self.IMUSignalPeakIndex = np.array([MidSwing, HeelStrike, ToeOff])
        self.IMUSignalPeakTimes = np.array([IMUStep[MidSwing,0], IMUStep[HeelStrike,0], IMUStep[ToeOff,0]])

        MidSwingBioZIndex = np.argmin(np.abs(self.IMUSignalPeakTimes[0]-BioZStep[0,:,0]))
        HeelStrikeSwingBioZIndex = np.argmin(np.abs(self.IMUSignalPeakTimes[1]-BioZStep[0,:,0]))
        ToeOffBioZIndex = np.argmin(np.abs(self.IMUSignalPeakTimes[2]-BioZStep[0,:,0]))

        self.BioZIndexIMUSignalPeaks = np.array([MidSwingBioZIndex,HeelStrikeSwingBioZIndex,ToeOffBioZIndex])
        if(Debug):
            fig, ax = plt.subplots(3, figsize=(9, 7))
            ax[0].plot(BioZStep[0, :, 0], BioZStep[0, :, 1])
            ax[0].scatter(BioZStep[0, :, 0] , BioZStep[0, :, 1])
            ax[0].scatter(BioZStep[0, self.BioZIndexIMUSignalPeaks, 0] , BioZStep[0, self.BioZIndexIMUSignalPeaks, 1])
            ax[1].plot(BioZStep[1, :, 0] , BioZStep[1, :, 1])
            ax[1].scatter(BioZStep[1, :, 0] , BioZStep[1, :, 1])
            ax[1].scatter(BioZStep[1, self.BioZIndexIMUSignalPeaks, 0] , BioZStep[1, self.BioZIndexIMUSignalPeaks, 1])
            ax[2].plot(IMUStep[:, 0] , IMUStep[:, 6])
            ax[2].scatter(IMUStep[:, 0] , IMUStep[:, 6])
            ax[2].scatter(IMUStep[self.IMUSignalPeakIndex, 0] , IMUStep[self.IMUSignalPeakIndex, 6])
            plt.tight_layout()
            plt.show()
        self.BioZSwing = self.BioZ[:,:HeelStrikeSwingBioZIndex,:]
        self.BioZStance = self.BioZ[:,HeelStrikeSwingBioZIndex:ToeOffBioZIndex,:]
        self.IMUSwing = self.IMU[:HeelStrike, :]
        self.IMUStance = self.IMU[HeelStrike:, :]

#Sometimes the device is left on and not on tha patient or the patient is not moving, this function is provided with the bioz, and imu data and filtered over a specific time window.
#Each data point will be a time in the day tracked from the logs along with pain score etc.
# This will help minimize the amount of data.
# The windowing will only be applied for the 5k100k data, but not the full sweep.

class KneeDataSnapShot:
    def __init__(self, FSBioZClass, Fivek100kBioZClass, IMUClass, TimeOfDeviceOn, StartTimeWindow, StopTimeWindow, Pain, Swelling, Stiffness, Leg, Notes):
        self.BioZ0 = FSBioZClass
        self.BioZ1 = Fivek100kBioZClass
        self.IMU = IMUClass
        self.TimeOfDeviceOn = TimeOfDeviceOn
        self.TimeOfData = TimeOfDeviceOn + datetime.timedelta(milliseconds=StartTimeWindow*3600000)
        self.StartWindow = StartTimeWindow
        self.StopWindow = StopTimeWindow
        self.Pain = Pain
        self.Swelling = Swelling
        self.Stiffness = Stiffness
        self.Notes = Notes
        try:
            BioZ1StartIndex = np.argmin(np.abs(self.BioZ1.Data[0,:,0]-StartTimeWindow*3600000))
            BioZ1StopIndex = np.argmin(np.abs(self.BioZ1.Data[0,:,0]-StopTimeWindow*3600000))
            self.BioZ1.Data = self.BioZ1.Data[:, BioZ1StartIndex:BioZ1StopIndex, :]
            self.BioZ1.ElectrodeImpedance = self.BioZ1.ElectrodeImpedance[BioZ1StartIndex:BioZ1StopIndex]
        except Exception as e:
            print(e)
        try:
            IMUDataStartIndex = np.argmin(np.abs(self.IMU.Data[:,0]-StartTimeWindow*3600000))
            IMUDataStopIndex = np.argmin(np.abs(self.IMU.Data[:,0]-StopTimeWindow*3600000))
            self.IMU.Data = self.IMU.Data[IMUDataStartIndex:IMUDataStopIndex, :]
        except Exception as e:
            print(e)
        self.Leg = Leg
    def GetStepsFromWindow(self,AxisIndexToUse,IMUDataSign, StartTime, StopTime, peak_height=0.1, distance_between_peaks=50):
        self.BioZStepsData, self.IMUStepsData = GetStepsFromTimeWindow(self.BioZ1.Data, self.IMU.Data, AxisIndexToUse,IMUDataSign, StartTime, StopTime, peak_height, distance_between_peaks)

    def GetStepsFromTimeWindowStartingHeelStrike(self, StartTime,StopTime, MidSwingGyroIndex, MidSwingGyroSign, mid_swing_peak_height, mid_distance_between_peaks,
                               HeelStrikeAcclIndex, HeelStrikeSign):
        self.MidSwingGyroIndex = MidSwingGyroIndex
        self.MidSwingGyroSign = MidSwingGyroSign
        self.mid_swing_peak_height = mid_swing_peak_height
        self.mid_distance_between_peaks = mid_distance_between_peaks
        self.HeelStrikeAcclIndex = HeelStrikeAcclIndex
        self.HeelStrikeSign = HeelStrikeSign
        self.BioZStepsData, self.IMUStepsData = GetStepsFromTimeWindowStartingHeelStrike(self.BioZ1.Data, self.IMU.Data,StartTime,StopTime,
                                                                           MidSwingGyroIndex, MidSwingGyroSign,mid_swing_peak_height, mid_distance_between_peaks,HeelStrikeAcclIndex, HeelStrikeSign)

    def ResampleStepData(self,ResampleSize):
        try:
            self.ResampledBioZStepData =  ResampleBioZSteps(self.BioZStepsData, ResampleSize)
            print('here')
            self.ResampledIMUStepData =  ResampleIMUSteps(self.IMUStepsData, ResampleSize)
        except Exception as e:
            print(e)
    def GetStepsFromTimes(self, StepTimes):
         self.BioZStepsData = []
         for i in range(len(StepTimes)):
             StartIndex = np.argmin(np.abs(self.BioZ1.Data[0, :, 0] - StepTimes[i][0] * 3600000))
             StopIndex = np.argmin(np.abs(self.BioZ1.Data[0, :, 0] - StepTimes[i][1] * 3600000))
             self.BioZStepsData.append(self.BioZ1.Data[:, StartIndex:StopIndex, :])

    def ResampleStepsFromIMU(self, GyroAxis = 6, GyroSign = -1, HeelStrikeSign=1, FSLength=15, StanceLength=30, BackwardSwingLength=5):
        self.ResampledBioZPhases = np.zeros([len(self.BioZStepsData), FSLength+StanceLength+BackwardSwingLength,2])
        self.ResampledIMUPhases = np.zeros([len(self.BioZStepsData), FSLength+StanceLength+BackwardSwingLength,12])

        for i in range(len(self.BioZStepsData)):
            try:
                TempStep = Step(self.BioZStepsData[i], self.IMUStepsData[i], GyroAxis, GyroSign, IMUHealStrikeSign=HeelStrikeSign)
                for j in range(2):
                    FS = signal.resample(self.BioZStepsData[i][j, :TempStep.BioZIndexIMUSignalPeaks[1], 1], FSLength)
                    SP = signal.resample(self.BioZStepsData[i][j,TempStep.BioZIndexIMUSignalPeaks[1]:TempStep.BioZIndexIMUSignalPeaks[2], 1], StanceLength)
                    BS = signal.resample(self.BioZStepsData[i][j, TempStep.BioZIndexIMUSignalPeaks[2]:, 1], BackwardSwingLength)
                    self.ResampledBioZPhases[i,:,j] = np.concatenate((FS,SP,BS))
                for j in range(12):
                    FS = signal.resample(self.IMUStepsData[i][:TempStep.IMUSignalPeakIndex[1], j], FSLength)
                    SP = signal.resample(self.IMUStepsData[i][TempStep.IMUSignalPeakIndex[1]:TempStep.IMUSignalPeakIndex[2], j], StanceLength)
                    BS = signal.resample(self.IMUStepsData[i][TempStep.IMUSignalPeakIndex[2]:, j], BackwardSwingLength)
                    self.ResampledIMUPhases[i,:,j] = np.concatenate((FS,SP,BS))
            except Exception as e:
                print(i,e)


class FE_Analysis:
    def __init__(self, SS):
        self.SS = SS
    def SegmentFECycles(self, AxisDirection,FilterWindow = 21, DistanceBWPeaksx = 100, DistanceBWPeaksy=100, xPeakHeight=0.3, yPeakHeight=0.3, plot_rawdata=False):
        self.KneeFlexionBioZ = []
        self.KneeExtensionBioZ = []
        self.KneeFlexionBioZ100k = []
        self.KneeExtensionBioZ100k = []
        self.KneeFlexionIMU = []
        self.KneeExtensionIMU = []
        self.KneeFlexionAngle = []
        self.KneeExtensionAngle = []
        self.RawBioZ = self.SS.BioZ1.Data[0, :, 1]
        self.FilteredBioZ = signal.savgol_filter(self.RawBioZ, FilterWindow, 3)
        self.RawBioZ100k = self.SS.BioZ1.Data[1, :, 1]
        self.FilteredBioZ100k = signal.savgol_filter(self.RawBioZ100k, FilterWindow, 3)
        self.TempIMUx = signal.resample(self.SS.IMU.Data[:, 1], len(self.FilteredBioZ))
        self.TempIMU2x = signal.resample(self.SS.IMU.Data[:, 7], len(self.FilteredBioZ))
        self.FilteredIMUx = signal.savgol_filter(self.TempIMUx, FilterWindow, 3)
        self.FilteredIMU2x = signal.savgol_filter(self.TempIMU2x, FilterWindow, 3)
        self.TempIMUy = signal.resample(self.SS.IMU.Data[:, 2], len(self.FilteredBioZ))
        self.FilteredIMUy = signal.savgol_filter(self.TempIMUy, FilterWindow, 3)
        self.Peaksy = signal.find_peaks((-1 * self.FilteredIMUy) + 1, distance=DistanceBWPeaksy, height=yPeakHeight)[0]
        self.Peaksx = signal.find_peaks((AxisDirection * self.FilteredIMUx) + 1, distance=DistanceBWPeaksx, height=xPeakHeight)[0]
        try:
            self.Peaksx = self.Peaksx[np.where(self.Peaksy[0] > self.Peaksx)[0][-1]:]
        except Exception as e:
            # print(e)
            self.Peaksy = self.Peaksy[1:]
            self.Peaksx = self.Peaksx[np.where(self.Peaksy[0] > self.Peaksx)[0][-1]:]
        if plot_rawdata:
            fig, ax = plt.subplots(3)
            ax[0].plot(self.FilteredIMUy)
            ax[0].scatter(self.Peaksy,self.FilteredIMUy[self.Peaksy])
            ax[1].plot(self.FilteredIMUx)
            ax[1].scatter(self.Peaksx,self.FilteredIMUx[self.Peaksx])
            ax[2].plot(self.FilteredBioZ)
            plt.tight_layout()
            plt.show()
            fig, ax = plt.subplots(2)
        for j in range(len(self.Peaksx) - 1):
            try:
                ResampleToSamples = 70
                self.KneeExtensionBioZ.append(signal.resample(self.FilteredBioZ[self.Peaksx[j]:self.Peaksy[j]], ResampleToSamples))
                self.KneeFlexionBioZ.append(signal.resample(self.FilteredBioZ[self.Peaksy[j]:self.Peaksx[j + 1]], ResampleToSamples))
                self.KneeExtensionBioZ100k.append(signal.resample(self.FilteredBioZ100k[self.Peaksx[j]:self.Peaksy[j]], ResampleToSamples))
                self.KneeFlexionBioZ100k.append(signal.resample(self.FilteredBioZ100k[self.Peaksy[j]:self.Peaksx[j + 1]], ResampleToSamples))
                self.KneeExtensionIMU.append(signal.resample(self.FilteredIMUy[self.Peaksx[j]:self.Peaksy[j]], ResampleToSamples))
                self.KneeExtensionAngle.append(90*signal.resample(self.FilteredIMUy[self.Peaksx[j]:self.Peaksy[j]]-self.FilteredIMU2x[self.Peaksx[j]:self.Peaksy[j]], ResampleToSamples))
                self.KneeFlexionIMU.append(signal.resample(self.FilteredIMUy[self.Peaksy[j]:self.Peaksx[j + 1]], ResampleToSamples))
                self.KneeFlexionAngle.append(90*signal.resample(self.FilteredIMUy[self.Peaksy[j]:self.Peaksx[j + 1]]-self.FilteredIMU2x[self.Peaksy[j]:self.Peaksx[j + 1]], ResampleToSamples))
                if plot_rawdata:
                    ax[0].plot(self.FilteredBioZ[self.Peaksx[j]:self.Peaksy[j]]-self.FilteredBioZ[self.Peaksy[j]],90*self.FilteredIMUy[self.Peaksx[j]:self.Peaksy[j]],color="blue")
                    ax[0].plot(self.FilteredBioZ[self.Peaksy[j]:self.Peaksx[j+1]]-self.FilteredBioZ[self.Peaksy[j]],90*self.FilteredIMUy[self.Peaksy[j]:self.Peaksx[j+1]], color="blue")
                    # print(len(FilteredBioZ[Peaksx[j]:Peaksy[j]]),len(FilteredBioZ[Peaksy[j]:Peaksx[j+1]]-FilteredBioZ[Peaksy[j]]))
                    ax[1].plot(self.FilteredBioZ[self.Peaksx[j]:self.Peaksy[j]]-self.FilteredBioZ[self.Peaksy[j]],90*(self.FilteredIMUy[self.Peaksx[j]:self.Peaksy[j]]-self.FilteredIMU2x[self.Peaksx[j]:self.Peaksy[j]]), color="blue")
                    ax[1].plot(self.FilteredBioZ[self.Peaksy[j]:self.Peaksx[j+1]]-self.FilteredBioZ[self.Peaksy[j]],90*(self.FilteredIMUy[self.Peaksy[j]:self.Peaksx[j+1]]-self.FilteredIMU2x[self.Peaksy[j]:self.Peaksx[j+1]]),color="blue")
                    ax[1].plot(self.KneeExtensionBioZ[j]-self.KneeExtensionBioZ[j][-1],self.KneeExtensionAngle[j],color="red")
                    ax[1].plot(self.KneeFlexionBioZ[j]-self.KneeFlexionBioZ[j][0],self.KneeFlexionAngle[j],color="red")
            except Exception as e:
                print(e)
        if plot_rawdata:
            plt.tight_layout()
            plt.show()


        # plt.close('all')
    def AnalyzeByAngle(self, StartAngle=30, EndAngle=60, AngleResolution=5):
        self.AnglesToCompareAt = np.arange(StartAngle, EndAngle+AngleResolution, AngleResolution)
        fig, ax = plt.subplots(3, figsize=(9, 7))
        ax[0].plot(np.array(self.KneeExtensionBioZ).mean(axis=0),  np.array(self.KneeExtensionAngle).mean(axis=0))
        ax[0].plot(np.array(self.KneeFlexionBioZ).mean(axis=0), np.array(self.KneeFlexionAngle).mean(axis=0))
        self.diffPerAngle = []
        self.diff100kPerAngle = []
        self.diffExtensionPerAngle = []
        self.diff100kExtensionPerAngle = []
        self.diffFlexionPerAngle = []
        self.diff100kFlexionPerAngle = []
        self.FlexionBioZAtAngles = []
        self.FlexionBioZ100kAtAngles = []
        self.ExtensionBioZAtAngles = []
        self.ExtensionBioZ100kAtAngles = []
        self.FlexionAngles = []
        self.ExtensionAngles = []
        for j in range(len(self.KneeExtensionBioZ)):
            a = np.array(self.KneeExtensionAngle[j])
            b = np.array(self.KneeFlexionAngle[j])
            a_bioz = np.array(self.KneeExtensionBioZ[j])
            a_bioz100k = np.array(self.KneeExtensionBioZ100k[j])
            b_bioz = np.array(self.KneeFlexionBioZ[j])
            b_bioz100k =np.array(self.KneeFlexionBioZ100k[j])
            temp_a = np.argmin(np.abs(a[:, np.newaxis] - self.AnglesToCompareAt), axis=0)
            temp_b = np.argmin(np.abs(b[:, np.newaxis] - self.AnglesToCompareAt), axis=0)
            # ax[1].scatter(a[temp_a], (a_bioz[temp_a] - b_bioz[temp_b]))
            ax[0].scatter((a_bioz[temp_a]),a[temp_a],  marker='x')  # -a_bioz[temp_a][-1]
            ax[0].scatter((b_bioz[temp_b]),b[temp_b], marker='o')  # -b_bioz[temp_b][-1]
            ax[1].scatter(a[temp_a], (a_bioz[temp_a]), marker='x')  # -a_bioz[temp_a][-1]
            ax[1].scatter(b[temp_b], (b_bioz[temp_b]), marker='o')  # -b_bioz[temp_b][-1]
            self.FlexionBioZAtAngles.append(b_bioz[temp_b])
            self.FlexionBioZ100kAtAngles.append(b_bioz100k[temp_b])
            self.ExtensionBioZAtAngles.append(a_bioz[temp_a])
            self.ExtensionBioZ100kAtAngles.append(a_bioz100k[temp_a])
            self.FlexionAngles.append(b[temp_b])
            self.ExtensionAngles.append(a[temp_a])
            self.diffPerAngle.append((a_bioz[temp_a]) - (b_bioz[temp_b]))
            self.diff100kPerAngle.append((a_bioz100k[temp_a]) - (b_bioz100k[temp_b]))
            self.diffExtensionPerAngle.append(a_bioz[temp_a] - (a_bioz[temp_a][0]))
            self.diff100kExtensionPerAngle.append(a_bioz100k[temp_a] - (a_bioz100k[temp_a][0]))
            self.diffFlexionPerAngle.append(b_bioz[temp_b] - b_bioz[temp_b][0])
            self.diff100kFlexionPerAngle.append(b_bioz100k[temp_b] - b_bioz100k[temp_b][0])
            # ax[2].scatter(AnglesToCompareAt, (a_bioz[temp_a] -a_bioz[temp_a][-1])-(b_bioz[temp_b] -b_bioz[temp_b][-1]), marker='x')
        self.diffPerAngle = np.array(self.diffPerAngle)
        self.diff100kPerAngle = np.array(self.diff100kPerAngle)
        self.diffExtensionPerAngle = np.array(self.diffExtensionPerAngle)
        self.diff100kExtensionPerAngle = np.array(self.diff100kExtensionPerAngle)
        self.diffFlexionPerAngle = np.array(self.diffFlexionPerAngle)
        self.diff100kFlexionPerAngle = np.array(self.diff100kFlexionPerAngle)
        self.FlexionAngles = np.array(self.FlexionAngles)
        self.ExtensionAngles = np.array(self.ExtensionAngles)
        self.ExtensionBioZAtAngles = np.array(self.ExtensionBioZAtAngles)
        self.ExtensionBioZ100kAtAngles = np.array(self.ExtensionBioZ100kAtAngles)
        self.FlexionBioZAtAngles = np.array(self.FlexionBioZAtAngles)
        self.FlexionBioZ100kAtAngles = np.array(self.FlexionBioZ100kAtAngles)
        ax[2].scatter(self.AnglesToCompareAt, self.diffPerAngle.mean(axis=0), marker='x', color='blue')
        # print(self.diffPerAngle.mean(axis=0)[0:2], np.sum(np.abs(self.diffPerAngle.mean(axis=0))), np.sum(np.abs(
        # self.diffExtensionPerAngle.mean(axis=0))), np.sum(np.abs(self.diffFlexionPerAngle.mean(axis=0))))
        plt.tight_layout()
        plt.show()


class FE_Analysis_v2:
    def __init__(self, SS):
        self.SS = SS
    def SegmentFECycles(self, AxisDirection,FilterWindow = 21, DistanceBWPeaksx = 300, DistanceBWPeaksy=300, xPeakHeight=0.3, yPeakHeight=0.3, plot_rawdata=False):
        #First filter bioz data
        self.BioZ1 = copy.deepcopy(self.SS.BioZ1)
        self.IMU = copy.deepcopy(self.SS.IMU)
        self.BioZ1.Data[0,:,1] = signal.savgol_filter(self.BioZ1.Data[0,:,1], FilterWindow, 3)
        self.BioZ1.Data[0,:,2] = signal.savgol_filter(self.BioZ1.Data[0,:,2], FilterWindow, 3)
        self.BioZ1.Data[1,:,1] = signal.savgol_filter(self.BioZ1.Data[1,:,1], FilterWindow, 3)
        self.BioZ1.Data[1,:,2] = signal.savgol_filter(self.BioZ1.Data[1,:,2], FilterWindow, 3)
        for i in range(12):
            self.IMU.Data[:,i+1] = signal.savgol_filter(self.IMU.Data[:,i+1], FilterWindow,3)
        #2nd is find peaks in IMU data
        self.Peaksy = signal.find_peaks((-1 * self.IMU.Data[:,2]) + 1, distance=DistanceBWPeaksy, height=yPeakHeight)[0]
        self.Peaksx = signal.find_peaks((AxisDirection * self.IMU.Data[:,1]) + 1, distance=DistanceBWPeaksx, height=xPeakHeight)[0]
        try:
            self.Peaksx = self.Peaksx[np.where(self.Peaksy[0] > self.Peaksx)[0][-1]:]
        except Exception as e:
            # print(e)
            self.Peaksy = self.Peaksy[1:]
            self.Peaksx = self.Peaksx[np.where(self.Peaksy[0] > self.Peaksx)[0][-1]:]
        self.BioZPeaksXEqv = self.FindEquivalentIndexFromTimeArray(self.IMU.Data[self.Peaksx,0], self.BioZ1.Data[0,:,0])
        self.BioZPeaksYEqv = self.FindEquivalentIndexFromTimeArray(self.IMU.Data[self.Peaksy,0], self.BioZ1.Data[0,:,0])
        if plot_rawdata:
            fig, ax = plt.subplots(3)
            ax[0].plot(self.IMU.Data[:,2])
            ax[0].scatter(self.Peaksy,self.IMU.Data[self.Peaksy,2])
            ax[1].plot(self.IMU.Data[:,1])
            ax[1].scatter(self.Peaksx,self.IMU.Data[self.Peaksx,1])
            ax[1].plot(self.IMU.Data[:,7])
            ax[1].scatter(self.Peaksx,self.IMU.Data[self.Peaksx,7])
            if AxisDirection == -1:
                ax[1].plot(1-(self.IMU.Data[:,7]+self.IMU.Data[:,1]))
            else:
                ax[1].plot(1+(self.IMU.Data[:,1]-self.IMU.Data[:,7]))
            ax[2].plot(self.BioZ1.Data[0,:,1])
            ax[2].scatter(self.BioZPeaksXEqv,self.BioZ1.Data[0,self.BioZPeaksXEqv,1])
            ax[2].scatter(self.BioZPeaksYEqv,self.BioZ1.Data[0,self.BioZPeaksYEqv,1])
            plt.tight_layout()
            plt.show()
        self.BioZFlexion = []
        self.BioZExtension = []
        self.IMUFlexion = []
        self.IMUExtension = []
        self.AngleExtension = []
        self.AngleFlexion = []
        self.IMUFlexionDS = []
        self.IMUExtensionDS = []
        self.AngleFlexionDS = []
        self.AngleExtensionDS = []
        if plot_rawdata:
            fig, ax = plt.subplots(2,figsize=(9,7))
        for j in range(len(self.Peaksx) - 1):
            try:
                self.BioZExtension.append(copy.deepcopy(self.BioZ1.Data[:,self.BioZPeaksXEqv[j]:self.BioZPeaksYEqv[j],:]))
                self.BioZFlexion.append(copy.deepcopy(self.BioZ1.Data[:,self.BioZPeaksYEqv[j]:self.BioZPeaksXEqv[j + 1],:]))
                self.IMUExtension.append(copy.deepcopy(self.IMU.Data[self.Peaksx[j]:self.Peaksy[j],:]))
                self.IMUFlexion.append(copy.deepcopy(self.IMU.Data[self.Peaksy[j]:self.Peaksx[j + 1],:]))
                self.AngleExtension.append(90 *(self.IMUExtension[j][:, 2]  -self.IMUExtension[j][:, 7]))
                self.AngleFlexion.append(90 * (self.IMUFlexion[j][:, 2] - self.IMUFlexion[j][:, 7]))
                FlexionIMUIndex = self.FindEquivalentIndexFromTimeArray(self.BioZFlexion[j][0,:,0], self.IMUFlexion[j][:,0])
                self.IMUFlexionDS.append(self.IMUFlexion[j][FlexionIMUIndex,:])
                self.AngleFlexionDS.append(self.AngleFlexion[j][FlexionIMUIndex] - self.AngleFlexion[j][FlexionIMUIndex][0])
                ExtensionIMUIndex = self.FindEquivalentIndexFromTimeArray(self.BioZExtension[j][0, :, 0],
                                                                        self.IMUExtension[j][:,0])
                self.IMUExtensionDS.append(self.IMUExtension[j][ExtensionIMUIndex, :])
                self.AngleExtensionDS.append(self.AngleExtension[j][ExtensionIMUIndex]-self.AngleExtension[j][ExtensionIMUIndex][-1])
                if plot_rawdata:
                    ax[0].plot(self.BioZExtension[j][0,:,1]-self.BioZExtension[j][0,-1,1], self.AngleExtensionDS[j], color='blue')
                    ax[0].plot(self.BioZFlexion[j][0,:,1]-self.BioZFlexion[j][0,0,1], self.AngleFlexionDS[j], color='red')
                    ax[1].plot(self.BioZExtension[j][1,:,1]-self.BioZExtension[j][1,-1,1], self.AngleExtensionDS[j], color='blue')
                    ax[1].plot(self.BioZFlexion[j][1,:,1]-self.BioZFlexion[j][1,0,1], self.AngleFlexionDS[j], color='red')
            except Exception as e:
                print(e)
        if plot_rawdata:
            plt.tight_layout()
            plt.show()
        #2nd find equivalent ones in BioZ data
        #Segment bioz array and IMU array by peaks
        #Downsample IMU data to match the data points of bioz
    def FindEquivalentIndexFromTimeArray(self, BaseArray, ArrayToMatch):
        MatchingIndex = []
        for i in range(len(BaseArray)):
            MatchingIndex.append(np.argmin(np.abs(ArrayToMatch-BaseArray[i])))
        MatchingIndex = np.array(MatchingIndex)
        return MatchingIndex

    def GetMatchingAngleIndex(self, EAngles, FAngles,UpToAngle=60, AngleDiffToAccept=5):
        IndexToStopAt = np.where(FAngles < UpToAngle)[0][-1]
        FAnglesToMatch = FAngles[:IndexToStopAt]
        FIndex = []
        EIndex = []
        for i in range(IndexToStopAt):
            AToS = FAnglesToMatch[i]
            Index = np.argmin(np.abs(AToS - EAngles))
            # print(AToS, Index, FE.KneeExtensionAngle[CycleIndex][Index])
            if np.abs(AToS - EAngles[Index]) < AngleDiffToAccept:
                FIndex.append(i)
                EIndex.append(Index)
        return FIndex, EIndex


class FE_Analysis_v3:
    def __init__(self, SS):
        self.SS = SS
    def SegmentFECycles(self, AxisDirection,FilterWindow = 21, DistanceBWPeaks=300, plot_rawdata=False):
        #First filter bioz data
        self.BioZ1 = copy.deepcopy(self.SS.BioZ1)
        self.IMU = copy.deepcopy(self.SS.IMU)
        self.BioZ1.Data[0,:,1] = signal.savgol_filter(self.BioZ1.Data[0,:,1], FilterWindow, 3)
        self.BioZ1.Data[0,:,2] = signal.savgol_filter(self.BioZ1.Data[0,:,2], FilterWindow, 3)
        self.BioZ1.Data[1,:,1] = signal.savgol_filter(self.BioZ1.Data[1,:,1], FilterWindow, 3)
        self.BioZ1.Data[1,:,2] = signal.savgol_filter(self.BioZ1.Data[1,:,2], FilterWindow, 3)
        for i in range(12):
            self.IMU.Data[:,i+1] = signal.savgol_filter(self.IMU.Data[:,i+1], FilterWindow,3)
        self.ThighAngle = np.abs(90+(180/np.pi)*np.arctan2((np.pi/180)*90*self.IMU.Data[:,8], (np.pi/180)*90*self.IMU.Data[:,7]))
        self.ShankAngle = np.abs((180/np.pi)*np.arctan2((np.pi/180)*90*self.IMU.Data[:,2], (np.pi/180)*90*self.IMU.Data[:,1]))
        self.KneeAngle = -1*AxisDirection*(self.ShankAngle-self.ThighAngle)
        #2nd is find peaks in IMU data
        self.ZeroDegreePeak = signal.find_peaks((-1 * self.KneeAngle)+90, distance=DistanceBWPeaks, height=50)[0]
        print(self.ZeroDegreePeak)
        self.NintyDegreePeak = signal.find_peaks((self.KneeAngle), distance=DistanceBWPeaks, height=50)[0]
        print(self.NintyDegreePeak)
        while(True):
            try:
                self.NintyDegreePeak = self.NintyDegreePeak[np.where(self.ZeroDegreePeak[0] > self.NintyDegreePeak)[0][-1]:]
                break
            except Exception as e:
                print(e)
                self.ZeroDegreePeak = self.ZeroDegreePeak[1:]

        print(self.ZeroDegreePeak)
        print(self.NintyDegreePeak)

        #     self.NintyDegreePeak = self.NintyDegreePeak[1:]
        #     self.ZeroDegreePeak = self.ZeroDegreePeak[np.where(self.ZeroDegreePeak[0] > self.ZeroDegreePeak)[0][-1]:]
        self.BioZPeaks0Deg = self.FindEquivalentIndexFromTimeArray(self.IMU.Data[self.ZeroDegreePeak,0], self.BioZ1.Data[0,:,0])
        self.BioZPeaks90Deg = self.FindEquivalentIndexFromTimeArray(self.IMU.Data[self.NintyDegreePeak,0], self.BioZ1.Data[0,:,0])
        if plot_rawdata:
            fig, ax = plt.subplots(3)
            ax[0].plot(self.ThighAngle)
            ax[0].scatter(self.ZeroDegreePeak,self.ThighAngle[self.ZeroDegreePeak], marker='x')
            ax[0].scatter(self.NintyDegreePeak,self.ThighAngle[self.NintyDegreePeak], marker='o')
            ax[1].plot(self.ShankAngle)
            ax[1].scatter(self.ZeroDegreePeak,self.ShankAngle[self.ZeroDegreePeak], marker='x')
            ax[1].scatter(self.NintyDegreePeak,self.ShankAngle[self.NintyDegreePeak], marker='o')
            ax[2].plot(self.KneeAngle)
            ax[2].scatter(self.ZeroDegreePeak,self.KneeAngle[self.ZeroDegreePeak], marker='x')
            ax[2].scatter(self.NintyDegreePeak,self.KneeAngle[self.NintyDegreePeak], marker='o')
            plt.tight_layout()
            plt.show()
        self.BioZFlexion = []
        self.BioZExtension = []
        self.BioZExtensionNorm = []
        self.BioZFlexionNorm = []
        self.IMUFlexion = []
        self.IMUExtension = []
        self.AngleExtension = []
        self.AngleFlexion = []
        self.IMUFlexionDS = []
        self.IMUExtensionDS = []
        self.AngleFlexionDS = []
        self.AngleExtensionDS = []
        if plot_rawdata:
            fig, ax = plt.subplots(2,figsize=(9,7))
        for j in range(len(self.NintyDegreePeak) - 1):
            try:
                self.BioZExtension.append(copy.deepcopy(self.BioZ1.Data[:,self.BioZPeaks90Deg[j]:self.BioZPeaks0Deg[j],:]))
                self.BioZFlexion.append(copy.deepcopy(self.BioZ1.Data[:,self.BioZPeaks0Deg[j]:self.BioZPeaks90Deg[j + 1],:]))
                self.IMUExtension.append(copy.deepcopy(self.IMU.Data[self.NintyDegreePeak[j]:self.ZeroDegreePeak[j],:]))
                self.IMUFlexion.append(copy.deepcopy(self.IMU.Data[self.ZeroDegreePeak[j]:self.NintyDegreePeak[j + 1],:]))
                self.AngleExtension.append(self.KneeAngle[self.NintyDegreePeak[j]:self.ZeroDegreePeak[j]])
                self.AngleFlexion.append(self.KneeAngle[self.ZeroDegreePeak[j]:self.NintyDegreePeak[j+1]])
                FlexionIMUIndex = self.FindEquivalentIndexFromTimeArray(self.BioZFlexion[j][0,:,0], self.IMUFlexion[j][:,0])
                self.IMUFlexionDS.append(self.IMUFlexion[j][FlexionIMUIndex,:])
                self.AngleFlexionDS.append(self.AngleFlexion[j][FlexionIMUIndex] - self.AngleFlexion[j][FlexionIMUIndex][0])
                ExtensionIMUIndex = self.FindEquivalentIndexFromTimeArray(self.BioZExtension[j][0, :, 0],
                                                                        self.IMUExtension[j][:,0])
                self.IMUExtensionDS.append(self.IMUExtension[j][ExtensionIMUIndex, :])
                self.AngleExtensionDS.append(self.AngleExtension[j][ExtensionIMUIndex]-self.AngleExtension[j][ExtensionIMUIndex][-1])

                self.BioZFlexionNorm.append(copy.deepcopy(self.BioZFlexion[j]))
                self.BioZExtensionNorm.append(copy.deepcopy(self.BioZExtension[j]))
                self.BioZFlexionNorm[j][0,:,1] = self.Normalize( self.BioZFlexionNorm[j][0,:,1],0)
                self.BioZFlexionNorm[j][1,:,1] = self.Normalize( self.BioZFlexionNorm[j][1,:,1],0)
                self.BioZFlexionNorm[j][0,:,2] = self.Normalize( self.BioZFlexionNorm[j][0,:,2],0)
                self.BioZFlexionNorm[j][1,:,2] = self.Normalize( self.BioZFlexionNorm[j][1,:,2],0)

                self.BioZExtensionNorm[j][0,:,1] = self.Normalize( self.BioZExtensionNorm[j][0,:,1],-1)
                self.BioZExtensionNorm[j][1,:,1] = self.Normalize( self.BioZExtensionNorm[j][1,:,1],-1)
                self.BioZExtensionNorm[j][0,:,2] = self.Normalize( self.BioZExtensionNorm[j][0,:,2],-1)
                self.BioZExtensionNorm[j][1,:,2] = self.Normalize( self.BioZExtensionNorm[j][1,:,2],-1)

                if plot_rawdata:
                    # ax[0].plot(self.BioZExtension[j][0,:,1]-self.BioZExtension[j][0,-1,1], self.AngleExtensionDS[j], color='blue')
                    # ax[0].plot(self.BioZFlexion[j][0,:,1]-self.BioZFlexion[j][0,0,1], self.AngleFlexionDS[j], color='red')
                    ax[0].plot(self.BioZExtension[j][1,:,1]-self.BioZExtension[j][1,-1,1], self.AngleExtensionDS[j], color='blue')
                    ax[0].plot(self.BioZFlexion[j][1,:,1]-self.BioZFlexion[j][1,0,1], self.AngleFlexionDS[j], color='red')
                    ax[1].plot(self.BioZExtensionNorm[j][1,:,1], self.AngleExtensionDS[j], color='blue')
                    ax[1].plot(self.BioZFlexionNorm[j][1,:,1], self.AngleFlexionDS[j], color='red')
            except Exception as e:
                print(e)
        if plot_rawdata:
            ax[0].invert_yaxis()
            ax[1].invert_yaxis()
            ax[0].invert_xaxis()
            ax[1].invert_xaxis()
            plt.tight_layout()
            plt.show()
        #2nd find equivalent ones in BioZ data
        #Segment bioz array and IMU array by peaks
        #Downsample IMU data to match the data points of bioz
    def FindEquivalentIndexFromTimeArray(self, BaseArray, ArrayToMatch):
        MatchingIndex = []
        for i in range(len(BaseArray)):
            MatchingIndex.append(np.argmin(np.abs(ArrayToMatch-BaseArray[i])))
        MatchingIndex = np.array(MatchingIndex)
        return MatchingIndex

    def GetMatchingAngleIndex(self, EAngles, FAngles,UpToAngle=60, AngleDiffToAccept=5):
        IndexToStopAt = np.where(FAngles < UpToAngle)[0][-1]
        FAnglesToMatch = FAngles[:IndexToStopAt]
        FIndex = []
        EIndex = []
        for i in range(IndexToStopAt):
            AToS = FAnglesToMatch[i]
            Index = np.argmin(np.abs(AToS - EAngles))
            # print(AToS, Index, FE.KneeExtensionAngle[CycleIndex][Index])
            if np.abs(AToS - EAngles[Index]) < AngleDiffToAccept:
                FIndex.append(i)
                EIndex.append(Index)
        return FIndex, EIndex
    def AnalyzeDifferenceFromLine(self, StartAngle, EndAngle, Resolution,Frequency, plot_data=True):
        AnglesToAnalyze = np.arange(StartAngle,EndAngle+Resolution,Resolution)
        BioZFlexions = []
        BioZExtensions = []
        BioZFlexionsNorm = []
        BioZExtensionsNorm = []
        BioZFlexionsNorm2 = []
        BioZExtensionsNorm2 = []
        BioZDiff = []
        BioZFAdjusted = []
        BioZEAdjusted = []
        for i in range(len(self.AngleExtensionDS)):
            try:
                FIndex = np.argmin(np.abs(self.AngleFlexionDS[i][:, np.newaxis] - AnglesToAnalyze), axis=0)
                EIndex = np.argmin(np.abs(self.AngleExtensionDS[i][:, np.newaxis] - AnglesToAnalyze), axis=0)
                # ax[0].scatter(FE.BioZFlexion[i][0,FIndex,1], FE.AngleFlexionDS[i][FIndex], color=colors[j%4])
                BioZFlexions.append(self.BioZFlexion[i][Frequency, FIndex, 1])
                BioZExtensions.append(self.BioZExtension[i][Frequency, EIndex, 1])
                BioZFlexionsNorm.append(self.BioZFlexionNorm[i][Frequency,FIndex,1])
                BioZExtensionsNorm.append(self.BioZExtensionNorm[i][Frequency,EIndex,1])
                BioZExtensionsNorm2.append(self.Normalize(self.BioZExtension[i][Frequency, EIndex, 1], 0))
                BioZFlexionsNorm2.append(self.Normalize(self.BioZFlexion[i][Frequency, FIndex, 1], 0))
                # BioZDiff.append(100.0*(BioZFlexions[i]-BioZExtensions[i])/BioZFlexions[0])
                BioZFAdjusted.append(
                    BioZFlexions[i] - np.linspace(BioZFlexions[i][0], BioZFlexions[i][-1], len(AnglesToAnalyze)))
                BioZEAdjusted.append(
                    np.linspace(BioZExtensions[i][0], BioZExtensions[i][-1], len(AnglesToAnalyze)) - BioZExtensions[i])
                BioZDiff.append(BioZFAdjusted[i] - BioZEAdjusted[i])
                # BioZFAdjusted.append(BioZFlexions[i]-np.ones(len(AnglesToAnalyze))*BioZFlexions[i][0])
                # BioZEAdjusted.append(np.ones(len(AnglesToAnalyze))*BioZExtensions[i][0]-BioZExtensions[i])
                # BioZDiff.append(BioZFAdjusted[i]-BioZEAdjusted[i])
                # ax[0].plot(BioZFlexions[i],AnglesToAnalyze, color='red')
                # ax[0].scatter(BioZFlexions[i],AnglesToAnalyze,  color='red', marker='o')
                # ax[0].plot(BioZExtensions[i],AnglesToAnalyze, color='blue')
                # ax[0].scatter(BioZExtensions[i],AnglesToAnalyze,color='blue', marker='x')
                # # ax[1].scatter(AnglesToAnalyze,BioZFlexions[i]-BioZExtensions[i], color=colors[j % 4])
                # ax[1].scatter(AnglesToAnalyze, BioZDiff[i], color=colors[j % 4])
                # ax[2].scatter(AnglesToAnalyze, BioZFAdjusted[i], marker='o', color=colors[j % 4])
                # ax[2].scatter(AnglesToAnalyze, BioZEAdjusted[i], marker='x', color=colors[j % 4])
            except Exception as e:
                print(e)
        self.BioZAtAngleFlexions = np.array(BioZFlexions)
        self.BioZAtAngleExtensions = np.array(BioZExtensions)
        self.BioZAtAngleExtensionsNorm = np.array(BioZExtensionsNorm)
        self.BioZAtAngleFlexionsNorm = np.array(BioZFlexionsNorm)
        self.BioZAtAngleExtensionsNorm2 = np.array(BioZExtensionsNorm2)
        self.BioZAtAngleFlexionsNorm2 = np.array(BioZFlexionsNorm2)
        self.BioZAtAngleDiff = np.array(BioZDiff)
        self.BioZFAdjusted = np.array(BioZFAdjusted)
        self.BioZEAdjusted = np.array(BioZEAdjusted)
        if plot_data == True:
            fig, ax = plt.subplots(4, figsize=((9,7)))
            ax[0].plot(self.BioZAtAngleFlexions.mean(axis=0), AnglesToAnalyze)
            ax[0].scatter(self.BioZAtAngleFlexions.mean(axis=0), AnglesToAnalyze, marker='o')
            ax[0].plot(self.BioZAtAngleExtensions .mean(axis=0), AnglesToAnalyze)
            ax[0].scatter(self.BioZAtAngleExtensions .mean(axis=0), AnglesToAnalyze, marker='x')
            ax[1].scatter(AnglesToAnalyze, self.BioZAtAngleDiff .mean(axis=0), )
            ax[2].scatter(AnglesToAnalyze, self.BioZFAdjusted .mean(axis=0), marker='o')
            ax[2].scatter(AnglesToAnalyze, self.BioZEAdjusted.mean(axis=0), marker='x')
            # ax[3].scatter( self.BioZAtAngleFlexionsNorm.mean(axis=0),AnglesToAnalyze, marker='o')
            # ax[3].scatter( self.BioZAtAngleExtensionsNorm.mean(axis=0),AnglesToAnalyze, marker='x')
            ax[3].plot(self.BioZAtAngleFlexionsNorm.mean(axis=0),AnglesToAnalyze)
            ax[3].plot(self.BioZAtAngleExtensionsNorm.mean(axis=0),AnglesToAnalyze)
            plt.tight_layout()
            plt.show()
    def Normalize(self, Array, IndexToNormTo):
        return ((Array-Array[IndexToNormTo])/(Array.max()-Array.min()))

