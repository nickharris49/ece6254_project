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
from Utilities.FileParser import *
from sklearn.preprocessing import PolynomialFeatures



def eval2R1C(load, f):
    if load[1] !=0 and load[2]!=0:
        w = 2 * math.pi * f
        Z = 1 / (1 / load[0] + 1 / (load[1] + 1 / (1j * w * load[2] * (10 ** -9))))
        return (Z.real, Z.imag)
    else:
        Z = np.zeros(f.shape[0], dtype=complex)
        Z.real = load[0]
        return (Z.real, Z.imag)

def find_nearest(frequency, FrequencyArray):
        return np.abs(frequency-FrequencyArray).argmin()


class Calibration:
    def __init__(self, DeviceID, PolynomialDegrees=1, FirmwareVersion=1):
        self.PolynomialDegrees = PolynomialDegrees
        self.RealLinearModels = []
        self.ImagLinearModels = []
        if FirmwareVersion == 3:
            NumMeasurements = 7
            CalibrationDataSets = []
            for i in range(NumMeasurements):
                CalibrationDataSets.append(AdvancedLoggingRawDataParser('Data/CalibrationData/C001/run00%i'%(i+1)))
            self.Frequency = np.zeros([1+int((int(CalibrationDataSets[0].JsonDict['BioZ0']['endFreq'])-int(CalibrationDataSets[0].JsonDict['BioZ0']['startFreq']))/int(CalibrationDataSets[0].JsonDict['BioZ0']['stepFreq']))])
            for i in range(self.Frequency.shape[0]):
                self.Frequency[i] = int(CalibrationDataSets[0].JsonDict['BioZ0']['startFreq']) + i * int(CalibrationDataSets[0].JsonDict['BioZ0']['stepFreq'])
            dfr = pd.read_csv('Data/CalibrationData/AllImpedancesActualValues.csv')  # collected data
            ImpedanceActualValues = dfr.to_numpy()[:NumMeasurements]

            MeasuredReal = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ0.shape[0]])
            MeasuredImag = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ0.shape[0]])
            ActualReal = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ0.shape[0]])
            ActualImag = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ0.shape[0]])

            dfr = pd.read_csv('Data/CalibrationData/AllImpedancesActualValues.csv')  # collected data
            ImpedanceActualValues = dfr.to_numpy()
            for i in range(len(CalibrationDataSets)):
                MeasuredReal[i] = CalibrationDataSets[i].BioZ0[:, :, 1].mean(axis=1)
                MeasuredImag[i] = CalibrationDataSets[i].BioZ0[:, :, 2].mean(axis=1)
                ActualReal[i], ActualImag[i] = eval2R1C(ImpedanceActualValues[i], self.Frequency)
        else:
            if FirmwareVersion == 1:
                DeviceID = 'A008'
            else:
                DeviceID = 'B001'
            CalibrationDataSets = parse_file('Data/CalibrationData/%s/MeasuredValues.txt' % DeviceID)
            self.Frequency = np.zeros([int(CalibrationDataSets[0].JsonDict['BioZ']['NumFrequencies'])])
            for i in range(self.Frequency.shape[0]):
                self.Frequency[i] = int(CalibrationDataSets[0].JsonDict['BioZ']['StartFrequency']) + i * int(
                    CalibrationDataSets[0].JsonDict['BioZ']['Resolution'])
            MeasuredReal = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
            MeasuredImag = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
            ActualReal = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
            ActualImag = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])

            dfr = pd.read_csv('Data/CalibrationData/AllImpedancesActualValues.csv')  # collected data
            ImpedanceActualValues = dfr.to_numpy()
            for i in range(len(CalibrationDataSets)):
                MeasuredReal[i] = CalibrationDataSets[i].BioZ[:, :, 1].mean(axis=1)
                MeasuredImag[i] = CalibrationDataSets[i].BioZ[:, :, 2].mean(axis=1)
                ActualReal[i], ActualImag[i] = eval2R1C(ImpedanceActualValues[i], self.Frequency)

        Poly = PolynomialFeatures(degree=self.PolynomialDegrees)

        for i in range(self.Frequency.shape[0]):
            Xpoly = Poly.fit_transform(np.concatenate((MeasuredReal[:, i].reshape(MeasuredReal.shape[0], 1),
                                                       MeasuredImag[:, i].reshape(MeasuredImag.shape[0], 1)), axis=1))

            self.RealLinearModels.append(linear_model.LinearRegression(fit_intercept=False))
            self.ImagLinearModels.append(linear_model.LinearRegression(fit_intercept=False))

            self.RealLinearModels[i].fit(Xpoly, ActualReal[:, i])
            self.ImagLinearModels[i].fit(Xpoly, ActualImag[:, i])
    def predict(self, MeasuredReal, MeasuredImag, FrequenciesToPredict):
        i = 0
        CalibratedReal = np.zeros(MeasuredReal.shape)
        CalibratedImag = np.zeros(MeasuredImag.shape)
        Poly = PolynomialFeatures(degree=self.PolynomialDegrees)
        for Frequency in FrequenciesToPredict:
            FrequencyIndex = find_nearest(Frequency, self.Frequency)
            Xpoly = Poly.fit_transform(np.concatenate((MeasuredReal[i,:].reshape(MeasuredReal.shape[1], 1),
                                                       MeasuredImag[i,:].reshape(MeasuredImag.shape[1], 1)), axis=1))
            CalibratedReal[i, :] = self.RealLinearModels[FrequencyIndex].predict(Xpoly)
            CalibratedImag[i, :] = self.ImagLinearModels[FrequencyIndex].predict(Xpoly)
            i = i+1
        return CalibratedReal,CalibratedImag


