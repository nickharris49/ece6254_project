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
from scipy.optimize import curve_fit
def eval2R1C(load, f):
    if load[1] !=0 and load[2]!=0:
        w = 2 * math.pi * f
        Z = 1 / (1 / load[0] + 1 / (load[1] + 1 / (1j * w * load[2] * (10 ** -9))))
        return (Z.real, Z.imag)
    else:
        Z = np.zeros(f.shape[0], dtype=complex)
        Z.real = load[0]
        return (Z.real, Z.imag)

def fit_data(MeasuredData, x_real, x_imag):
    A = np.ones([MeasuredData.shape[0], 3])
    A[:,0] = MeasuredData.real
    A[:,1] = MeasuredData.imag
    PredictedReal = np.dot(A, x_real)
    PredictedImag = np.dot(A, x_imag)
    return (PredictedReal +1j*PredictedImag)

def train_raw_data(ImpedanceData, MeasuredRawData):
    A = np.ones([MeasuredRawData.shape[0], MeasuredRawData.shape[1]+1])
    A[:,:-1] = MeasuredRawData
    x_real, resid, rank, s = np.linalg.lstsq(A, ImpedanceData.real)
    x_imag, resid, rank, s = np.linalg.lstsq(A, ImpedanceData.imag)
    return x_real, x_imag
def fit_raw_data(MeasuredRawData, x_real, x_imag):
    A = np.ones([MeasuredRawData.shape[0], MeasuredRawData.shape[1]+1])
    A[:,:-1] = MeasuredRawData
    PredictedReal = np.dot(A, x_real)
    PredictedImag = np.dot(A, x_imag)
    return (PredictedReal +1j*PredictedImag)


def datasheet_calibration(CoeffMag, CoeffPhase, MeasuredImpedance):
    CalibratedMag = np.abs(MeasuredImpedance)*CoeffMag
    CalibratedPhase = np.angle(MeasuredImpedance) + CoeffPhase
    return (CalibratedMag*np.cos(CalibratedPhase) - 1j*CalibratedMag*np.sin(CalibratedPhase))

def train_data(ImpedanceData, MeasuredData):
    A = np.ones([MeasuredData.shape[0], 3])
    A[:, 0] = MeasuredData.real
    A[:, 1] = MeasuredData.imag
    x_real, resid, rank, s = np.linalg.lstsq(A, ImpedanceData.real)
    x_imag, resid, rank, s = np.linalg.lstsq(A, ImpedanceData.imag)
    return x_real, x_imag

def AD5940_Calibrate_Data(RawMeasuredImpedanceData):
    NumFrequencies = 257
    Frequencies = np.arange(5000, 100000, int((100000-5000)/(NumFrequencies-1)))

    dfr = pd.read_csv('Data/AD5940/AD5940_Calibration_Full_Sweep.csv')  # collected data
    ImpData = dfr.as_matrix()
    MeasuredCalibrationImpedance = np.zeros([ImpData.shape[0], int(ImpData.shape[1]/4)], dtype=complex)
    for i in range(NumFrequencies):
        MeasuredCalibrationImpedance[:,i] = (ImpData[:,2 + i*4]-1j*ImpData[:,3+ i*4])/(ImpData[:,0+ i*4]-1j*ImpData[:,1+ i*4])


    dfr = pd.read_csv('Data/AD5940/AllImpedancesActualValues.csv')  # collected data
    CalibrationImpedanceActualValues = dfr.as_matrix()
    CalibrationImpedanceReal = np.zeros([CalibrationImpedanceActualValues.shape[0],NumFrequencies])
    CalibrationImpedanceImag = np.zeros([CalibrationImpedanceActualValues.shape[0],NumFrequencies])
    for i in range(CalibrationImpedanceReal.shape[0]):
        CalibrationImpedanceReal[i], CalibrationImpedanceImag[i] = eval2R1C(CalibrationImpedanceActualValues[i], Frequencies)
    CalibrationImpedance = CalibrationImpedanceReal -1j*CalibrationImpedanceImag
    x_real = np.zeros([NumFrequencies,3])
    x_imag = np.zeros([NumFrequencies,3])
    for i in range(NumFrequencies):
        x_real[i] , x_imag[i] = train_data(CalibrationImpedance[:,i], MeasuredCalibrationImpedance[:,i])

    RawMeasuredImpedance = np.zeros([RawMeasuredImpedanceData.shape[0], int(RawMeasuredImpedanceData.shape[1] / 4)],
                                    dtype=complex)
    for i in range(NumFrequencies):
        RawMeasuredImpedance[:, i] = (RawMeasuredImpedanceData[:, 2 + i * 4] - 1j * RawMeasuredImpedanceData[:,
                                                                                    3 + i * 4]) / (
                                                 RawMeasuredImpedanceData[:, 0 + i * 4] - 1j * RawMeasuredImpedanceData[
                                                                                               :, 1 + i * 4])

    ProcessedImpedance = np.zeros(RawMeasuredImpedance.shape, dtype=complex)

    for i in range(NumFrequencies):
        ProcessedImpedance[:, i] = fit_data(RawMeasuredImpedance[:, i], x_real[i], x_imag[i])

    return ProcessedImpedance

#
# dfr = pd.read_csv('Data/CaitlinData/CaitlinKendell.csv')  # collected data
# RawMeasuredImpedanceData = dfr.as_matrix()\
#
# ProcessedImpedance = AD5940_Calibrate_Data(RawMeasuredImpedanceData)
#
# ElectrodeImp1 = (ProcessedImpedance[1,:]-ProcessedImpedance[0,:])/2
# ElectrodeImp2 = (ProcessedImpedance[2,:]-ProcessedImpedance[0,:])/2
# ElectrodeImp3 = (ProcessedImpedance[3,:]-ProcessedImpedance[0,:])
# ElectrodeImp4 = (ProcessedImpedance[4,:]-ProcessedImpedance[0,:])
#
#
#
# fig, ax = plt.subplots(1)
# ax.scatter(ElectrodeImp1.real, ElectrodeImp1.imag)
# ax.scatter(ElectrodeImp2.real, ElectrodeImp2.imag)
# ax.scatter(ElectrodeImp3.real, ElectrodeImp3.imag)
# ax.scatter(ElectrodeImp4.real, ElectrodeImp4.imag)
# plt.show()

