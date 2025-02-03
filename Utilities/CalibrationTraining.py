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
from Utilities.FileParser import parse_file
from Utilities.Calibration import eval2R1C
from sklearn.preprocessing import PolynomialFeatures

from mpl_toolkits.mplot3d import Axes3D

def generate_calibration_coeffecients(DeviceID):
    CalibrationDataSets = parse_file('Data/CalibrationData/%s/MeasuredValues.txt' % DeviceID)
    Frequency = np.zeros([int(CalibrationDataSets[0].JsonDict['BioZ']['NumFrequencies'])])
    for i in range(Frequency.shape[0]):
        Frequency[i] = int(CalibrationDataSets[0].JsonDict['BioZ']['StartFrequency']) + i*int(CalibrationDataSets[0].JsonDict['BioZ']['Resolution'])
    MeasuredReal = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
    MeasuredImag = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
    ActualReal = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
    ActualImag = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
    RealCoeffecients = np.zeros([Frequency.shape[0], 3])
    ImagCoeffecients = np.zeros(RealCoeffecients.shape)
    dfr = pd.read_csv('Data/CalibrationData/AllImpedancesActualValues.csv')  # collected data
    ImpedanceActualValues = dfr.to_numpy()
    for i in range(len(CalibrationDataSets)):
        MeasuredReal[i] = CalibrationDataSets[i].BioZ[:,:,1].mean(axis=1)
        MeasuredImag[i] = CalibrationDataSets[i].BioZ[:,:,2].mean(axis=1)
        ActualReal[i], ActualImag[i] = eval2R1C(ImpedanceActualValues[i], Frequency)

    bias = np.ones([MeasuredReal.shape[0], 1])

    for i in range(Frequency.shape[0]):
        XLinear = np.concatenate((MeasuredReal[:, i].reshape(MeasuredReal.shape[0], 1),
                                  MeasuredImag[:, i].reshape(MeasuredImag.shape[0], 1), bias),
                                 axis=1)

        RealLinearRegression = linear_model.LinearRegression(fit_intercept=False)
        ImagLinearRegression = linear_model.LinearRegression(fit_intercept=False)

        RealLinearRegression.fit(XLinear, ActualReal[:, i])
        ImagLinearRegression.fit(XLinear, ActualImag[:, i])

        RealCoeffecients[i] = RealLinearRegression.coef_
        ImagCoeffecients[i] = ImagLinearRegression.coef_

    TestReal = np.zeros(ActualReal.shape)
    TestImag = np.zeros(ActualImag.shape)

    ErrorReal = np.zeros(Frequency.shape)
    ErrorImag = np.zeros(Frequency.shape)
    for i in range(Frequency.shape[0]):
        TestReal[:, i] = MeasuredReal[:, i] * RealCoeffecients[i, 0] + MeasuredImag[:, i] * \
                         RealCoeffecients[i, 1] + RealCoeffecients[i, 2]
        TestImag[:, i] = MeasuredReal[:, i] * ImagCoeffecients[i, 0] + MeasuredImag[:, i] * \
                         ImagCoeffecients[i, 1] + ImagCoeffecients[i, 2]
        ErrorReal[i] = mean_squared_error(TestReal[:, i], ActualReal[:, i])
        ErrorImag[i] = mean_squared_error(TestImag[:, i], ActualImag[:, i])
    np.savetxt("Data/CalibrationData/%s/RealCoeffecients.csv"%DeviceID, RealCoeffecients, delimiter=',')
    np.savetxt("Data/CalibrationData/%s/ImagCoeffecients.csv"%DeviceID, ImagCoeffecients, delimiter=',')
    fig, ax = plt.subplots(2)
    ax[0].plot(ErrorReal)
    ax[1].plot(ErrorImag)
    return RealCoeffecients, ImagCoeffecients

def second_order_polynomial_calibration(DeviceID):
    DeviceID = 'A008'
    Indicies = [0,1,2,3,4,5,6,7,8,9,10,11, 12, 13,14, 15,18, 19, 20, 21,22,23,24,25,26]
    CalibrationDataSets = parse_file('Data/CalibrationData/%s/MeasuredValues.txt' % DeviceID)
    # for set in CalibrationDataSets:
    #     fig, ax = plt.subplots(1)
    #     ax.plot(set.BioZ[:,0,1], -1*set.BioZ[:,0,2])
    #     plt.show()
    TempCalibrationDataSets = CalibrationDataSets
    CalibrationDataSets = []
    for Index in Indicies:
        CalibrationDataSets.append(TempCalibrationDataSets[Index])
    Frequency = np.zeros([int(CalibrationDataSets[0].JsonDict['BioZ']['NumFrequencies'])])
    for i in range(Frequency.shape[0]):
        Frequency[i] = int(CalibrationDataSets[0].JsonDict['BioZ']['StartFrequency']) + i*int(CalibrationDataSets[0].JsonDict['BioZ']['Resolution'])
    MeasuredReal = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
    MeasuredImag = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
    ActualReal = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
    ActualImag = np.zeros([len(CalibrationDataSets), CalibrationDataSets[0].BioZ.shape[0]])
    TestReal = np.zeros(ActualReal.shape)
    TestImag = np.zeros(ActualImag.shape)
    ErrorReal = np.zeros(Frequency.shape)
    ErrorImag = np.zeros(Frequency.shape)

    RealCoeffecients = np.zeros([Frequency.shape[0], 10])
    ImagCoeffecients = np.zeros(RealCoeffecients.shape)
    dfr = pd.read_csv('Data/CalibrationData/AllImpedancesActualValues.csv')  # collected data
    ImpedanceActualValues = dfr.to_numpy()
    TempImpedanceActualValues = ImpedanceActualValues
    ImpedanceActualValues = []
    for Index in Indicies:
        ImpedanceActualValues.append(TempImpedanceActualValues[Index])
    for i in range(len(CalibrationDataSets)):
        MeasuredReal[i] = CalibrationDataSets[i].BioZ[:,:,1].mean(axis=1)
        MeasuredImag[i] = CalibrationDataSets[i].BioZ[:,:,2].mean(axis=1)
        ActualReal[i], ActualImag[i] = eval2R1C(ImpedanceActualValues[i], Frequency)

    Poly = PolynomialFeatures(degree=3)

    RealLinearRegression = []
    ImagLinearRegression = []
    for i in range(Frequency.shape[0]):
        Xpoly = Poly.fit_transform(np.concatenate((MeasuredReal[:, i].reshape(MeasuredReal.shape[0], 1),
                                    MeasuredImag[:, i].reshape(MeasuredImag.shape[0], 1)), axis=1))



        RealLinearRegression.append(linear_model.LinearRegression(fit_intercept=False))
        ImagLinearRegression.append(linear_model.LinearRegression(fit_intercept=False))

        RealLinearRegression[i].fit(Xpoly, ActualReal[:, i])
        ImagLinearRegression[i].fit(Xpoly, ActualImag[:, i])


        TestReal[:, i] = RealLinearRegression[i].predict(Xpoly)
        TestImag[:, i] = ImagLinearRegression[i].predict(Xpoly)


        ErrorReal[i] = mean_squared_error(TestReal[:, i], ActualReal[:, i])
        ErrorImag[i] = mean_squared_error(TestImag[:, i], ActualImag[:, i])


    fig, ax = plt.subplots(2)
    ax[0].plot(Frequency,ErrorReal)
    ax[1].plot(Frequency,ErrorImag)
    plt.show()

    TestImpedanceDatasets = parse_file('Data/TestData/%s/MeasuredTestImpedanceFullSpectroscopy.txt' % DeviceID)
    TestMeasuredReal = np.zeros([len(TestImpedanceDatasets),TestImpedanceDatasets[0].BioZ.shape[0]])
    TestMeasuredImag = np.zeros([len(TestImpedanceDatasets),TestImpedanceDatasets[0].BioZ.shape[0]])
    dfr = pd.read_csv('Data/TestData/ActualTestImpedances.csv')  # collected data
    TestImpedanceActual = dfr.to_numpy()
    TestActualReal = np.zeros(TestMeasuredReal.shape)
    TestActualImag = np.zeros(TestMeasuredReal.shape)
    for i in range(len(TestImpedanceDatasets)):
        TestMeasuredReal[i] = TestImpedanceDatasets[i].BioZ[:,:,1].mean(axis=1)
        TestMeasuredImag[i] = TestImpedanceDatasets[i].BioZ[:, :, 2].mean(axis=1)
        TestActualReal[i], TestActualImag[i] = eval2R1C(TestImpedanceActual[i], Frequency)

    TestDataReal = np.zeros(TestActualReal.shape)
    TestDataImag = np.zeros(TestActualImag.shape)
    ErrorRealTest = np.zeros([Frequency.shape[0]])
    ErrorImagTest = np.zeros([Frequency.shape[0]])

    for i in range(Frequency.shape[0]):
        XpolyTest = Poly.fit_transform(np.concatenate((TestMeasuredReal[:, i].reshape(TestMeasuredReal.shape[0], 1),
                                                       TestMeasuredImag[:, i].reshape(TestMeasuredImag.shape[0], 1)), axis=1))
        TestDataReal[:, i] = RealLinearRegression[i].predict(XpolyTest)
        TestDataImag[:, i] = ImagLinearRegression[i].predict(XpolyTest)

        ErrorRealTest[i] = mean_squared_error(TestDataReal[:, i], TestActualReal[:, i])
        ErrorImagTest[i] = mean_squared_error(TestDataImag[:, i], TestActualImag[:, i])

    fig, ax = plt.subplots(2)
    ax[0].plot(Frequency,ErrorRealTest)
    ax[1].plot(Frequency,ErrorImagTest)
    plt.show()

    FrequencyIndex = 130
    X = np.arange(16000, -16000, -1000)
    Y = np.arange(16000, -16000, -1000)
    X,Y = np.meshgrid(X,Y)

    XX = X.flatten()
    YY = Y.flatten()
    Xpoly = Poly.fit_transform(np.concatenate((XX.reshape(XX.shape[0], 1),
                                    YY.reshape(YY.shape[0], 1)), axis=1))

    Z = RealLinearRegression[FrequencyIndex].predict(Xpoly).reshape(X.shape)

    x = np.zeros([len(Indicies)])
    y = np.zeros([len(Indicies)])
    Real = np.zeros([len(Indicies)])
    Imag = np.zeros([len(Indicies)])


    for i in range(len(Indicies)):
        x[i] = TempCalibrationDataSets[Indicies[i]].BioZ[FrequencyIndex,:,1].mean()
        y[i] = TempCalibrationDataSets[Indicies[i]].BioZ[FrequencyIndex,:,2].mean()
        Real[i] = eval2R1C(TempImpedanceActualValues[Indicies[i]], Frequency)[0][FrequencyIndex]
        Imag[i] = eval2R1C(TempImpedanceActualValues[Indicies[i]], Frequency)[1][FrequencyIndex]



    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x,y,Real)
    ax.scatter3D(TestMeasuredReal[:,FrequencyIndex], TestMeasuredImag[:, FrequencyIndex], TestActualReal[:, FrequencyIndex], marker='x')
    # for i in range(Real.shape[0]):
    #     ax.text(x[i], y[i], Real[i], '%i,%i'%(Real[i], Imag[i]))
    ax.plot_surface(X,Y,Z,rstride=1, cstride=1, alpha=0.2)
    ax.set_xlabel('Real axis')
    ax.set_ylabel('Imag axis')
    ax.set_zlabel('Z axis')
    plt.tight_layout()
    plt.show()
    return RealCoeffecients, ImagCoeffecients