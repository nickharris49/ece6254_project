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
from Utilities.Calibration import  Calibration
from Utilities.Calibration import eval2R1C
from scipy.optimize import curve_fit

class Bioimpedance:
    def __init__(self,BioZData, JsonDict, DeviceID, FirmwareVersion = 1, CurrentVoltage = np.array([])):
        self.CalibrationInstance = Calibration(DeviceID, FirmwareVersion=FirmwareVersion)
        self.Frequencies = np.zeros([int(JsonDict['NumFrequencies'])])
        for i in range(int(JsonDict['NumFrequencies'])):
            self.Frequencies[i] = int(JsonDict['StartFrequency']) + i*int(JsonDict['Resolution'])
        self.Data = BioZData
        self.Data[:,:,1], self.Data[:,:,2] = self.CalibrationInstance.predict(BioZData[:,:, 1], BioZData[:,:,2], self.Frequencies)
        # Calibrate Electrode Z for only 5k for now
        self.ElectrodeImpedance = None
        try:
            if CurrentVoltage.shape[1]>0 and (self.Frequencies[0] == 5000):
                Xreal = np.array([-7.17218459e+07, 1.80172259e+08, 4.75042865e+02])
                Ximag = np.array([-1.76842937e+08, -3.90023497e+07,  2.16039374e+02])
                A = np.ones([CurrentVoltage.shape[1], 3])
                A[:, 0] = (1 / CurrentVoltage[0, :, 1]).real
                A[:, 1] = (1 / CurrentVoltage[0, :, 1]).imag
                Zprotection = (1 / (1j * 2 * np.pi * 5000 * 0.01e-6)) + (1 / (1j * 2 * np.pi * 5000 * 0.47e-6)) + 2000
                self.ElectrodeImpedance = (np.dot(A, Xreal) - Zprotection.real) -1j*(np.dot(A, Ximag) - Zprotection.imag)
        except Exception as e:
            print(e)

    def plot_data_point(self, index = [0]):
        for i in index:
            fig, ax = plt.subplots(figsize=(9 ,7))
            ax.scatter(self.Data[:,i,1], -self.Data[:, i, 2])
            ax.set_xlabel('R (Ohms)')
            ax.set_ylabel('-X (Ohms)')
            ax.set_title('R vs X at Index %i ' %i)
            plt.tight_layout()
            plt.show()

    def plot(self, FrequenciesToPlot=[5000, 100000], ylimR = [0,300], ylimX=[-20,20]):
        for Frequency in FrequenciesToPlot:
            Index = self._find_nearest(Frequency)
            fig, ax = plt.subplots(2, figsize=(9 ,7))
            ax[0].plot(self.Data[Index, :, 0]/3600000, self.Data[Index, :, 1])
            ax[0].scatter(self.Data[Index, :, 0]/3600000, self.Data[Index, :, 1], alpha=0.5, s=30)
            ax[0].set_ylabel('R (Ohms)')
            ax[0].set_ylim(ylimR)
            ax[0].set_title('Resistance vs Time at %iHz ' %Frequency)
            ax[1].plot(self.Data[Index, :, 0]/3600000, -1*self.Data[Index, :, 2])
            ax[1].scatter(self.Data[Index, :, 0]/3600000, -1*self.Data[Index, :, 2],alpha=0.5, s=30)
            ax[1].set_xlabel('Time(hr)')
            ax[1].set_ylabel('-X (Ohms)')
            ax[1].set_ylim(ylimX)
            ax[1].set_title('Reactance vs Time at %iHz ' %Frequency)
            plt.tight_layout()
            plt.show()

    def cole_cole_analysis(self, index=[0]):
        ColeColeLoads = np.zeros([len(index), 4])
        for n, i in enumerate(index):
            popt_real, pcov = curve_fit(self.eval_cole_cole_real, self.Frequencies, self.Data[:, i, 1],
                                        bounds=(0, [1000.0, 10000.0, 1.0, 10000000]))
            print(popt_real)
            # popt_imag, pcov = curve_fit(self.eval_cole_cole_Imag, self.Frequencies, self.Data[:, i, 1], bounds=([popt_real[0]-10, popt_real[1]-100, popt_real[2]-0.1, popt_real[3]-10000], [popt_real[0]+10, popt_real[1]+100, popt_real[2]+0.1, popt_real[3]+10000]))
            ColeColeLoads[n] = popt_real
            fig ,ax = plt.subplots(figsize = (9 ,7))
            ax.scatter(self.Data[:,i,1], -self.Data[:,i,2])
            ax.scatter(self.eval_cole_cole(self.Frequencies, ColeColeLoads[n,0], ColeColeLoads[n,1], ColeColeLoads[n,2], ColeColeLoads[n,3])[0],
                       -self.eval_cole_cole(self.Frequencies, ColeColeLoads[n,0], ColeColeLoads[n,1], ColeColeLoads[n,2], ColeColeLoads[n,3])[1])
            Re, Ri, C = self.Cole_Cole_To_Fricke_Morse(ColeColeLoads[n,0], ColeColeLoads[n,1], ColeColeLoads[n,3])
            print(Re, Ri, C)
            ax.scatter(self._eval_2r1c_real(self.Frequencies, Re, Ri, C*10**9), -self._eval_2r1c_imag(self.Frequencies, Re, Ri, C*10**9))
            plt.show()
        return ColeColeLoads
    def cole_cole_3R1C_analysis(self, index=[0]):
        ColeColeLoads = np.zeros([len(index), 4])
        for n, i in enumerate(index):
            popt_real, pcov = curve_fit(self.eval3R1C_real, self.Frequencies, self.Data[:, i, 1],bounds=(0, [1000.0, 10000.0, 50e-9, 10000000]))
            print(popt_real)
            # popt_imag, pcov = curve_fit(self.eval3R1C_real, self.Frequencies, self.Data[:, i, 1], bounds=([popt_real[0]-10, popt_real[1]-100, popt_real[2]-0.1, popt_real[3]-10000], [popt_real[0]+10, popt_real[1]+100, popt_real[2]+0.1, popt_real[3]+10000]))
            ColeColeLoads[n] = popt_real
            fig ,ax = plt.subplots(figsize = (9 ,7))
            ax.scatter(self.Data[:,i,1], -self.Data[:,i,2])
            Z = self.eval3R1C(self.Frequencies, ColeColeLoads[n,0], ColeColeLoads[n,1], ColeColeLoads[n,2], ColeColeLoads[n,3])
            ax.scatter(Z.real, -Z.imag)
            print(ColeColeLoads[n])
            plt.show()
        return ColeColeLoads

    def fricke_morse_analysis(self, index=[0]):
        ColeColeLoads = np.zeros([len(index), 3])
        for n, i in enumerate(index):
            popt_real, pcov = curve_fit(self._eval_2r1c_real, self.Frequencies, self.Data[:, i, 1], bounds=(0, [1000, 2000, 50]))
            print(popt_real)
            popt_imag, pcov = curve_fit(self._eval_2r1c_imag, self.Frequencies, self.Data[:, i, 2], bounds=(
                [popt_real[0] - 10, popt_real[1]-100, popt_real[2]-20], [popt_real[0] + 10, popt_real[1] + 100, popt_real[2] + 20]))
            ColeColeLoads[n] = popt_imag
            fig ,ax = plt.subplots(figsize = (9 ,7))
            ax.scatter(self.Data[:,i,1], -self.Data[:,i,2])
            ax.scatter(eval2R1C(ColeColeLoads[n], self.Frequencies)[0],
                       -eval2R1C(ColeColeLoads[n], self.Frequencies)[1])
            plt.show()
        return ColeColeLoads
    def CNLLN(self, index = [0]):
        ColeColeLoads = np.zeros([len(index), 3])
        for n, i in enumerate(index):
            popt, pcov = curve_fit(self._eval_2r1c_complex, self.Frequencies, self.Data[:, i, 1] +1j*self.Data[:,i,2], bounds=(0, [200, 10000, 100]))
            ColeColeLoads[n] = popt
            fig ,ax = plt.subplots(figsize = (9 ,7))
            ax.scatter(self.Data[:,i,1], -self.Data[:,i,2])
            ax.scatter(eval2R1C(ColeColeLoads[n], self.Frequencies)[0],
                       -eval2R1C(ColeColeLoads[n], self.Frequencies)[1])
            plt.show()
        return ColeColeLoads

    def _find_nearest(self, frequency):
        return np.abs(frequency -self.Frequencies).argmin()

    def _eval_2r1c_real(self,freq, Re, Ri, C):
        return eval2R1C([Re, Ri,C],freq)[0]

    def _eval_2r1c_imag(self, freq, Re, Ri ,C):
        return eval2R1C([Re, Ri,C],freq)[1]
    def _eval_2r1c_complex(self, freq, Re, Ri ,C):
        Z = eval2R1C([Re, Ri, C], freq)
        return (Z[0] + 1j*Z[1])

    def eval_cole_cole_real(self, f, R0, Rinf, a, wc):
        w = 2 * math.pi * f
        Num = (R0 - Rinf) * (1 + ((w / wc) ** (1 - a)) * np.sin(a * np.pi / 2))
        Denom = 1 + 2 * ((w / wc) ** (1 - a)) * np.sin(a * np.pi / 2) + ((w / wc) ** (2 * (1 - a)))
        Z = Rinf + Num / Denom
        return Z

    def eval_cole_cole_Imag(self, f, R0, Rinf, a, wc):
        w = 2 * math.pi * f
        Num = (R0 - Rinf) * (((w / wc) ** (1 - a)) * np.cos(a * np.pi / 2))
        Denom = 1 + 2 * ((w / wc) ** (1 - a)) * np.sin(a * np.pi / 2) + ((w / wc) ** (2 * (1 - a)))
        Z = 1 * Num / Denom
        return Z

    def eval_cole_cole(self, f, R0, Rinf, a, wc):
        w = 2 * math.pi * f
        Num = (R0 - Rinf)
        Denom = 1 + (1j * w / wc) ** (1 - a)
        Z = Rinf + Num / Denom
        return (Z.real, Z.imag)

    def Cole_Cole_To_Fricke_Morse(self, R0, Rinf, wc):
        Re = R0
        Ri = (R0 * Rinf) / (R0 - Rinf)
        Cm = 1 / (wc * (Re + Ri))
        return (Re, Ri, Cm)

    def Fricke_Morse_To_Cole_Cole(self, Re, Ri, C):
        R0 = Re
        Rinf = (R0 * Ri) / (Ri + R0)
        wc = 1 / (C * (Ri + Re))
        return R0, Rinf, wc
    def evalCellMembraneImpedance(self, Frequency, Cm, Rm):
        w = 2 * math.pi * Frequency
        Z = 1/((1/Rm) + 1j*w*Cm)
        return Z
    def evalCellImpedance(self, Frequency, Ri, Cm, Rm):
        Zcell = Ri+self.evalCellMembraneImpedance(Frequency, Cm, Rm)
        return Zcell
    def eval3R1C(self, Frequency, Re, Ri, Cm, Rm):
        Zcell = self.evalCellImpedance(Frequency, Ri, Cm, Rm)
        Z = (Re*Zcell)/(Re+Zcell)
        return Z
    def eval3R1C_real(self, Frequency, Re, Ri, Cm, Rm):
        Z = self.eval3R1C(Frequency, Re, Ri, Cm, Rm)
        return Z.real
    def eval3R1C_imag(self, Frequency, Re, Ri, Cm, Rm):
        Z = self.eval3R1C(Frequency, Re, Ri, Cm, Rm)
        return Z.imag