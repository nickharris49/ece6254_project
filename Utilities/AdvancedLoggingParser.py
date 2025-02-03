import matplotlib
import os
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import json
from Utilities.FileParser import DatasetData
from Utilities.Bioimpedance import Bioimpedance
from Utilities.Accelerometer import Acceleromter
from Utilities.Temperature import Temperature
from Utilities.FileParser import AdvancedLoggingRawDataParser

class AdvancedLoggingDataset:
    def __init__(self, PathToFolder):
        self.RawDataset = AdvancedLoggingRawDataParser(PathToFolder)
        self.JsonDict = self.RawDataset.JsonDict
        HWVersion = 2
        if 'HWVersion' in self.JsonDict:
            HWVersion = self.JsonDict['HWVersion']
        print(HWVersion)
        DirList = os.listdir(PathToFolder)
        if 'BioZ0' in self.JsonDict and 'data0.txt' in DirList:
            NumFrequencies = int(np.floor(
                (self.JsonDict['BioZ0']['endFreq'] - self.JsonDict['BioZ0']['startFreq']) / self.JsonDict['BioZ0'][
                    'stepFreq']) + 1)
            TempJson = json.loads('{"NumFrequencies":%i, "StartFrequency":%i, "Resolution":%i}'%(NumFrequencies, self.JsonDict['BioZ0']['startFreq'], self.JsonDict['BioZ0']['stepFreq']))
            self.BioZ0 = Bioimpedance(self.RawDataset.BioZ0, TempJson ,TempJson, HWVersion,CurrentVoltage=self.RawDataset.CurrentVoltage0)
        if 'BioZ1' in self.JsonDict and 'data1.txt' in DirList:
            NumFrequencies = int(np.floor(
                (self.JsonDict['BioZ1']['endFreq'] - self.JsonDict['BioZ1']['startFreq']) / self.JsonDict['BioZ1'][
                    'stepFreq']) + 1)
            TempJson = json.loads('{"NumFrequencies":%i, "StartFrequency":%i, "Resolution":%i}'%(NumFrequencies, self.JsonDict['BioZ1']['startFreq'], self.JsonDict['BioZ1']['stepFreq']))
            self.BioZ1 = Bioimpedance(self.RawDataset.BioZ1, TempJson ,TempJson, HWVersion, CurrentVoltage=self.RawDataset.CurrentVoltage1)
        if 'imu.txt' in DirList:
            self.IMU = Acceleromter(self.RawDataset.IMU, self.JsonDict)
        # self.Temperature = Temperature(self.RawDataset.Temperature, self.JsonDict)


# Temp = AdvancedLoggingDataset('Data/NewFWTest/Walking-9-1-22/Samer_1_31_23/Right')

# Temp = AdvancedLoggingDataset('Data/MotionCaptureLab_Amro/ChritophData_1_30_23_Left')

#
# Z = (Temp.RawDataset.CurrentVoltage1[0,:,2]/Temp.RawDataset.CurrentVoltage1[0,:,1])
# BioZ5kReal = Z.real*182.54697395 + Z.imag*1.50599035 -0.19904527
# BioZ5kImag =Z.real*4.60398197e+00 +Z.imag*1.90584418e+02 -3.09455842e-02
#
# Z = (Temp.RawDataset.CurrentVoltage1[-1,:,2]/Temp.RawDataset.CurrentVoltage1[-1,:,1])
# BioZ100kReal = Z.real * 1.76266244e+02 +  Z.imag* -8.58906171e+01 - 1.52396868e-01
# BioZ100kImag = Z.real* 95.13348879 +  Z.imag* 164.66473575 - 0.90657555
#
# Temp.BioZ1.Data[0,:,1] = BioZ5kReal
# Temp.BioZ1.Data[0,:,2] = BioZ5kImag
# Temp.BioZ1.Data[-1,:,1] = BioZ100kReal
# Temp.BioZ1.Data[-1,:,2] = BioZ100kImag
#
# plt.plot(Temp.BioZ1.Data[0,:,1]/Temp.BioZ1.Data[-1,:,1])
#
# File = open('Data/NewFWTest/BreastData/data1_old.txt', 'rb')
# file2 = open('Data/NewFWTest/BreastData/data1.txt', 'wb')
#
# data_bytes = File.read()
# # file2.write(data_bytes[-1000*4096:])
# file2.write(data_bytes[-110000*32:])
# file2.close()
# File.close()
#

#EachBlock = 128 Sweeps *32 bytes
#-110000*32