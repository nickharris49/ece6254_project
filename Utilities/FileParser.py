import matplotlib
import os
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
#import paths
import json

NEW_DATASET_DELIMITER = b'\nNew Dataset\n'
NEW_DATA_DUMP_DELIMITER = b'\nDataDump\n'

class RawData:
    def get_raw_data_from_json(self, RawData, JsonDict):
        Delimiter = b'\n' + JsonDict['Delimiter'].encode()
        PacketSize = int(JsonDict['PacketSize'])
        NumPerPackets = int(JsonDict["NumPerPacket"])
        SizeOfInstance = int(PacketSize / NumPerPackets)
        DataIndicies = []
        Index = RawData.find(Delimiter)
        ExtractedData = []
        LastIndex = 0
        while (Index != -1):
            DataIndicies.append(Index + LastIndex + len(Delimiter))
            LastIndex += Index + len(Delimiter)
            Index = RawData[LastIndex:].find(Delimiter)
        for i in (DataIndicies):
            for j in range(NumPerPackets):
                ExtractedData.append(RawData[i + j * SizeOfInstance: i + (j + 1) * SizeOfInstance])
        return ExtractedData
    def __init__(self, RawData, JsonDict):
        self.BioZ = self.get_raw_data_from_json(RawData, JsonDict['BioZ'])
        self.IMU = self.get_raw_data_from_json(RawData, JsonDict["IMU"])
        self.Temperature = self.get_raw_data_from_json(RawData, JsonDict["Temperature"])

class DatasetData:
    def format_bioz_data(self, Bytes, NumFrequencies, FirmwareVersion):
        if FirmwareVersion == 1:
            np_array = np.zeros([NumFrequencies, int(len(Bytes) / NumFrequencies), 3])
            i = 0
            for ByteArray in Bytes:
                Time = ByteArray[:4]
                Real = ByteArray[4:6]
                Imag = ByteArray[6:]
                Time = (int.from_bytes(Time[0:2], byteorder='little', signed=False) << 16 | int.from_bytes(
                    Time[2:], byteorder='little', signed=False)) / 2.048
                Real = int.from_bytes(Real, byteorder='little', signed=True)
                Imag = int.from_bytes(Imag, byteorder='little', signed=True)
                np_array[i % (NumFrequencies), int(i / (NumFrequencies)), 0] = Time
                np_array[i % (NumFrequencies), int(i / (NumFrequencies)), 1] = Real
                np_array[i % (NumFrequencies), int(i / (NumFrequencies)), 2] = Imag
                i = i + 1
        elif FirmwareVersion == 2:
            np_array = np.zeros([NumFrequencies, int(len(Bytes) / (NumFrequencies)), 3])
            i = 0
            for ByteArray in Bytes:
                Time = (int.from_bytes(ByteArray[:4], byteorder='little', signed=False)) / 2.048
                RealCurrent = int.from_bytes(ByteArray[4:8], byteorder='little', signed=True)
                ImagCurrent = int.from_bytes(ByteArray[8:12], byteorder='little', signed=True)
                RealVoltage = int.from_bytes(ByteArray[12:16], byteorder='little', signed=True)
                ImagVoltage = int.from_bytes(ByteArray[16:], byteorder='little', signed=True)
                Current = -1*(RealCurrent - 1j*ImagCurrent)
                Voltage = RealVoltage - 1j*ImagVoltage
                try:
                    Z = Voltage/Current
                except Exception as e:
                    Z = 0 + 0j
                np_array[i % (NumFrequencies), int(i / (NumFrequencies)), 0] = Time
                np_array[i % (NumFrequencies), int(i / (NumFrequencies)), 1] = Z.real
                np_array[i % (NumFrequencies), int(i / (NumFrequencies)), 2] = Z.imag
                i = i+1
        return np_array
    def format_imu_data(self, Bytes):
        if Bytes:
            np_array = np.zeros([len(Bytes), int(len(Bytes[0])/2) - 1])
            Index = 0
            for ByteArray in Bytes:
                Time = ByteArray[0:4]
                Time = (int.from_bytes(Time[0:2], byteorder='little', signed=False) << 16 | int.from_bytes(
                    Time[2:], byteorder='little', signed=False)) / 2.048
                np_array[Index][0] = Time
                for ByteIndex in range(int(len(ByteArray[4:])/2)):
                    np_array[Index][ByteIndex+1] = int.from_bytes(ByteArray[4+2*ByteIndex: 4+2*(ByteIndex+1)], byteorder='little', signed=True)/(2**14)
                Index = Index+1
            return np_array
        else:
            return None
    def format_temperature_data(self, Bytes):
        np_array = np.zeros([len(Bytes), 3])
        Index = 0
        for ByteArray in Bytes:
            Time = ByteArray[0:4]
            Time = (int.from_bytes(Time[0:2], byteorder='little', signed=False) << 16 | int.from_bytes(
                Time[2:], byteorder='little', signed=False)) / 2.048
            Temperature_1 = int.from_bytes(ByteArray[4:6],  byteorder='little', signed=True)*0.0625
            Temperature_2 = int.from_bytes(ByteArray[6:8],  byteorder='little', signed=True)*0.0625
            np_array[Index] = [Time, Temperature_1, Temperature_2]
            Index = Index+1
        return np_array
    def format_current_and_voltage_data(self, Bytes, NumFrequencies):
            np_array = np.zeros([NumFrequencies, int(len(Bytes) / (NumFrequencies)), 3], dtype=complex)
            i = 0
            for ByteArray in Bytes:
                Time = (int.from_bytes(ByteArray[:4], byteorder='little', signed=False)) / 2.048
                RealCurrent = int.from_bytes(ByteArray[4:8], byteorder='little', signed=True)
                ImagCurrent = int.from_bytes(ByteArray[8:12], byteorder='little', signed=True)
                RealVoltage = int.from_bytes(ByteArray[12:16], byteorder='little', signed=True)
                ImagVoltage = int.from_bytes(ByteArray[16:], byteorder='little', signed=True)
                Current = -1*(RealCurrent - 1j*ImagCurrent)
                Voltage = RealVoltage - 1j*ImagVoltage
                np_array[i % (NumFrequencies), int(i / (NumFrequencies)), 0] = Time
                np_array[i % (NumFrequencies), int(i / (NumFrequencies)), 1] = Current
                np_array[i % (NumFrequencies), int(i / (NumFrequencies)), 2] = Voltage
                i = i+1
            return np_array

    def __init__(self, DataSetData):
        Index = DataSetData.find(NEW_DATA_DUMP_DELIMITER)
        self.JsonDict = json.loads(DataSetData[:Index])
        self.RawDataBytes = RawData(DataSetData[Index:], self.JsonDict)
        self.BioZ = self.format_bioz_data(self.RawDataBytes.BioZ, int(self.JsonDict['BioZ']['NumFrequencies']), int(self.JsonDict['FirmwareVersion'][0]))
        self.IMU = self.format_imu_data(self.RawDataBytes.IMU)
        self.Temperature = self.format_temperature_data(self.RawDataBytes.Temperature)
        if int(self.JsonDict['FirmwareVersion'][0]) == 2:
            self.CurrentVoltage = self.format_current_and_voltage_data(self.RawDataBytes.BioZ, int(self.JsonDict['BioZ']['NumFrequencies']))
        else:
            self.CurrentVoltage = None

def parse_file(PathToFile):
    file = open(PathToFile, 'rb')
    data = file.read()

    DataSets = []
    DataSetsIndicies = []
    Index = data.find(NEW_DATASET_DELIMITER)
    LastIndex = 0
    while (Index != -1):
        DataSetsIndicies.append(Index + LastIndex + len(NEW_DATASET_DELIMITER))
        LastIndex += Index + len(NEW_DATASET_DELIMITER)
        Index = data[LastIndex:].find(NEW_DATASET_DELIMITER)
    for i in range(len(DataSetsIndicies)):
        if i == (len(DataSetsIndicies) - 1):
            DataSets.append(DatasetData(data[DataSetsIndicies[i]:]))
        else:
            DataSets.append(
                DatasetData(data[DataSetsIndicies[i]: DataSetsIndicies[i + 1] - len(NEW_DATASET_DELIMITER)]))
    return DataSets

def split_datasets_into_files(OldPath, NewPath):
    file = open(OldPath, 'rb')
    data = file.read()

    DataSets = []
    DataSetsIndicies = []
    Index = data.find(NEW_DATASET_DELIMITER)
    LastIndex = 0
    while (Index != -1):
        DataSetsIndicies.append(Index + LastIndex + len(NEW_DATASET_DELIMITER))
        LastIndex += Index + len(NEW_DATASET_DELIMITER)
        Index = data[LastIndex:].find(NEW_DATASET_DELIMITER)
    for i in range(len(DataSetsIndicies)):
        file = open("%s/%i.txt"%(NewPath,i), 'wb')
        file.write(NEW_DATASET_DELIMITER)
        if i == (len(DataSetsIndicies) - 1):
            file.write(data[DataSetsIndicies[i]:])
        else:
            file.write(data[DataSetsIndicies[i]: DataSetsIndicies[i + 1] - len(NEW_DATASET_DELIMITER)])
    return DataSets


# Enables sign extension - Assumes AD5940 data - 18 Bits
def sign_ext_bioz(data):
    if data & 0x0002_0000:
        data = data - (1 << 18)
    return data

class AdvancedLoggingRawDataParser:
    def format_bioZ_data(self, Bytes, NumFrequencies, PacketSize):
        NumInstance = int(len(Bytes)/(NumFrequencies*PacketSize))
        CurrentVoltage = np.zeros([NumFrequencies, NumInstance, 3], dtype=complex)
        BioZ = np.zeros([NumFrequencies, NumInstance, 3])
        for i in range(NumInstance):
            for j in range(NumFrequencies):
                Index = (i * NumFrequencies) + j
                PacketBytes = Bytes[Index * PacketSize: (Index + 1) * PacketSize]
                # Demo - Info Extraction - With mux expansion mode - 256 Mappings utilizing time[31:30]
                sample = PacketBytes[15::-1]  # Little endian swap
                data = [int(sample[0:4].hex(), 16), int(sample[4:8].hex(), 16), int(sample[8:12].hex(), 16),
                        int(sample[12:16].hex(), 16)]

                time = (data[3] & 0x3FFF_FFFF) / 2.048  # ticks (32768 -> 2 per ms)
                freq = data[2] >> 14  # Int freq Hz
                mux = ((data[2] >> 8) & 0x3F) + ((data[3] >> 24) & 0xC0)  # Mux mapping (8 -bits expansion mode)
                I_real = ((data[2] & 0x0000_00FF) << 10) + (
                        data[1] >> 22)  # Typical raw bioz values (18 bit measuremnts expanded to int)
                I_img = (data[1] >> 4) & 0x0003_FFFF
                V_real = ((data[1] & 0x0000_000F) << 14) + (data[0] >> 18)
                V_img = data[0] & 0x0003_FFFF
                V_img = sign_ext_bioz(V_img)
                V_real = sign_ext_bioz(V_real)
                I_img = sign_ext_bioz(I_img)
                I_real = sign_ext_bioz(I_real)

                CurrentVoltage[j, i, 0] = time
                CurrentVoltage[j, i, 1] = -1 * (I_real - 1j * I_img)
                CurrentVoltage[j, i, 2] = V_real - 1j * V_img
                BioZ[j,i,0] = time
                if (CurrentVoltage[j,i,1]== 0+0j):
                    CurrentVoltage[j, i, 1] = 1+1j
                BioZ[j,i,1] = (CurrentVoltage[j,i,2]/CurrentVoltage[j,i,1]).real
                BioZ[j,i,2] = (CurrentVoltage[j,i,2]/CurrentVoltage[j,i,1]).imag
        return (CurrentVoltage,BioZ)
    def format_imu_data(self,DataBytes, InstanceSize):
        if DataBytes:
            NumMeasurements = int(len(DataBytes)/ InstanceSize)
            np_array = np.zeros([NumMeasurements, 13])
            for i in range(NumMeasurements):
                ByteArray = DataBytes[i*InstanceSize:(i+1)*InstanceSize]
                Time = ByteArray[0:4]
                Time = (int.from_bytes(Time[0:2], byteorder='little', signed=False) << 16 | int.from_bytes(
                    Time[2:], byteorder='little', signed=False)) / 2.048
                np_array[i][0] = Time
                for ByteIndex in range(int(len(ByteArray[4:])/2)):
                    np_array[i][ByteIndex+1] = int.from_bytes(ByteArray[4+2*ByteIndex: 4+2*(ByteIndex+1)], byteorder='little', signed=True)/(2**14)
            return np_array
        else:
            return None

    def __init__(self, PathToFolder):
        DirList = os.listdir(PathToFolder)
        File = open(PathToFolder + '/settings.json', 'rb')
        self.JsonDict = json.loads(File.read())
        BIOZ_PACKET_SIZE = 16
        IMU_PACKET_SIZE = 4+ 2*(6*2)
        if 'BioZ0' in self.JsonDict and 'data0.txt' in DirList:
            NumFrequencies = int(np.floor(
                (self.JsonDict['BioZ0']['endFreq'] - self.JsonDict['BioZ0']['startFreq']) / self.JsonDict['BioZ0']['stepFreq']) + 1)
            File = open(PathToFolder + '/data0.txt', 'rb')
            DataBytes = File.read()
            self.CurrentVoltage0 ,self.BioZ0 = (self.format_bioZ_data(DataBytes, NumFrequencies, BIOZ_PACKET_SIZE))
        if 'BioZ1' in self.JsonDict and 'data1.txt' in DirList:
            NumFrequencies = int(np.floor(
                (self.JsonDict['BioZ1']['endFreq'] - self.JsonDict['BioZ1']['startFreq']) / self.JsonDict['BioZ1'][
                    'stepFreq']) + 1)
            File = open(PathToFolder + '/data1.txt', 'rb')
            DataBytes = File.read()
            self.CurrentVoltage1 ,self.BioZ1 = (self.format_bioZ_data(DataBytes, NumFrequencies, BIOZ_PACKET_SIZE))
        if 'imu.txt' in DirList:
            File = open(PathToFolder + '/imu.txt', 'rb')
            DataBytes = File.read()
            self.IMU = self.format_imu_data(DataBytes,IMU_PACKET_SIZE)

