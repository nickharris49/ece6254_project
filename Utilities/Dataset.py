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

NEW_DATASET_DELIMITER = b'\nNew Dataset\n'
NEW_DATA_DUMP_DELIMITER = b'\nDataDump\n'


class Dataset:
    def __init__(self, DataSetData):
        self.RawDataset = DatasetData(DataSetData)
        self.JsonDict = self.RawDataset.JsonDict
        self.BioZ = Bioimpedance(self.RawDataset.BioZ, self.JsonDict['BioZ'] ,self.JsonDict['DeviceID'], int(self.JsonDict['FirmwareVersion'][0]), self.RawDataset.CurrentVoltage)
        self.IMU = Acceleromter(self.RawDataset.IMU, self.JsonDict)
        self.Temperature = Temperature(self.RawDataset.Temperature, self.JsonDict)

def parse_file_for_data_sets(PathToFile):
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
        try:
            if i == (len(DataSetsIndicies) - 1):
                DataSets.append(Dataset(data[DataSetsIndicies[i]:]))
            else:
                DataSets.append(
                    Dataset(data[DataSetsIndicies[i]: DataSetsIndicies[i + 1] - len(NEW_DATASET_DELIMITER)]))
        except Exception as e:
            print(e)
    return DataSets





