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
import cv2
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from scipy import ndimage

NumElectrodes = 8
Rows = 100
Cols = 100
Img = np.zeros([Cols,Rows])

Electrodes = np.zeros([NumElectrodes,2])
DegreeBetweenElectrodes = 2 * np.pi / NumElectrodes
RotatationMatrix =
Origin = [Rows / 2, Cols / 2]
for i in range(NumElectrodes - 1):
    RotMatrix = np.array([[np.cos((i + 1) * DegreeBetweenElectrodes), -1 * np.sin((i + 1) * DegreeBetweenElectrodes)],
                          [np.sin((i + 1) * DegreeBetweenElectrodes), np.cos((i + 1) * DegreeBetweenElectrodes)]])
    Electrodes[i + 1] = (np.dot(RotMatrix, self.Electrodes[0] - Origin) + Origin).astype(np.int)