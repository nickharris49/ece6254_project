import matplotlib
import pandas as pd
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math

import os

class Acceleromter:
    def __init__(self,IMUData, JsonDict):
        self.Data = IMUData
    def plot_accelerometer(self):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(9, 7))
        ax[0,0].scatter(self.Data[:, 0] / 3600000, self.Data[:, 1], alpha=0.5, s=30)
        ax[0,0].plot(self.Data[:, 0] / 3600000, self.Data[:, 1])
        ax[1,0].scatter(self.Data[:, 0] / 3600000, self.Data[:, 2], alpha=0.5, s=30)
        ax[1,0].plot(self.Data[:, 0] / 3600000, self.Data[:, 2])
        ax[2,0].scatter(self.Data[:, 0] / 3600000, self.Data[:, 3], alpha=0.5, s=30)
        ax[2,0].plot(self.Data[:, 0] / 3600000, self.Data[:, 3])
        ax[0,1].scatter(self.Data[:, 0] / 3600000, self.Data[:, 7], alpha=0.5, s=30)
        ax[0,1].plot(self.Data[:, 0] / 3600000, self.Data[:, 7])
        ax[1,1].scatter(self.Data[:, 0] / 3600000, self.Data[:, 8], alpha=0.5, s=30)
        ax[1,1].plot(self.Data[:, 0] / 3600000, self.Data[:, 8])
        ax[2,1].scatter(self.Data[:, 0] / 3600000, self.Data[:, 9], alpha=0.5, s=30)
        ax[2,1].plot(self.Data[:, 0] / 3600000, self.Data[:, 9])
        plt.tight_layout()
        plt.show()
    def plot_gyro(self):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(9, 7))
        ax[0,0].scatter(self.Data[:, 0] / 3600000, self.Data[:, 4], alpha=0.5, s=30)
        ax[0,0].plot(self.Data[:, 0] / 3600000, self.Data[:, 4])
        ax[1,0].scatter(self.Data[:, 0] / 3600000, self.Data[:, 5], alpha=0.5, s=30)
        ax[1,0].plot(self.Data[:, 0] / 3600000, self.Data[:, 5])
        ax[2,0].scatter(self.Data[:, 0] / 3600000, self.Data[:, 6], alpha=0.5, s=30)
        ax[2,0].plot(self.Data[:, 0] / 3600000, self.Data[:, 6])
        ax[0,1].scatter(self.Data[:, 0] / 3600000, self.Data[:, 10], alpha=0.5, s=30)
        ax[0,1].plot(self.Data[:, 0] / 3600000, self.Data[:, 10])
        ax[1,1].scatter(self.Data[:, 0] / 3600000, self.Data[:, 11], alpha=0.5, s=30)
        ax[1,1].plot(self.Data[:, 0] / 3600000, self.Data[:, 11])
        ax[2,1].scatter(self.Data[:, 0] / 3600000, self.Data[:, 12], alpha=0.5, s=30)
        ax[2,1].plot(self.Data[:, 0] / 3600000, self.Data[:, 12])
        plt.tight_layout()
        plt.show()

