import matplotlib
import pandas as pd
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import math

import os

class Temperature:
    def __init__(self,TemperatureData, JsonDict):
        self.Data = TemperatureData
    def plot(self):
        fig, ax = plt.subplots(2, figsize=(9, 7))
        ax[0].scatter(self.Data[:, 0] / 3600000, self.Data[:, 1], alpha=0.5, s=30)
        ax[0].plot(self.Data[:, 0] / 3600000, self.Data[:, 1])
        ax[1].scatter(self.Data[:, 0] / 3600000, self.Data[:, 2], alpha=0.5, s=30)
        ax[1].plot(self.Data[:, 0] / 3600000, self.Data[:, 2])
        plt.tight_layout()
        plt.show()
