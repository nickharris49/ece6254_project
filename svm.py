import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.model_selection import train_test_split


def main():
    datadir = "DATASET/"
    datafiles = os.listdir(datadir)
    X_path = datadir + "feature_vector_full.npy"
    y_path = datadir + "y.npy"
    X = np.load(X_path).T
    y = np.load(y_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dummy = 1

if __name__ == '__main__':
    main()