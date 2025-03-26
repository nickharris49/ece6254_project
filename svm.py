import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.model_selection import train_test_split


def main():
    # pathing
    datadir = "DATASET/"
    datafiles = os.listdir(datadir)
    
    X_path = datadir + "feature_vector_full.npy" # change this to change the input feature vector
    y_path = datadir + "y.npy"

    # load in data
    X = np.load(X_path)
    y = np.load(y_path)

    # split into train, val, and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42)

    # Implementation of SVR
    
    dummy = 1

if __name__ == '__main__':
    main()