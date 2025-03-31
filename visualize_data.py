import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D


def plot_predictions(y_test, y_pred):    
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='green', alpha=0.5, label='y_test')
    plt.scatter(range(len(y_test)), y_pred, color='red', alpha=0.5, label='y_pred')
    plt.legend(loc="upper right")  
    plt.title("Test: Actual vs. Predicted")
    plt.show()


def main():

    x = np.load('DATASET/feature_vector_full_normalized.npy')
    y = np.load('DATASET/y_normalized.npy')
    resist_5k = x[:,0]
    react_5k = x[:,1]
    resist_100k = x[:,2]
    react_100k = x[:,3]
    # fig, axs = plt.subplots(nrows=4)
    # for i, ax in enumerate(axs):
    #     ax.scatter(x[:,i], y, alpha=0.01)
    # plt.show(block=True)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.scatter(resist_100k[200100:200550], react_100k[200100:200550], y[200100:200550,0])
    plt.show()
    
    X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)
    y_pred = np.load("rf_pred.npy")
    # y_pred = np.load("xgboost_pred.npy")
    plot_predictions(y_test, y_pred)    

if __name__ == '__main__':
    main()