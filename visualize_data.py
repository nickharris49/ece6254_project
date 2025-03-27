import numpy as np
import matplotlib.pyplot as plt

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
    

if __name__ == '__main__':
    main()