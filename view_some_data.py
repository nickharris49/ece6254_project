import numpy as np
import matplotlib.pyplot as plt

def main():

    x = np.load('DATASET/feature_vector_full.npy')
    y = np.load('DATASET/y.npy')
    resist_5k = x[0]
    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(y[:1000])
    axs[1].plot(resist_5k[:1000])
    plt.show(block=True)
    

if __name__ == '__main__':
    main()