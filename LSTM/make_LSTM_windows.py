## file for making the windowed data for the convolutional LSTM model
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


def generate_lstm_windows(window_size=50, 
                          features=['knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance'],
                          output="knee_angle_l",
                          subject="11"):
    if os.getlogin() == 'nicholasharris':
        base_path = '/Users/nicholasharris/OneDrive - Georgia Institute of Technology/BioZ Dataset'

    full_path = '../DATASET/'

    # get all the data file names
    # data_files = os.listdir(full_path)
    # data_files.remove('.DS_Store')
    # for data_file in data_files:
    #     subj_num = data_file.split("_")[-1].split(".")[0]

    window_frequency = 40

    data_file = "ankle_base_" + subject + ".npy"

    data = np.load(full_path + data_file)

    # LSTM takes data of size (n_windows x window_size x num_features)
    ## knee angle L (y output)
    y_output = data[output]
    y_output = (y_output - np.mean(y_output)) / np.std(y_output)

    x_input = []
    for feature in features:
        x_input.append(data[feature])
    
    x_input = np.stack(x_input, axis=2)
    x_input = (x_input - np.mean(x_input, axis = (0,1))) / np.std(x_input, axis=(0,1))
    

    x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size=0.1, random_state=42)


    x_train_windows = []
    y_train_windows = []
    for i in range(np.shape(x_train)[0]):

        # fig, axs = plt.subplots(nrows=2, sharex=True)
        # axs[0].plot(x_train[i,:,0])
        # axs[1].plot(y_train[i])
        # plt.show(block=True)

        # fft_x = np.fft.fft(x_train[i,:,0])
        # fft_y = np.fft.fft(y_train[i])
        # freq = np.fft.fftfreq(np.shape(y_train[i])[-1])
        # fig, axs = plt.subplots(nrows=2, sharex=True)
        # axs[0].plot(freq, np.sqrt(np.square(fft_x.real) + np.square(fft_x.imag)))
        # axs[1].plot(freq, np.sqrt(np.square(fft_y.real) + np.square(fft_y.imag)))
        # plt.show(block=True)
        for j in range(np.shape(x_train)[1] - window_size):
            if j % window_frequency:
                continue
            x_train_windows.append(x_train[i, j:j+window_size])
            y_train_windows.append(y_train[i, j+window_size])
            1

    x_train_windows = np.stack(x_train_windows, axis=0)
    y_train_windows = np.stack(y_train_windows, axis=0)
    y_train_windows = np.expand_dims(y_train_windows, axis=-1)

    
    x_test_windows = []
    y_test_windows = []
    for i in range(np.shape(x_test)[0]):
        for j in range(np.shape(x_test)[1] - window_size):
            if j % window_frequency:
                continue
            x_test_windows.append(x_test[i, j:j+window_size])
            y_test_windows.append(y_test[i, j+window_size])

    x_test_windows = np.stack(x_test_windows, axis=0)
    y_test_windows = np.stack(y_test_windows, axis=0)
    y_test_windows = np.expand_dims(y_test_windows, axis=-1)

    return x_train_windows, x_test_windows, y_train_windows, y_test_windows


def generate_lstm_windows_multiple_subs(window_size=50, 
                          features=['knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance'],
                          output="knee_angle_l",
                          subjects = ['8', '11']):
    if os.getlogin() == 'nicholasharris':
        base_path = '/Users/nicholasharris/OneDrive - Georgia Institute of Technology/BioZ Dataset'

    full_path = '../DATASET/'

    # get all the data file names
    # data_files = os.listdir(full_path)
    # data_files.remove('.DS_Store')
    # for data_file in data_files:
    #     subj_num = data_file.split("_")[-1].split(".")[0]

    window_frequency = 40

    x_train_windows = []
    y_train_windows = []
    x_test_windows = []
    y_test_windows = []
    for subject in subjects:
        data_file = "ankle_base_" + subject + ".npy"

        data = np.load(full_path + data_file)

        # LSTM takes data of size (n_windows x window_size x num_features)
        ## knee angle L (y output)
        y_output = data[output]
        y_output = (y_output - np.mean(y_output)) / np.std(y_output)

        x_input = []
        for feature in features:
            x_input.append(data[feature])
        
        x_input = np.stack(x_input, axis=2)
        x_input = (x_input - np.mean(x_input, axis = (0,1))) / np.std(x_input, axis=(0,1))
        

        x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size=0.1, random_state=42)

        for i in range(np.shape(x_train)[0]):

            # fig, axs = plt.subplots(nrows=2, sharex=True)
            # axs[0].plot(x_train[i,:,0])
            # axs[1].plot(y_train[i])
            # plt.show(block=True)

            # fft_x = np.fft.fft(x_train[i,:,0])
            # fft_y = np.fft.fft(y_train[i])
            # freq = np.fft.fftfreq(np.shape(y_train[i])[-1])
            # fig, axs = plt.subplots(nrows=2, sharex=True)
            # axs[0].plot(freq, np.sqrt(np.square(fft_x.real) + np.square(fft_x.imag)))
            # axs[1].plot(freq, np.sqrt(np.square(fft_y.real) + np.square(fft_y.imag)))
            # plt.show(block=True)
            for j in range(np.shape(x_train)[1] - window_size):
                if j % window_frequency:
                    continue
                x_train_windows.append(x_train[i, j:j+window_size])
                y_train_windows.append(y_train[i, j+window_size])
                1

        for i in range(np.shape(x_test)[0]):
            for j in range(np.shape(x_test)[1] - window_size):
                if j % window_frequency:
                    continue
                x_test_windows.append(x_test[i, j:j+window_size])
                y_test_windows.append(y_test[i, j+window_size])

    x_train_windows = np.stack(x_train_windows, axis=0)
    y_train_windows = np.stack(y_train_windows, axis=0)
    y_train_windows = np.expand_dims(y_train_windows, axis=-1)

    x_test_windows = np.stack(x_test_windows, axis=0)
    y_test_windows = np.stack(y_test_windows, axis=0)
    y_test_windows = np.expand_dims(y_test_windows, axis=-1)

    return x_train_windows, x_test_windows, y_train_windows, y_test_windows

