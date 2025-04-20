import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from make_LSTM_windows import *

from DeepConvLSTMModel import HARModel, init_weights


def main():
    WINDOW_SIZE = 20  # 1 sec window with 50 Hz - CHANGE THIS
    num_classes = 1 # one because this is regression
    test_results = []
    batch_size=32

    if torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')   # CUDA MEANS YOU ARE USING GPU
    print("device: ", device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')   # PREPARATION IS THE ONLY THING YOU SHOULD CHANGE
    # todo: IMPLEMENT THE FOLLOWING DATASET PREPARATION SCRIPT
    # todo: make sure that train_x and test_x will have shapes (N, window_len, num_features)
    # N: number of samples
    # window_len: number of time steps in your one sample (e.g., if you are using 1 sec window and your
    # device is 50Hz, this number is 50)
    # num_features: dimensionality of your data (i.e., number of modalities) -> e.g., audio, imu_acc, imu_gyro 3 modalities
    # make sure that each of these modalities have the same number of time steps
    # todo: make sure that train_y and test_y will have shapes (N, num_classes)
    # num_classes: if it is a classification problem, it is the number of classes. For each sample, use one hot encoding
    # i.e., if that sample i belongs to class 1 and if you have 4 classes, test_y[i] = [1 0 0 0]
    # if it is a regression problem, it is the number of things that you want to predict.
    
  

    # features = ['knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance']
    # output_feature = "knee_angle_l"
    subject = "11"    
    
    # features = ['ankle_bioz_5k_resistance', 'ankle_bioz_5k_reactance', 'ankle_bioz_100k_resistance', 'ankle_bioz_100k_reactance']
    # output_feature = "ankle_angle_l"

    features = ['ankle_bioz_100k_resistance', 'ankle_bioz_100k_reactance']
    output_feature = 'ankle_bioz_5k_resistance'
    #subjects = ['3', '4', '5', '6', '7', '8', '11']
    subjects_ankle_base = [3, 4, 5, 6, 7, 8, 11]

    num_modalities = len(features)  # number of signals you are fusing OR number of features

    print("feature vector")
    print(features)
    print("output feature")
    print(output_feature)
    print("subjects")
    print(subjects_ankle_base)

    # print("subjects")
    # print(subjects)

    train_x, test_x, train_y, test_y = generate_lstm_windows_multiple_subs(WINDOW_SIZE, features, output_feature, subjects_ankle_base, window_frequency=1)
    #train_x, test_x, train_y, test_y = generate_lstm_windows_multiple_subs(WINDOW_SIZE, features, output_feature, subjects)

    # conversion to torch and transferring data to GPU
    shape = train_x.shape
    train_x = torch.from_numpy(np.reshape(train_x.astype(float), [shape[0], 1, shape[1], shape[2]]))
    train_x = train_x.type(torch.FloatTensor).to(device)
    train_y = torch.from_numpy(train_y)
    train_y = train_y.type(torch.FloatTensor).to(device)
    test_x = torch.from_numpy(np.reshape(test_x.astype(float), [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]))
    test_x = test_x.type(torch.FloatTensor).to(device)
    test_y = torch.from_numpy(test_y.astype(np.float32))
    test_y = test_y.type(torch.FloatTensor).to(device)



    trainset = Data.TensorDataset(train_x, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_x, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    print("Data shapes!")
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    net = HARModel(num_sensor_channels=num_modalities,
                   window_length=WINDOW_SIZE,
                   n_classes=num_classes)  #

    # STATE DICT PATH
    state_dict_path = "trained_state_dicts/conv_lstm_20_window.pth"
    state_dict = torch.load(state_dict_path, weights_only=True)
    net.load_state_dict(state_dict)

    ## check if GPU is available
    run_on_gpu = torch.mps.is_available()
    if (run_on_gpu):
        print('Predicting on GPU!')
    else:
        print('No GPU available, training on CPU; consider making n_epochs very small.')
    # run training function

    if (run_on_gpu):
        net.to(device)
    
    test_h = net.init_hidden(batch_size)

    output = []
    targets_big = []
    net.eval()
    with torch.no_grad():
            # since I use cross validation, test set is basically validation set
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #test_h = tuple([each.data for each in test_h])

            if (run_on_gpu):
                inputs, targets = inputs.to(device), targets.to(device)

            out, test_h = net(inputs, test_h, batch_size)     
            output.append(out)
            targets_big.append(targets)
            # plt.plot(out.cpu().numpy())
            # plt.plot(targets.cpu().numpy())
            # plt.show(block=True)
    output = [out.cpu().numpy() for out in output]
    output = np.concatenate(output, axis=0)
    targets_big = [target.cpu().numpy() for target in targets_big]
    targets = np.concatenate(targets_big, axis=0)
    fig = plt.figure(figsize=(8,6))
    plt.title("train")
    plt.plot(targets, label='Actual Train')
    plt.plot(output, label='Predicted Train')
    plt.legend()
    plt.title('ConvLSTM Actual vs Predicted 5kHz Resistance (Training Set)')
    plt.show(block=False)
    mse_train = metrics.mean_squared_error(output, targets)
    r2_train = metrics.r2_score(output, targets)
    print(f'Mean Squared Error (Train Set): {mse_train:.2f}')
    print(f'R² Score (Train Set): {r2_train:.2f}')

    
    output = []
    targets_big = []
    net.eval()
    with torch.no_grad():
            # since I use cross validation, test set is basically validation set
        for batch_idx, (inputs, targets) in enumerate(testloader):
            test_h = tuple([each.data for each in test_h])

            if (run_on_gpu):
                inputs, targets = inputs.to(device), targets.to(device)

            out, test_h = net(inputs, test_h, batch_size)     
            output.append(out)
            targets_big.append(targets)
            # plt.plot(out.cpu().numpy())
            # plt.plot(targets.cpu().numpy())
            # plt.show(block=True)
    output = [out.cpu().numpy() for out in output]
    output = np.concatenate(output, axis=0)
    targets_big = [target.cpu().numpy() for target in targets_big]
    targets = np.concatenate(targets_big, axis=0)
    fig2 = plt.figure(figsize=(8,6))
    mse_test = metrics.mean_squared_error(output, targets)
    r2_test = metrics.r2_score(output, targets)
    print(f'Mean Squared Error (Train Set): {mse_test:.2f}')
    print(f'R² Score (Train Set): {r2_test:.2f}')
    plt.title("test")
    plt.plot(targets[:179399], label='Actual Test')
    plt.plot(output[:179399], label='Predicted Test', alpha=0.8)
    plt.legend()
    plt.title('ConvLSTM 5kHz Resistance Prediction (Subject 8)')

    plt.show(block=True)
    1
        
    return

if __name__ == "__main__":
    main()