import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn

from make_LSTM_windows import *
from DeepConvLSTMModel import HARModel, init_weights
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
import time
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math

def train(net, train_loader, test_loader, epochs=10, batch_size=100, lr=0.001):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

    # todo: If it is a regression problem, change the following to nn.MSELoss()
    criterion = nn.MSELoss()

    if (train_on_gpu):
        net.to(device)

    epochs_list = []
    errors = []
    r2s = []
    per_epoch_train_loss = []
    for e in range(epochs):

        # initialize hidden state
        h = net.init_hidden(batch_size)
        train_losses = []
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if (train_on_gpu):
                inputs, targets = inputs.to(device), targets.to(device)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            # zero accumulated gradients
            opt.zero_grad()

            # get the output from the model
            output, h = net(inputs, h, batch_size)
            loss = criterion(output, targets)
            train_losses.append(loss.item())
            loss.backward()
            opt.step()
        per_epoch_train_loss.append(np.min(train_losses))

        val_h = net.init_hidden(batch_size)
        val_losses = []

        net.eval()

        all_ground_truth = []
        all_predictions = []
        with torch.no_grad():
            # since I use cross validation, test set is basically validation set
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                val_h = tuple([each.data for each in val_h])

                if (train_on_gpu):
                    inputs, targets = inputs.to(device), targets.to(device)

                output, val_h = net(inputs, val_h, batch_size)

                val_loss = criterion(output, targets)
                val_losses.append(val_loss.item())
                all_ground_truth = all_ground_truth + targets.to(device).view(-1).tolist()
                all_predictions = all_predictions + output.to(device).view(-1).tolist()
                ## NICK TODO: implement some better scoring metrics

                # todo: following three lines are for classification framework - you do not need them if it is a regression problem
                # top_p, top_class = output.topk(1, dim=1)
                # all_ground_truth = all_ground_truth + torch.max(targets, 1)[1].cuda().view(-1).tolist()
                # all_predictions = all_predictions + top_class.view(-1).tolist()

        net.train()  # reset to train mode after iterating through validation data
        # todo: you can replace the following metrics with the suitable ones as well

        ## NICK TODO: replace these with appropriate metrics for regression

        mse = metrics.mean_squared_error(all_ground_truth, all_predictions)
        r2 = metrics.r2_score(all_ground_truth, all_predictions)
        #acc = metrics.accuracy_score(all_ground_truth, all_predictions)
        print("Epoch: ", e)
        print(f"MSE: {mse}")
        print(f"R2: {r2}")
        test_results.append([mse, r2, e])
        errors.append(mse)
        epochs_list.append(e)
        r2s.append(r2)
        result_np = np.array(test_results, dtype=float)
        np.savetxt("lstm_results.csv", result_np, fmt='%.4f', delimiter=',')
        scheduler.step(val_loss)

    torch.save(net.state_dict(), "trained_state_dicts/weights.pth")
    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(epochs_list, per_epoch_train_loss)
    axs[0].set_title("Training Loss (MSE)")
    axs[1].plot(epochs_list, errors)
    axs[1].set_title("Testing Loss (MSE)")
    fig.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":

    WINDOW_SIZE = 20  # 1 sec window with 50 Hz - CHANGE THIS
    BATCH_SIZE = 32
    num_classes = 1 # one because this is regression
    test_results = []

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
    # subjects_ankle_base = [5, 6, 7, 8, 11, 3, 4]

    num_modalities = len(features)  # number of signals you are fusing OR number of features

    print("feature vector")
    print(features)
    print("output feature")
    print(output_feature)
    print("subject")
    print(subject)

    # print("subjects")
    # print(subjects)

    train_x, test_x, train_y, test_y = generate_lstm_windows_multiple_subs(WINDOW_SIZE, features, output_feature, subjects_ankle_base, window_frequency=20)
    #train_x, test_x, train_y, test_y = generate_lstm_windows_multiple_subs(WINDOW_SIZE, features, output_feature, subjects)
    
    ## MLP training stuff
    X_train = train_x.reshape((np.shape(train_x)[0], -1))
    y_train = train_y.reshape((np.shape(train_y)[0], -1))
    X_test = test_x.reshape((np.shape(test_x)[0], -1))
    y_test = test_y.reshape((np.shape(test_y)[0], -1))

    mlp = MLPRegressor(hidden_layer_sizes=(40, 40), max_iter=300, alpha=0.01, random_state=42, verbose=True, activation='relu')
    mlp.fit(X_train, y_train)

    # Save the model parameters to a file
    joblib.dump(mlp, 'Results/mlp_model_20_window.pkl')
    train_x, test_x, train_y, test_y = generate_lstm_windows_multiple_subs(WINDOW_SIZE, features, output_feature, subjects_ankle_base, window_frequency=1)

    X_train = train_x.reshape((np.shape(train_x)[0], -1))
    y_train = train_y.reshape((np.shape(train_y)[0], -1))
    X_test = test_x.reshape((np.shape(test_x)[0], -1))
    y_test = test_y.reshape((np.shape(test_y)[0], -1))
    # Predictions for the test set
    y_pred_test = mlp.predict(X_test)

    # Save the predicted values for the test set to a file
    np.save('Results/y_pred_test_MLP_20_window.npy', y_pred_test)

    # Calculate Mean Squared Error for the test set
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f'Mean Squared Error (Test Set): {mse_test:.2f}')

    # Calculate R² Score for the test set
    r2_test = r2_score(y_test, y_pred_test)
    print(f'R² Score (Test Set): {r2_test:.2f}')

    # Predictions for the training set
    y_pred_train = mlp.predict(X_train)

    end_time = time.time()

    # Save the predicted values for the training set to a file
    np.save('Results/y_pred_train_MLP_20_window.npy', y_pred_train)

    # Calculate Mean Squared Error for the training set
    mse_train = mean_squared_error(y_train, y_pred_train)
    print(f'Mean Squared Error (Training Set): {mse_train:.2f}')

    # Calculate R² Score for the training set
    r2_train = r2_score(y_train, y_pred_train)
    print(f'R² Score (Training Set): {r2_train:.2f}')

    # Plot the actual and predicted signals for the training set
    plt.figure(figsize=(8, 6))
    plt.plot(y_train[60100:60400], label='Actual Train')
    plt.plot(y_pred_train[60100:60400], label='Predicted Train')
    plt.legend()
    plt.title('MLP Actual vs Predicted 5kHz Resistance (Training Set)')
    plt.show(block=False)

    # Plot the actual and predicted signals for the test set
    plt.figure(figsize=(8, 6))
    plt.plot(y_test[20400:20700], label='Actual Test')
    plt.plot(y_pred_test[20400:20700], label='Predicted Test')
    plt.legend()
    plt.title('MLP Actual vs Predicted 5kHz Resistance (Test Set)')
    plt.show(block=False)


    train_x, test_x, train_y, test_y = generate_lstm_windows_multiple_subs(WINDOW_SIZE, features, output_feature, subjects_ankle_base, window_frequency=20)
    # conversion to torch and transferring data to GPU in preparation for training the LSTM
    shape = train_x.shape
    train_x = torch.from_numpy(np.reshape(train_x.astype(float), [shape[0], 1, shape[1], shape[2]]))
    train_x = train_x.type(torch.FloatTensor).to(device)
    train_y = torch.from_numpy(train_y)
    train_y = train_y.type(torch.FloatTensor).to(device)
    test_x = torch.from_numpy(np.reshape(test_x.astype(float), [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]))
    test_x = test_x.type(torch.FloatTensor).to(device)
    test_y = torch.from_numpy(test_y.astype(np.float32))
    test_y = test_y.type(torch.FloatTensor).to(device)

    tmp_x = np.squeeze(train_x[50].cpu().numpy())
    fig = plt.figure()
    plt.plot(tmp_x)
    plt.show(block=False)

    # dataloader generation

    trainset = Data.TensorDataset(train_x, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_x, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    print("Data shapes!")
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    net = HARModel(num_sensor_channels=num_modalities,
                   window_length=WINDOW_SIZE,
                   n_classes=num_classes)  #
    net.apply(init_weights)

    ## check if GPU is available
    train_on_gpu = torch.mps.is_available()
    if (train_on_gpu):
        print('Training on GPU!')
    else:
        print('No GPU available, training on CPU; consider making n_epochs very small.')
    # run training function
    train(net, trainloader, testloader, epochs=50, batch_size=BATCH_SIZE, lr=0.001)

    ########################################
    ## CODE TO TEST THE MODEL
    ########################################
    torch.mps.empty_cache()
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

    tmp_x = np.squeeze(train_x[50].cpu().numpy())
    fig = plt.figure()
    plt.plot(tmp_x)
    plt.show(block=False)

    trainset = Data.TensorDataset(train_x, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_x, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

    print("Data shapes!")
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    batch_size = BATCH_SIZE
    run_on_gpu = train_on_gpu
    output = []
    targets_big = []
    test_h = net.init_hidden(batch_size)

    net.eval()
    with torch.no_grad():
        # since I use cross validation, test set is basically validation set
        for batch_idx, (inputs, targets) in enumerate(trainloader):
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
    fig = plt.figure()
    plt.title("ConvLSTM Actual vs Predicted 5kHz Resistance (Training Set)")
    plt.plot(targets[60100:60400], label="Actual Train")
    plt.plot(output[60100:60400], label="Predicted Train")
    plt.legend()
    plt.show(block=False)

    
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
    fig2 = plt.figure()
    plt.title("ConvLSTM Actual vs Predicted 5kHz Resistance (Testing Set)")
    plt.plot(targets[20400:20700], label="Actual Test")
    plt.plot(output[20400:20700], label="Predicted Test")
    plt.legend()

    plt.show(block=True)    