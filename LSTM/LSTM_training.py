import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn

from make_LSTM_windows import *
from DeepConvLSTMModel import HARModel, init_weights
from sklearn import metrics


def train(net, train_loader, test_loader, epochs=10, batch_size=100, lr=0.001):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # todo: If it is a regression problem, change the following to nn.MSELoss()
    criterion = nn.MSELoss()

    if (train_on_gpu):
        net.to(device)

    epochs_list = []
    errors = []
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

        mae = metrics.mean_absolute_error(all_ground_truth, all_predictions)

        #acc = metrics.accuracy_score(all_ground_truth, all_predictions)
        print("Epoch: ", e)
        print(f"MAE: {mae}")
        test_results.append([mae, e])
        errors.append(mae)
        epochs_list.append(e)
        result_np = np.array(test_results, dtype=float)
        np.savetxt("lstm_results.csv", result_np, fmt='%.4f', delimiter=',')
    torch.save(net.state_dict(), "trained_state_dicts/weights.pth")
    fig, axs = plt.subplots(nrows=2)
    axs[0].plot(epochs_list, per_epoch_train_loss)
    axs[0].set_title("Training Loss (MAE)")
    axs[1].plot(epochs_list, errors)
    axs[1].set_title("Testing Loss (MAE)")
    fig.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":

    WINDOW_SIZE = 20  # 1 sec window with 50 Hz - CHANGE THIS
    num_modalities = 4  # number of signals you are fusing OR number of features
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
    
  

    features = ['knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance']
    output_feature = "knee_angle_l"
    subject = "11"

    print("feature vector")
    print(features)
    print("output feature")
    print(output_feature)
    print("subject")
    print(subject)

    train_x, test_x, train_y, test_y = generate_lstm_windows(WINDOW_SIZE, features, output_feature, subject)

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
    plt.plot(tmp_x)
    plt.show(block=True)

    # dataloader generation

    trainset = Data.TensorDataset(train_x, train_y)
    trainloader = Data.DataLoader(dataset=trainset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    testset = Data.TensorDataset(test_x, test_y)
    testloader = Data.DataLoader(dataset=testset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

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
    train(net, trainloader, testloader, epochs=400, batch_size=64, lr=0.001)