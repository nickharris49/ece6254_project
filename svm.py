import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# r5, x5, r100, x100 -> feature vector

def main():

    # pathing
    datadir = "DATASET/"
    datafiles = os.listdir(datadir)
    
    X_path = datadir + "ankle_feature_vector_full_11.npy" # change this to change the input feature vector

    y_path = datadir + "ankle_y_11.npy"

    data_path = datadir + "ankle_base_11.npy"

    full_data = np.load(data_path)
    input = [full_data['ankle_bioz_5k_resistance'], full_data['ankle_bioz_5k_reactance'], full_data['ankle_bioz_100k_resistance'], full_data['ankle_bioz_100k_reactance']]
    input = np.stack(input, axis=2)
    # input = np.reshape(input, (np.shape(input)[0], np.shape(input)[1]*np.shape(input)[2]))

    # output = full_data['ankle_angle_r']
    # output = np.reshape(output, np.shape(output)[0]*np.shape(output)[1])
    # plt.plot(output[:9000])
    # plt.show(block=True)
    # input[0] = np.arctan(input[1] / input[0])
    # input[2] = np.sqrt(np.square(input[2]) + np.square(input[3]))
    # for i in range(np.shape(output)[0]):
    #     fig, axs = plt.subplots(nrows=5, ncols=2)
    #     for j in range(np.shape(input)[0]):
    #         axs[j,0].plot(input[j,i,:1000])
    #         axs[j,1].scatter(input[j,i,:1000], output[i,:1000], alpha=0.1)
    #     axs[4,0].plot(output[i,:1000])
    #     axs[4,1].scatter(output[i,:1000],output[i,:1000], alpha=0.01)
    #     plt.show(block=True)
    #     plt.scatter(resist_10k[i], output[i], alpha=0.01)
    #     plt.show(block=True)
    # load in data

    X = np.load(X_path)
    # compute magnitude (optional)
    y = np.load(y_path)

    y = X[:,0]
    X = X[:,-2:]

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)
    # X = input
    # y = output

    # split into train, val, and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42, shuffle=True)
    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    y_test = y_test.squeeze()

    # features = ["r5k", "x5k", "r100k", "x100k"]
    # fig, axs = plt.subplots(nrows=np.shape(X)[1]+1)
    # for i in range(len(axs) -  1):
    #     axs[i].plot(X[:1000,i])
    #     axs[i].set_ylabel(features[i])

    # axs[np.shape(X)[1]].plot(y[:1000])
    # axs[np.shape(X)[1]].set_ylabel("knee angle")
    # plt.show(block=True)
    
    # README
    # Linear regressor seems to be pretty not uh, good -> 0.0035415044310641575
    # Seems to get a bit better when just using one subj -> 0.010846252870242545
    # for what it's worth, these both fucking suck!!

    # LinearRegressor
    reg = LinearRegression().fit(X_train, y_train)
    pred = reg.predict(X_val)
    score = reg.score(X_val, y_val)
    print(score)

    y_pred = reg.predict(X_test)
    score_lin = reg.score(X_test, y_test)
    mse = np.mean(np.square(y_pred - y_test))

    print("MSE: " + str(mse))
    print("R2 Score: " + str(score_lin))

    plt.plot(y_test)
    plt.plot(y_pred)
    plt.title("Linear Regression")
    plt.xlabel("Time (Samples, 100Hz)")
    plt.ylabel("Ankle Angle")
    plt.show(block=True)

    # Linear SVR
    # pretty weird, negative score now -> -0.03359
    svr_lin = svm.SVR()
    svr_lin = svm.SVR(kernel="linear", C=1.0, gamma='scale')
    svr_lin.fit(X_train[:5000], y_train[:5000])
    score_lin = svr_lin.score(X_val[:1000], y_val[:1000])
    print(score_lin)

    y_pred = svr_lin.predict(X_test)
    score_lin = svr_lin.score(X_test, y_test)
    mse = np.mean(np.square(y_pred - y_test))

    print("MSE: " + str(mse))
    print("R2 Score: " + str(score_lin))

    plt.plot(y_test)
    plt.plot(y_pred)
    plt.title("Support Vector Regression (Linear)")
    plt.xlabel("Time (Samples, 100Hz)")
    plt.ylabel("Ankle Angle")
    plt.show(block=True)

    # RBF SVR
    # Score is better for sure, but still not ideal -> 0.0222
    # Admittedly, this was without any sort of search through params

    svr_rbf = svm.SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1)
    svr_rbf.fit(X_train[:5000], y_train[:5000])
    score_rbf = svr_rbf.score(X_val[:1000], y_val[:1000])


    y_pred = svr_rbf.predict(X_test)
    score_rbf = svr_rbf.score(X_test, y_test)
    mse = np.mean(np.square(y_pred - y_test))

    print("MSE: " + str(mse))
    print("R2 Score: " + str(score_rbf))

    plt.plot(y_test)
    plt.plot(y_pred)

    plt.title("Support Vector Regression (RBF)")
    plt.xlabel("Time (Samples, 100Hz)")
    plt.ylabel("Ankle Angle")
    plt.show(block=True)

    from sklearn.model_selection import GridSearchCV
    # param_grid = {
    #     'C': [0.01, 0.1, 1, 10, 100],
    #     'epsilon': [0.01, 0.1, 1, 10, 100],
    #     'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10, 100]
    # }   
    # {'C': 0.1, 'epsilon': 1, 'gamma': 10}
    
    # param_grid = {
    #     'C': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    #     'epsilon': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    #     'gamma': ['scale', 'auto', 0.00001, 0.0001, 0.001, 0.01, 0.1]
    # }   
    # {'C': 0.1, 'epsilon': 0.1, 'gamma': 'auto'}
    
    # param_grid = {
    #     'C': np.logspace(-2, 7, 7),
    #     'epsilon': np.logspace(-4, 1, 9),
    #     'gamma': np.logspace(1, 7, 7),
    # }   

    # svr = svm.SVR(kernel='rbf')
    # grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2')
    # grid_search.fit(X_train[:5000], y_train[:5000])
    # print(grid_search.best_params_)

    svr_rbf = svm.SVR(kernel='rbf', gamma=10, C=1.0, epsilon=1)
    svr_rbf.fit(X_train[:5000], y_train[:5000])
    score_rbf = svr_rbf.score(X_val[:1000], y_val[:1000])

    print(score_rbf)

    y_pred = svr_rbf.predict(X_test)
    score_rbf = svr_rbf.score(X_test, y_test)
    mse = np.mean(np.square(y_pred - y_test))

    print("MSE: " + str(mse))
    print("R2 Score: " + str(score_rbf))
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.title("Support Vector Regression (RBF)")
    plt.xlabel("Time (Samples, 100Hz)")
    plt.ylabel("Ankle Angle")
    plt.show(block=True)

    ## model too simple for the data I fear

    # performance seems pretty equivalent for full and 100k, so maybe 5k isn't really adding that much utility?
    # magnitude performs worse :(

    dummy = 1

if __name__ == '__main__':
    main()