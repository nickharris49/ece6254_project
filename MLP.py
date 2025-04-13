import time
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def main():
    datadir = "./DATASET/"

    # # To run on stratified list of all subjects
    # List of subjects
    subjects = [1, 2, 3, 5, 6, 7, 8, 11]  # Add all subject IDs here

    # Initialize lists to hold data and labels
    X_list = []
    y_list = []
    subject_list = []

    # Load data for each subject
    for subject in subjects:
        X_subject = np.load(datadir + f"feature_vector_full_normalized_{subject}.npy")
        y_subject = np.load(datadir + f"y_normalized_{subject}.npy").reshape(-1)
        
        X_list.append(X_subject)
        y_list.append(y_subject)
        subject_list.extend([subject] * len(y_subject))

    # Combine data from all subjects
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subjects_array = np.array(subject_list)

    # Split the data using stratified sampling based on subject IDs
    X_train, X_temp, y_train, y_temp, subjects_train, subjects_temp = train_test_split(
        X, y, subjects_array, test_size=0.2, random_state=42, stratify=subjects_array
    )

    X_val, X_test, y_val, y_test, subjects_val, subjects_test = train_test_split(
        X_temp, y_temp, subjects_temp, test_size=0.4, random_state=42, stratify=subjects_temp
    )

    # # for reading in just 1 subject file:
    # X = np.load(datadir + "feature_vector_full_normalized.npy") # 5k and 100k not normalized
    # y = np.load(datadir + "y_normalized.npy").reshape(-1)
    # # just pushed a normalized version of the data, just subtracted the mean and divided 
    # # by standard deviation for each 60s window to get the bioimpedance and angle data a little bit more similar across subjects
    # # up to y'all whether you want to use the unnormalized or normalized data, but they're both there

    print(f"Train Set: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation Set: X={X_val.shape}, y={y_val.shape}")
    print(f"Test Set: X={X_test.shape}, y={y_test.shape}")

    # checking dataformat
    # Show the first 5 datapoints in X_test and y_test
    print("First 5 datapoints in X_train:") # 4 cols are 100k and 5k - resistance and reactance
    print(X_train[:5])

    print("First 5 datapoints in y_train:") # typically knee angle is 0-55ish deg. but these values are normalized.
    # to calculate back the true value - need to multiply by the sd and add the mean for each window.
    print(y_train[:5])

    #######for running 1 set of params
    start_time = time.time()

    # to use parameters already found:
    # # Load the model from the file
    # loaded_mlp = joblib.load('mlp_model.pkl')
    # # Use the loaded model to make predictions on new data
    # new_data_predictions = loaded_mlp.predict(new_data)

    # Create and train the MLPRegressor - bc output angle is cts var.
    # # start with 1 hidden layer and go from there, 100 neurons
    # mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=300, alpha=0.001, random_state=42,)
    mlp.fit(X_train, y_train)

    # Save the model parameters to a file
    joblib.dump(mlp, 'mlp_model2.pkl')

    # Predictions for the test set
    y_pred_test = mlp.predict(X_test)

    # Save the predicted values for the test set to a file
    np.save('y_pred_test2.npy', y_pred_test)

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
    np.save('y_pred_train2.npy', y_pred_train)

    # Calculate Mean Squared Error for the training set
    mse_train = mean_squared_error(y_train, y_pred_train)
    print(f'Mean Squared Error (Training Set): {mse_train:.2f}')

    # Calculate R² Score for the training set
    r2_train = r2_score(y_train, y_pred_train)
    print(f'R² Score (Training Set): {r2_train:.2f}')

    # Print the elapsed time
    elapsed_time = end_time - start_time
    elapsed_min = elapsed_time/60
    # print(f"Time taken to run the code: {elapsed_time:.2f} seconds")
    print(f"Time taken to run the code: {elapsed_min:.2f} minutes")

    # using mlp_model params: 
    # Mean Squared Error (Test Set): 0.97
    # R² Score (Test Set): 0.03
    # Mean Squared Error (Training Set): 0.97
    # R² Score (Training Set): 0.03
    # Time taken to run the code: 2.48 minutes

# uncomment for running the gridsearch for ideal params.
    # start_time = time.time()
    # # started ~ 4:25 pm, finished 11:13 pm.

    # # Define the parameter grid
    # param_grid = {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    #     'max_iter': [300, 500, 700],
    #     'alpha': [0.0001, 0.001, 0.01]  # L2 regularization parameter
    # }

    # # Create the MLPRegressor
    # mlp = MLPRegressor(random_state=42)

    # # Perform grid search
    # grid_search = GridSearchCV(mlp, param_grid, cv=3)
    # grid_search.fit(X_train, y_train)

    # end_time = time.time()

    # # Print the best parameters
    # print(f'Best parameters: {grid_search.best_params_}')

    # # #gridsearch run 3/30/25 daytime
    # # Best parameters: {'alpha': 0.0001, 'hidden_layer_sizes': (100, 100), 'max_iter': 300}
    # # Mean Squared Error (Test Set): 0.96
    # # R² Score (Test Set): 0.03
    # # Mean Squared Error (Training Set): 0.96
    # # R² Score (Training Set): 0.04

    # # Use the best estimator for predictions
    # best_mlp = grid_search.best_estimator_

    # # Predictions for the test set
    # y_pred_test = best_mlp.predict(X_test)

    # # Calculate Mean Squared Error for the test set
    # mse_test = mean_squared_error(y_test, y_pred_test)
    # print(f'Mean Squared Error (Test Set): {mse_test:.2f}')

    # # Calculate R² Score for the test set
    # r2_test = r2_score(y_test, y_pred_test)
    # print(f'R² Score (Test Set): {r2_test:.2f}')

    # # Predictions for the training set
    # y_pred_train = best_mlp.predict(X_train)

    # # Calculate Mean Squared Error for the training set
    # mse_train = mean_squared_error(y_train, y_pred_train)
    # print(f'Mean Squared Error (Training Set): {mse_train:.2f}')

    # # Calculate R² Score for the training set
    # r2_train = r2_score(y_train, y_pred_train)
    # print(f'R² Score (Training Set): {r2_train:.2f}')

    # # Print the elapsed time
    # elapsed_time = end_time - start_time
    # elapsed_min = elapsed_time/60
    # # print(f"Time taken to run the code: {elapsed_time:.2f} seconds")
    # print(f"Time taken to run the code: {elapsed_min:.2f} minutes")

if __name__ == '__main__':
    main()
