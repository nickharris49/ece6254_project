import time
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def main():
    datadir = "DATASET/"

    # List of subjects
    subjects = [1, 3, 5, 6, 7, 8, 11]  # Add all subject IDs here
    # subjects = [1]

    # Initialize lists to hold data and labels
    X_list = []
    y_list = []
    subject_list = []

    # Load data for each subject
    for subject in subjects:
        X_subject = np.load(datadir + f"ankle_feature_vector_full_normalized_{subject}.npy")
        y_subject = np.load(datadir + f"ankle_y_normalized_{subject}.npy").reshape(-1)
        
        X_list.append(X_subject)
        y_list.append(y_subject)
        subject_list.extend([subject] * len(y_subject))

    # Combine data from all subjects
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subjects_array = np.array(subject_list)

    # Use the last two columns as features and the first column as the target
    X = X[:, -2:]
    y = X[:, 0]

    # Split the data using stratified sampling based on subject IDs
    # X_train, X_temp, y_train, y_temp, subjects_train, subjects_temp = train_test_split(
    #     X, y, subjects_array, test_size=0.2, random_state=42, stratify=subjects_array, shuffle=False
    # )

    # X_val, X_test, y_val, y_test, subjects_val, subjects_test = train_test_split(
    #     X_temp, y_temp, subjects_temp, test_size=0.4, random_state=42, stratify=subjects_temp, shuffle=False
    # )

    X_train, X_temp, y_train, y_temp, subjects_train, subjects_temp = train_test_split(
        X, y, subjects_array, test_size=0.2, random_state=42, shuffle=False
    )

    X_val, X_test, y_val, y_test, subjects_val, subjects_test = train_test_split(
        X_temp, y_temp, subjects_temp, test_size=0.4, random_state=42, shuffle=False
    )

    print(f"Train Set: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation Set: X={X_val.shape}, y={y_val.shape}")
    print(f"Test Set: X={X_test.shape}, y={y_test.shape}")

    print("First 5 datapoints in X_train:")
    print(X_train[:5])

    print("First 5 datapoints in y_train:")
    print(y_train[:5])

    start_time = time.time()

    # Create and train the MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=300, alpha=0.001, random_state=42)
    mlp.fit(X_train, y_train)

    # Save the model parameters to a file
    joblib.dump(mlp, 'Results/mlp_model.pkl')

    # Predictions for the test set
    y_pred_test = mlp.predict(X_test)

    # Save the predicted values for the test set to a file
    np.save('Results/y_pred_test_MLP.npy', y_pred_test)

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
    np.save('Results/y_pred_train_MLP.npy', y_pred_train)

    # Calculate Mean Squared Error for the training set
    mse_train = mean_squared_error(y_train, y_pred_train)
    print(f'Mean Squared Error (Training Set): {mse_train:.2f}')

    # Calculate R² Score for the training set
    r2_train = r2_score(y_train, y_pred_train)
    print(f'R² Score (Training Set): {r2_train:.2f}')

    # Plot the actual and predicted signals for the training set
    plt.figure(figsize=(10, 6))
    plt.plot(y_train, label='Actual Train')
    plt.plot(y_pred_train, label='Predicted Train')
    plt.legend()
    plt.title('Actual vs Predicted Signal (Training Set)')
    plt.show()

    # Plot the actual and predicted signals for the test set
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Test')
    plt.plot(y_pred_test, label='Predicted Test')
    plt.legend()
    plt.title('Actual vs Predicted Signal (Test Set)')
    plt.show()

    # Print the elapsed time
    elapsed_time = end_time - start_time
    elapsed_min = elapsed_time / 60
    print(f"Time taken to run the code: {elapsed_min:.2f} minutes")

if __name__ == '__main__':
    main()
