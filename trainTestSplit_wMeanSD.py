import time
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# issue this code solves: when using the normalized data that gets mixed up with all the other subjects in train_test_split, 
# we lose the ability to un-normalize because we don't know which y corresponds to which

import numpy as np
import matplotlib.pyplot as plt

def main():
    datadir = "./DATASET/"

    # List of subjects
    subjects = [1, 2, 3, 5, 6, 7, 8, 11]  # Add all subject IDs here

    # Initialize lists to hold data, labels, and normalization parameters
    X_list_raw = []
    y_list_raw = []
    X_list_norm = []
    y_list_norm = []
    subject_list = []
    mean_std_dict_X = {}
    mean_std_dict_y = {}

    # Load data for each subject and store normalization parameters
    for subject in subjects:
        try:
            # not normalized
            X_subject_raw = np.load(datadir + f"feature_vector_full_{subject}.npy")
            y_subject_raw = np.load(datadir + f"y_{subject}.npy").reshape(-1)
            # feature vector normalized
            X_subject_norm = np.load(datadir + f"feature_vector_full_normalized_{subject}.npy")
            y_subject_norm = np.load(datadir + f"y_normalized_{subject}.npy").reshape(-1)

            mean_X = np.mean(X_subject_raw, axis=0)
            std_X = np.std(X_subject_raw, axis=0)
            mean_std_dict_X[subject] = (mean_X, std_X)
            
            mean_y = np.mean(y_subject_raw)
            std_y = np.std(y_subject_raw)
            mean_std_dict_y[subject] = (mean_y, std_y)
            
            X_list_raw.append(X_subject_raw)
            y_list_raw.append(y_subject_raw)
            X_list_norm.append(X_subject_norm)
            y_list_norm.append(y_subject_norm)
            subject_list.extend([subject] * len(y_subject_norm))
        except FileNotFoundError as e:
            print(f"File not found for subject {subject}: {e}")

    # Combine data from all subjects
    if len(X_list_norm) == 0 or len(y_list_norm) == 0:
        print("No data loaded. Please check the file paths and subjects.")
        return

    X_raw = np.concatenate(X_list_raw, axis=0)
    y_raw = np.concatenate(y_list_raw, axis=0)
    X_norm = np.concatenate(X_list_norm, axis=0)
    y_norm = np.concatenate(y_list_norm, axis=0)
    subjects_array = np.array(subject_list)

    # # Split the data using stratified sampling based on subject IDs
    # X_train, X_temp, y_train, y_temp, subjects_train, subjects_temp = train_test_split(
    #     X, y, subjects_array, test_size=0.2, random_state=42, stratify=subjects_array
    # )

    # X_val, X_test, y_val, y_test, subjects_val, subjects_test = train_test_split(
    #     X_temp, y_temp, subjects_temp, test_size=0.4, random_state=42, stratify=subjects_temp
    # )

    # implement ML here

    # Function to unnormalize data
    def unnormalize_data(X_or_y, subjects, mean_std_dict):
        unnormalized_data = np.zeros_like(X_or_y)
        for i, subject in enumerate(subjects):
            mean, std = mean_std_dict[subject]
            unnormalized_data[i] = X_or_y[i] * std + mean
        return unnormalized_data

    # # Unnormalize the feature vectors and predictions
    # X_train_unnormalized = unnormalize_data(X_train, subjects_train, mean_std_dict_X)
    # X_val_unnormalized = unnormalize_data(X_val, subjects_val, mean_std_dict_X)
    # X_test_unnormalized = unnormalize_data(X_test, subjects_test, mean_std_dict_X)

    # y_train_unnormalized = unnormalize_data(y_train, subjects_train, mean_std_dict_y)
    # y_val_unnormalized = unnormalize_data(y_val, subjects_val, mean_std_dict_y)
    # y_test_unnormalized = unnormalize_data(y_test, subjects_test, mean_std_dict_y)

    # Unnormalize the predictions
    y_unnormalized = unnormalize_data(y_norm, subjects_array, mean_std_dict_y)

    # Plot every 1000th point of raw and unnormalized together in the whole dataset
    plt.figure(figsize=(12, 6))

    plt.plot(y_raw[::1000], label='Raw')
    plt.plot(y_unnormalized[::1000], label='Unnormalized')
    plt.title('Predictions (Whole Dataset)')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()