import numpy as np


def load_subject_data(subjects, datadir):
    X_all, y_all = [], []
    for subject in subjects:
        X = np.load(f"{datadir}/ankle_feature_vector_full_normalized_{subject}.npy")  # (N, 4)
        y = np.load(f"{datadir}/ankle_y_normalized_{subject}.npy").reshape(-1)  # (N,)
        X_all.append(X)
        y_all.append(y)
    return np.vstack(X_all), np.concatenate(y_all)


def create_windows(X, y, window_size=100, stride=10):
    X_windows = []
    y_windows = []
    for i in range(0, len(X) - window_size + 1, stride):
        X_windows.append(X[i : i + window_size])  # shape (window_size, 4)
        y_windows.append(y[i + window_size // 2])  # target: middle of window
    return np.array(X_windows), np.array(y_windows)


datadir = "./DATASET/"
window_size = 100
stride = 10
subjects = [1, 3, 4, 5, 6, 7, 8, 11]

train_subjects = [1, 3, 4, 5, 6]
val_subjects = [7, 8]
test_subjects = [11]

X_train_raw, y_train_raw = load_subject_data(train_subjects, datadir)
X_val_raw, y_val_raw = load_subject_data(val_subjects, datadir)
X_test_raw, y_test_raw = load_subject_data(test_subjects, datadir)

X_train, y_train = create_windows(X_train_raw, y_train_raw, window_size, stride)
X_val, y_val = create_windows(X_val_raw, y_val_raw, window_size, stride)
X_test, y_test = create_windows(X_test_raw, y_test_raw, window_size, stride)


print("Train set:", X_train.shape, y_train.shape)
print("Val set:  ", X_val.shape, y_val.shape)
print("Test set: ", X_test.shape, y_test.shape)
