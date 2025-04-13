# THIS IS THE CODE THAT WORKS THE BEST ON ALL SUBJECTS
# uses all subjects and train,val,split randomly

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, BatchNormalization, Dropout, Concatenate  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from IPython.display import Image, display


def create_cnn_windows(X, y, subject_ids, window_size, stride):
    X_windowed, y_windowed, subj_windowed = [], [], []
    for i in range(window_size, len(X), stride):
        X_windowed.append(X[i - window_size : i])
        y_windowed.append(y[i])
        subj_windowed.append(subject_ids[i])
    return np.array(X_windowed), np.array(y_windowed), np.array(subj_windowed)


def load_all_subjects(subjects, datadir):
    X_all, y_all, subj_all = [], [], []
    for subject in subjects:
        X = np.load(f"{datadir}/ankle_feature_vector_full_normalized_{subject}.npy")
        y = np.load(f"{datadir}/ankle_y_normalized_{subject}.npy").reshape(-1)
        subj_vec = np.full(len(y), subject)
        X_all.append(X)
        y_all.append(y)
        subj_all.append(subj_vec)
    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(subj_all)


# -------------------------------

subjects = [1, 3, 4, 5, 6, 7, 8, 11]
window_size = 50
stride = 2
datadir = "/content/ece6254_project/DATASET/"
# datadir = "./DATASET/"

X_raw, y_raw, subject_ids_raw = load_all_subjects(subjects, datadir)
X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8
X_raw = (X_raw - X_mean) / X_std
y_mean, y_std = y_raw.mean(), y_raw.std() + 1e-8
y_raw = (y_raw - y_mean) / y_std

X_windowed, y_windowed, subject_ids_windowed = create_cnn_windows(X_raw, y_raw, subject_ids_raw, window_size, stride)
encoder = OneHotEncoder(sparse_output=False)
subject_ids_encoded = encoder.fit_transform(subject_ids_windowed.reshape(-1, 1))


X_trainval, X_test, y_trainval, y_test, subj_trainval, subj_test = train_test_split(
    X_windowed, y_windowed, subject_ids_encoded, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val, subj_train, subj_val = train_test_split(X_trainval, y_trainval, subj_trainval, test_size=0.1111, random_state=42)

print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")

# -------------------------------
#  model with subject id input
input_signal = Input(shape=(window_size, 4))
x = Conv1D(256, 7, activation="relu")(input_signal)
x = BatchNormalization()(x)
x = Conv1D(128, 5, activation="relu")(x)
x = BatchNormalization()(x)
x = Conv1D(64, 3, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Flatten()(x)

input_subject = Input(shape=(subj_train.shape[1],))
x = Concatenate()([x, input_subject])
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1)(x)

model = Model(inputs=[input_signal, input_subject], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# -------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(
    [X_train, subj_train],
    y_train,
    validation_data=([X_val, subj_val], y_val),
    epochs=75,
    batch_size=128,
    callbacks=[early_stop],
)


y_pred = model.predict([X_test, subj_test]).flatten()
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"window size: {window_size} and stride: {stride}")
print(f"\nTest MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R² Score: {r2:.4f}")

# -------------------------------
# Per-subject R^2, MSE, MAE
subject_labels = encoder.inverse_transform(subj_test)
unique_subjects = np.unique(subject_labels)

print("\nPer-Subject Evaluation:")
for subj in unique_subjects:
    idx = subject_labels.flatten() == subj
    y_true_subj = y_test[idx]
    y_pred_subj = y_pred[idx]

    if len(y_true_subj) > 0:
        mse_subj = mean_squared_error(y_true_subj, y_pred_subj)
        mae_subj = mean_absolute_error(y_true_subj, y_pred_subj)
        r2_subj = r2_score(y_true_subj, y_pred_subj)
        print(f"Subject {int(subj)} — R²: {r2_subj:.4f}, MSE: {mse_subj:.4f}, MAE: {mae_subj:.4f}")

# -------------------------------

plt.figure(figsize=(10, 4))
plt.plot(y_test[:1000], label="True")
plt.plot(y_pred[:1000], label="Predicted")
plt.title("Predicted vs True Knee Angle")
plt.xlabel("Sample")
plt.ylabel("Normalized Knee Angle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("1dCNN1.png", dpi=300)

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("1dCNN2.png", dpi=300)

display(Image("1dCNN1.png"))
display(Image("1dCNN2.png"))
