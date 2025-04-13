# code that uses epoch=150, subject embedding, velo, and CNN with LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Conv1D, LSTM, Bidirectional, Flatten, Dense, BatchNormalization, Dropout, Concatenate, Embedding, Reshape  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from IPython.display import Image, display


def smoother(X, window):
    df = pd.DataFrame(X)
    return df.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def create_cnn_lstm_windows(X, y, subject_ids, window_size, stride, smooth_for_deriv=True):
    X_windowed, y_windowed, subj_windowed = [], [], []
    for i in range(window_size, len(X), stride):
        X_win = X[i - window_size : i]  # shape (window_size, 4)
        dX = smoother(X_win, window=5) if smooth_for_deriv else X_win
        dX_win = np.gradient(dX, axis=0)
        combined = np.concatenate([X_win, dX_win], axis=1)  # (window_size, 8)
        X_windowed.append(combined)
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


subjects = [1, 3, 4, 5, 6, 7, 8, 11]
datadir = "/content/ece6254_project/DATASET/"
# datadir = "./DATASET/"

window_size = 50
stride = 2

# normalize globally
X_raw, y_raw, subject_ids_raw = load_all_subjects(subjects, datadir)
X_raw = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)
y_raw = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-8)

X_windowed, y_windowed, subject_ids_windowed = create_cnn_lstm_windows(X_raw, y_raw, subject_ids_raw, window_size, stride, smooth_for_deriv=True)

label_encoder = LabelEncoder()
subject_ids_encoded = label_encoder.fit_transform(subject_ids_windowed)

X_trainval, X_test, y_trainval, y_test, subj_trainval, subj_test = train_test_split(
    X_windowed, y_windowed, subject_ids_encoded, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val, subj_train, subj_val = train_test_split(X_trainval, y_trainval, subj_trainval, test_size=0.1111, random_state=42)

# Hybrid CNN + LSTM Model---------------
num_subjects = len(np.unique(subject_ids_encoded))
embedding_dim = 4

input_signal = Input(shape=(window_size, 8))
x = Conv1D(64, 5, activation="relu")(input_signal)
x = BatchNormalization()(x)
x = Conv1D(32, 3, activation="relu")(x)
x = BatchNormalization()(x)
x = Bidirectional(LSTM(64, return_sequences=False))(x)
x = Dropout(0.3)(x)

input_subject = Input(shape=(1,))
subject_embed = Embedding(input_dim=num_subjects, output_dim=embedding_dim)(input_subject)
subject_embed = Reshape((embedding_dim,))(subject_embed)

x = Concatenate()([x, subject_embed])
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1)(x)

model = Model(inputs=[input_signal, input_subject], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# end model---------------------------


early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(
    [X_train, subj_train], y_train, validation_data=([X_val, subj_val], y_val), epochs=150, batch_size=128, callbacks=[early_stop], verbose=2
)


y_pred = model.predict([X_test, subj_test]).flatten()
print("\nTest MSE: {:.4f}".format(mean_squared_error(y_test, y_pred)))
print("Test MAE: {:.4f}".format(mean_absolute_error(y_test, y_pred)))
print("Test R^2: {:.4f}".format(r2_score(y_test, y_pred)))


plt.figure(figsize=(10, 4))
plt.plot(y_test[:1000], label="True")
plt.plot(y_pred[:1000], label="Predicted")
plt.legend()
plt.title("CNN+LSTM: Predicted vs True")
plt.grid(True)
plt.tight_layout()
plt.savefig("hybrid_pred.png", dpi=300)

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("hybrid_loss.png", dpi=300)
