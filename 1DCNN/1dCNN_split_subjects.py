import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore


def load_subject_data(subjects):
    X_all, y_all = [], []
    for subject in subjects:
        X = np.load(f"{datadir}/feature_vector_full_normalized_{subject}.npy")
        y = np.load(f"{datadir}/y_normalized_{subject}.npy").reshape(-1)
        X_all.append(X)
        y_all.append(y)
    return np.vstack(X_all), np.concatenate(y_all)


def create_cnn_windows(X, y, window_size, stride):
    X_windowed, y_windowed = [], []
    for i in range(window_size, len(X), stride):
        X_windowed.append(X[i - window_size : i])
        y_windowed.append(y[i])
    return np.array(X_windowed), np.array(y_windowed)


datadir = "./DATASET/"
window_size = 100
stride = 10

train_subjects = [1, 2, 3, 5, 6]
val_subjects = [7, 8]
test_subjects = [11]

X_train_raw, y_train_raw = load_subject_data(train_subjects)
X_val_raw, y_val_raw = load_subject_data(val_subjects)
X_test_raw, y_test_raw = load_subject_data(test_subjects)

X_train, y_train = create_cnn_windows(X_train_raw, y_train_raw, window_size, stride)
X_val, y_val = create_cnn_windows(X_val_raw, y_val_raw, window_size, stride)
X_test, y_test = create_cnn_windows(X_test_raw, y_test_raw, window_size, stride)

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


# CNN model
model = Sequential(
    [
        Input(shape=(window_size, 4)),
        Conv1D(64, kernel_size=3, activation="relu"),
        Conv1D(32, kernel_size=3, activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1),
    ]
)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# train
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=128,
    callbacks=[early_stop],
)

# evaluate
y_pred = model.predict(X_test).flatten()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R² Score: {r2:.4f}")


plt.figure(figsize=(10, 4))
plt.plot(y_test[:1000], label="True")
plt.plot(y_pred[:1000], label="Predicted")
plt.title("Predicted vs True Knee Angle (Subject 11)")
plt.xlabel("Sample")
plt.ylabel("Normalized Knee Angle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
