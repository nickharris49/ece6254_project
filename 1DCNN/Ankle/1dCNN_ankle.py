# single subject
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input, BatchNormalization, Dropout, LeakyReLU  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from IPython.display import Image, display


# -------------------------------
# Create windowed sequences with stride
def create_cnn_windows(X, y, window_size, stride):
    X_windowed, y_windowed = [], []
    for i in range(window_size, len(X), stride):
        X_windowed.append(X[i - window_size : i])  # shape: (window_size, 4)
        y_windowed.append(y[i])  # predict angle at time i
    return np.array(X_windowed), np.array(y_windowed)


# --------------------------------------------
# Load and window data and split into sets
datadir = "/content/ece6254_project/DATASET/"
X_raw = np.load(datadir + "ankle_feature_vector_full_normalized_3.npy")
if X_raw.shape[0] == 4:
    X_raw = X_raw.T  # ensure shape is (N, 4)

y_raw = np.load(datadir + "ankle_y_normalized_3.npy").reshape(-1)

# windowing with stride
window_size = 50
stride = 2
X_windowed, y_windowed = create_cnn_windows(X_raw, y_raw, window_size, stride)


X_train, X_temp, y_train, y_temp = train_test_split(X_windowed, y_windowed, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")

# ------------------------------------
# CNN model
"""
model = Sequential(
    [
        Input(shape=(window_size, 4)),  # shape: (100, 4)
        Conv1D(64, kernel_size=3, activation="relu"),
        Conv1D(32, kernel_size=3, activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1),  # Output: predicted knee angle
    ]
)
"""
"""
model = Sequential(
    [
        Input(shape=(window_size, 4)),
        Conv1D(128, kernel_size=5, activation="relu"),
        Conv1D(64, kernel_size=5, activation="relu"),
        Conv1D(32, kernel_size=3, activation="relu"),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1),
    ]
)
"""
# one I have been using for single subject.....

model = Sequential(
    [
        Input(shape=(window_size, 4)),
        Conv1D(128, 5, activation="relu"),
        BatchNormalization(),
        Conv1D(64, 5, activation="relu"),
        BatchNormalization(),
        Conv1D(32, 3, activation="relu"),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1),
    ]
)
"""
# .......
model = Sequential(
    [
        Input(shape=(window_size, 4)),
        Conv1D(256, 7),
        BatchNormalization(),
        LeakyReLU(),
        # Dropout(0.3),
        Conv1D(128, 5),
        BatchNormalization(),
        LeakyReLU(),
        # Dropout(0.3),
        Conv1D(64, 3),
        BatchNormalization(),
        LeakyReLU(),
        Flatten(),
        Dense(128),
        LeakyReLU(),
        # Dropout(0.3),
        Dense(1),
    ]
)
"""

"""
model = Sequential(
    [
        Input(shape=(window_size, 4)),
        Conv1D(256, 5, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(128, 5, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(64, 3, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1),
    ]
)
"""

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# -----------------------------
# train model
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=75,
    batch_size=128,
    callbacks=[early_stop],
)

# -------------------------------
# Evaluate
y_pred = model.predict(X_test).flatten()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"window size: {window_size} and stride: {stride}")
print(f"\nTest MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R² Score: {r2:.4f}")


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
plt.legend()
plt.title("Training vs Validation Loss")
plt.savefig("1dCNN2.png", dpi=300)
display(Image("1dCNN1.png"))
display(Image("1dCNN2.png"))
