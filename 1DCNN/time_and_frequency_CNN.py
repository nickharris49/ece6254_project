# CNN with time domain features and some frequency domain features
# has 5k reactance dropped and 100k reactacne smoothened
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore


X_raw_full = np.load("./DATASET/feature_vector_full_normalized_2.npy")  # shape: (N, 4)
y_raw = np.load("./DATASET/y_normalized_2.npy").reshape(-1)

# Drop 5k reactance (index 1)
X_reduced = np.delete(X_raw_full, 1, axis=1)  # Now shape: (N, 3)

# Smooth 100k reactance (now at index 2 after dropping)
df = pd.DataFrame(X_reduced)
df.iloc[:, 2] = df.iloc[:, 2].rolling(window=15, center=True, min_periods=1).mean()
X_smoothed = df.to_numpy()  # Still (N, 3)


# === Create Combined Time + Frequency Windows ===
def create_combined_time_frequency_windows(X, y, window_size, stride, fs=100):
    X_combined, y_combined = [], []
    for i in range(window_size, len(X), stride):
        window = X[i - window_size : i]
        freq_features = []
        for f in range(window.shape[1]):
            signal = window[:, f]
            fft_mag = np.abs(np.fft.rfft(signal * np.hanning(window_size)))
            fft_mag /= np.sum(fft_mag) + 1e-8
            freq_features.append(fft_mag[:10])  # top 10 bins
        freq_features = np.concatenate(freq_features)
        combined = np.concatenate([window.flatten(), freq_features])
        X_combined.append(combined)
        y_combined.append(y[i])
    return np.array(X_combined), np.array(y_combined)


window_size = 50
stride = 2
X_comb, y_comb = create_combined_time_frequency_windows(X_smoothed, y_raw, window_size, stride)


X_train, X_temp, y_train, y_temp = train_test_split(X_comb, y_comb, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

# model-----------------
model = Sequential(
    [
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1),
    ]
)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=128,
    verbose=1,
)

# Evaluate
y_pred = model.predict(X_test).flatten()
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred):.4f}")

plt.figure(figsize=(10, 4))
plt.plot(y_test[:1000], label="True")
plt.plot(y_pred[:1000], label="Predicted")
plt.title("Predicted vs True Knee Angle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
