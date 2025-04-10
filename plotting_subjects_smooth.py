import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import pandas as pd

"""
Plots the time-series of all four bioimpedance signals and the knee angle
for each subject listed in subject_ids.

Applies smoothing to the 100k Reactance channel (index 3) to reduce noise.
"""


def smooth_100k_reactance(X):
    df = pd.DataFrame(X.T)
    df.iloc[:, 3] = df.iloc[:, 3].rolling(window=15, center=True, min_periods=1).mean()
    return df.to_numpy().T


def plot_normalized_subject_time_series_with_smoothing(
    datadir,
    subject_ids,
    fs,
    start_time_sec,
    duration_sec,
    stride,
    root_dir="SUBJECT_FEATURE_PLOTS",
    subfolder="Normalized_Smoothed100k",
):
    save_dir = os.path.join(root_dir, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    feature_labels = [
        "5k Resistance",
        "5k Reactance",
        "100k Resistance",
        "100k Reactance (Smoothed)",
    ]

    for subject in subject_ids:
        try:
            X = np.load(os.path.join(datadir, f"feature_vector_full_normalized_{subject}.npy")).T
            y = np.load(os.path.join(datadir, f"y_normalized_{subject}.npy")).reshape(-1)

            X = smooth_100k_reactance(X)

            # Downsampling
            fs_down = fs / stride
            X_ds = X[:, ::stride]
            y_ds = y[::stride]
            t_ds = np.arange(X_ds.shape[1]) / fs_down

            start_idx = int(start_time_sec * fs_down)
            end_idx = int((start_time_sec + duration_sec) * fs_down)
            X_ds = X_ds[:, start_idx:end_idx]
            y_ds = y_ds[start_idx:end_idx]
            t_ds = t_ds[start_idx:end_idx]

            fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(
                f"Subject {subject} - Normalized Signals with Smoothed 100k Reactance (Downsampled x{stride})",
                fontsize=16,
            )

            for i in range(4):
                axs[i].plot(t_ds, X_ds[i], label=feature_labels[i])
                axs[i].set_ylabel(feature_labels[i])
                axs[i].legend(loc="upper right")

            axs[4].plot(t_ds, y_ds, label="Normalized Knee Angle", color="orange")
            axs[4].set_ylabel("Knee Angle (norm)")
            axs[4].set_xlabel("Time (s)")
            axs[4].xaxis.set_major_locator(ticker.MultipleLocator(1))
            axs[4].legend(loc="upper right")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = os.path.join(save_dir, f"subject_{subject}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Saved normalized + smoothed plot for subject {subject} to {plot_path}")

        except FileNotFoundError as e:
            print(f"Normalized data for subject {subject} not found: {e}")


def main():
    datadir = "./DATASET/"
    subject_ids = [1, 2, 3, 5, 6, 7, 8, 11]
    fs = 100
    start_time_sec = 100
    duration_sec = 1
    stride = 1

    plot_normalized_subject_time_series_with_smoothing(datadir, subject_ids, fs, start_time_sec, duration_sec, stride)


if __name__ == "__main__":
    main()
