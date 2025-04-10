import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics.pairwise import cosine_similarity

"""
Determines how similar the frequency content is between two subjects.
Computes cosine similarity of normalized FFT magnitudes for a given feature across all subjects.

(phase capttures alignment in time so interested in magntide)
(mag still needs to be normalized)

"""


def compute_fft_magnitude(signal, fs):
    N = len(signal)
    fft_vals = np.fft.rfft(signal * np.hanning(N))
    # hanning smooths out edges of the singal, reduced discontinuities, and yields cleaner spectrum
    # but it distorts energy magitude. find since normalizing the fft magnatide anyway
    fft_mag = np.abs(fft_vals)  # normlaizing
    fft_mag /= np.sum(fft_mag)
    return fft_mag


def subject_fft_similarity(
    datadir,
    subject_ids,
    feature_index,
    fs=100,
    duration_sec=20,
    title="Cosine Similarity of FFT Magnitudes (Normalized Data)",
):

    feature_map = {
        0: "5k Resistance",
        1: "5k Reactance",
        2: "100k Resistance",
        3: "100k Reactance",
    }

    fft_matrix = []
    valid_subjects = []

    for subject in subject_ids:
        try:
            X = np.load(os.path.join(datadir, f"feature_vector_full_normalized_{subject}.npy"))
            signal = X[feature_index, : fs * duration_sec]
            fft_mag = compute_fft_magnitude(signal, fs)
            fft_matrix.append(fft_mag)
            valid_subjects.append(subject)
        except Exception as e:
            print(f"Skipping subject {subject} due to error: {e}")

    fft_matrix = np.array(fft_matrix)  # shape: (num_subjects, freq_bins)

    similarity_matrix = cosine_similarity(fft_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        xticklabels=valid_subjects,
        yticklabels=valid_subjects,
        annot=True,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Cosine Similarity"},
    )
    plt.title(f"{title}\nFeature: {feature_map[feature_index]}")
    plt.tight_layout()
    plt.show()


def main():
    datadir = "./DATASET/"
    subject_ids = [1, 2, 3, 5, 6, 7, 8, 11]
    feature_index_list = [0, 1, 2, 3]
    for feature_index in feature_index_list:
        subject_fft_similarity(datadir, subject_ids, feature_index)


if __name__ == "__main__":
    main()
