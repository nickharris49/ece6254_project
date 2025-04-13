import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time


def main():
    datadir = "./DATASET/"
    X = np.load(datadir + "feature_vector_full_normalized.npy")  # shape (1338000, 4)
    y = np.load(datadir + "y_normalized.npy").reshape(-1)  # shape (1338000,)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, random_state=42
    )

    print(
        f"Train Set: X={X_train.shape}, y={y_train.shape}"
    )  # X=(1070400, 4), y=(1070400,)
    print(
        f"Validation Set: X={X_val.shape}, y={y_val.shape}"
    )  # X=(160560, 4), y=(160560,)
    print(f"Test Set: X={X_test.shape}, y={y_test.shape}")  # X=(107040, 4), y=(107040,)

    # k_vec = np.arange(1, 10)
    k_vec = np.array([10, 15, 25, 30, 50, 60, 80, 100])
    mse_grid = np.zeros((k_vec.shape[0]))
    print("\nTuning k using subsampled validation set...")
    time_array = []
    for i, k in enumerate(k_vec):
        start_time = time.time()
        model = KNeighborsRegressor(n_neighbors=k, weights="distance")
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)

        print(f"k={k} → Validation MSE: {mse:.2f}")
        end_time = time.time()
        time_array.append(end_time - start_time)
        mse_grid[i] = mse

    # print(f"time for each knn {time_array}")
    best_k_idx = np.argmin(mse_grid)
    best_k = k_vec[best_k_idx]
    print(f"Validation MSE {mse_grid}")
    print(f"\n Best k: {best_k} with Validation MSE: {mse_grid[best_k_idx]:.2f}")
    # Final evaluation on full test set using the same best_k but still on small train for now
    final_model = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
    final_model.fit(X_train, y_train)
    y_test_prediction = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Validation RMSE: {rmse:.2f}°")

    print("\n--- Final Evaluation on Partial Test Set ---")
    print(f"Test MSE: {mean_squared_error(y_test, y_test_prediction):.2f}")
    print(f"Test R²:  {r2_score(y_test, y_test_prediction):.4f}")


if __name__ == "__main__":
    main()
