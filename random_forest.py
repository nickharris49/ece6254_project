"""
This script trains two regression models (Random Forest and XGBoost).

1. Loads data from the "./DATASET/" directory.
2. Splits the data into training, validation, and test sets.
3. Tunes hyperparameters for both Random Forest and XGBoost using Optuna.
4. Trains both models using the best hyperparameters found.
5. Evaluates the models on validation and test sets, showing the following metrics:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R2 Score

### Dependencies:
- `numpy`
- `optuna`
- `scikit-learn`
- `tqdm`
- `xgboost`
"""

import numpy as np
import time
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import os

def unnormalize(y_norm, mean, std):
    return y_norm * std + mean

def plot_predictions(y_test, y_pred, mean_y, std_y, label="Model", max_points=100, save_path=None):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{label} Test Metrics:")
    print(f"Test MSE:  {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAE:  {mae:.3f}")
    print(f"Test R²:   {r2:.3f}")

    y_pred_unnorm = unnormalize(y_pred, mean_y, std_y)
    y_test_unnorm = unnormalize(y_test, mean_y, std_y)
    slice_test = slice(-max_points, None)
    y_test_tail = y_test_unnorm[slice_test]
    y_pred_tail = y_pred_unnorm[slice_test]

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(y_test_tail)), y_test_tail, label="Test (actual)", color='blue', alpha=0.6)
    plt.plot(range(len(y_pred_tail)), y_pred_tail, label=f"{label} Prediction", color='orange', linewidth=2)
    plt.title(f"{label} - Prediction vs Actual")
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "_combined.png")
    plt.close()

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(range(len(y_test_tail)), y_test_tail, label="Test (Actual)", color='blue')
    axs[0].set_title("Ground Truth (Test Set)")
    axs[0].set_ylabel("Target Value")
    axs[0].grid(True)

    axs[1].plot(range(len(y_pred_tail)), y_pred_tail, label=f"{label} Prediction", color='orange')
    axs[1].set_title(f"{label} Prediction")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Target Value")
    axs[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "_split.png")
    plt.close()

    return mse, rmse, mae, r2

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, mean_y, std_y, subject_id, results_dir):
    start_time = time.time()
    print("\nTraining Models...")

    best_rf_params = {
        "n_estimators": 150,
        "max_depth": 16,
        "min_samples_split": 3,
        "min_samples_leaf": 4,
        "random_state": 42,
    }
    best_xgb_params = {
        "n_estimators": 50,
        "max_depth": 14,
        "learning_rate": 0.13314778470142807,
        "subsample": 0.6349139707904927,
        "colsample_bytree": 0.9770790049352496,
        "min_child_weight": 7,
        "random_state": 42,
    }

    models = {
        "Random Forest": RandomForestRegressor(**best_rf_params),
        "XGBoost": XGBRegressor(objective="reg:squarederror", **best_xgb_params),
    }

    results = []
    for model_name, model in tqdm(models.items(), desc="Training Progress", unit="model"):
        print(f"\n Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        mse, rmse, mae, r2 = plot_predictions(
            y_test, y_pred_test, mean_y, std_y,
            label=f"{model_name} - Subject {subject_id}",
            save_path=os.path.join(results_dir, "plots", f"{model_name.lower().replace(' ', '_')}_subject_{subject_id}")
        )

        results.append({
            "Subject": subject_id,
            "Model": model_name,
            "Test MSE": mse,
            "Test RMSE": rmse,
            "Test MAE": mae,
            "Test R2": r2
        })

        print(f"{model_name} - Total training time: {time.time() - start_time:.2f} seconds")

    return results

def main():
    # subjects = [1, 3, 4, 5, 6, 7, 8, 11]
    subjects = [1,2,3,5,6,7,8,11]
    datadir = "./DATASET/"
    results_dir = "results/tree_models"
    os.makedirs(os.path.join(results_dir, "values"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    all_results = []
    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")
        # X = np.load(os.path.join(datadir, f"ankle_feature_vector_full_normalized_{subject_id}.npy"))
        # y = np.load(os.path.join(datadir, f"ankle_y_normalized_{subject_id}.npy")).reshape(-1)
        # y_raw = np.load(os.path.join(datadir, f"ankle_y_{subject_id}.npy"))
        X = np.load(os.path.join(datadir, f"feature_vector_100k_normalized_{subject_id}.npy")).reshape(-1,1)
        y = np.load(os.path.join(datadir, f"feature_vector_5k_normalized_{subject_id}.npy")).reshape(-1)
        y_raw = np.load(os.path.join(datadir, f"feature_vector_5k_{subject_id}.npy"))
        mean_y = np.mean(y_raw)
        std_y = np.std(y_raw)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

        results = train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, mean_y, std_y, subject_id, results_dir)
        all_results.extend(results)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(results_dir, "values", "tree_model_results_5k_100k.csv"), index=False)
    print("\n All results saved!")

if __name__ == '__main__':
    main()
