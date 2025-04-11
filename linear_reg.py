"""
This script explores different regression models to fit the given dataset and evaluates their performance.

### Steps Taken:
1. Linear & Polynomial Regression:
   - Implemented both Linear Regression and Polynomial Regression (degree 2).
   - Evaluated models using Mean Squared Error (MSE) and R² score on a validation set.

2. Outlier Removal:
   - Used Cook's Distance to identify and remove high-influence outliers.

3. Feature Selection & Multicollinearity Reduction:
   - Checked for Variance Inflation Factor (VIF) to detect multicollinearity.
   - Found high VIF values, so Principal Component Analysis (PCA) was applied to reduce dimensions and address multicollinearity.

4. Regression Assumption Checks:
   - Scatter plots showed a non-linear relationship between features and target.
   - Histogram of residuals was imbalanced, indicating non-normality.
   - Residuals vs. Predicted plot showed heteroscedasticity, violating regression assumptions.
   - QQ plot confirmed that residuals were not normally distributed.

5. Feature Transformations Attempted:
   - Applied log transformation, interaction terms, and polynomial expansion to linearize relationships.
   - None of these transformations improved model performance.

### Next Steps:
- Since the dataset does not conform to linear regression assumptions, tree-based models will be explored next:
  - Decision Trees
  - XGBoost (Extreme Gradient Boosting)
  - Potentially other ensemble methods like Random Forest.

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
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import os

def unnormalize(y_norm, mean, std):
    return y_norm * std + mean

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred, mean_y, std_y, label="Model", max_points=500, save_path=None):
    # Unnormalize for visualization
    y_pred_unnorm = unnormalize(y_pred, mean_y, std_y)
    y_test_unnorm = unnormalize(y_test, mean_y, std_y)
    
    # Metrics computed on normalized values
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{label} Test Metrics (Normalized):")
    print(f"Test MSE:  {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAE:  {mae:.3f}")
    print(f"Test R²:   {r2:.3f}")

    # Limit number of points for plotting
    slice_test = slice(-max_points, None)
    y_test_tail = y_test_unnorm[slice_test]
    y_pred_tail = y_pred_unnorm[slice_test]

    # Combined plot
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(y_test_tail)), y_test_tail, label="Test (Actual)", color='blue', alpha=0.6)
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

    # Split plot
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

def main():
    subjects = [1, 3, 4, 5, 6, 7, 8, 11]
    base_dir = "./DATASET/"
    results_dir = "Results/linear_regression"
    values_dir = os.path.join(results_dir, "values")
    plots_dir = os.path.join(results_dir, "plots")

    os.makedirs(values_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    all_results = []

    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")
        X = np.load(os.path.join(base_dir, f"ankle_feature_vector_full_normalized_{subject_id}.npy"))
        y = np.load(os.path.join(base_dir, f"ankle_y_normalized_{subject_id}.npy")).reshape(-1)
        y_raw = np.load(os.path.join(base_dir, f"ankle_y_{subject_id}.npy"))

        mean_y = np.mean(y_raw)
        std_y = np.std(y_raw)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        mse, rmse, mae, r2 = plot_predictions(
            y_test=y_test,
            y_pred=y_pred_test,
            mean_y=mean_y,
            std_y=std_y,
            label=f"Linear Regression - Subject {subject_id}",
            save_path=os.path.join(plots_dir, f"subject_{subject_id}")
        )

        all_results.append({
            "Subject": subject_id,
            "Model": "Linear Regression",
            "Test MSE": mse,
            "Test RMSE": rmse,
            "Test MAE": mae,
            "Test R2": r2
        })

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(values_dir, "linear_regression_results.csv"), index=False)
    print("\n All results saved!")

if __name__ == '__main__':
    main()

    # datadir = "DATASET/"
    # X = np.load(os.path.join(datadir, "ankle_feature_vector_full_normalized_1.npy"))
    # y = np.load(os.path.join(datadir, "ankle_y_normalized_1.npy"))
    # y_raw = np.load(os.path.join(datadir, "ankle_y_1.npy"))

    # mean_y = np.mean(y_raw)
    # std_y = np.std(y_raw)

    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    # y_train = y_train.ravel()
    # y_val = y_val.ravel()
    # y_test = y_test.ravel()

    # print(f"Train set shapes: X={X_train.shape}, y={y_train.shape}")
    # print(f"Validation set shapes: X={X_val.shape}, y={y_val.shape}")
    # print(f"Test set shapes: X={X_test.shape}, y={y_test.shape}")

    # check_feature_correlation(X_train, y_train)

    # X_train_cleaned, y_train_cleaned = check_outliers(X_train, y_train)

    # vif_vals_before = check_vif(X_train_cleaned)
    # if any(vif > 10 for vif in vif_vals_before):
    #     print("High VIF detected. Applying PCA...")
    #     X_final = apply_pca(X_train_cleaned)
    #     check_vif(X_final)
    # else:
    #     X_final = X_train_cleaned

    # model = train_and_evaluate_models(X_final, X_val, y_train_cleaned, y_val)

    # plot_and_evaluate_predictions(
    #     model=model,
    #     X_test=X_test,
    #     y_test=y_test,
    #     mean_y=mean_y,
    #     std_y=std_y,
    #     label="Linear Regression",
    #     max_points=100
    # )