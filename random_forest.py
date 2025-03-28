"""
This script trains two regression models (Random Forest and XGBoost).

1. Loads data from the "./DATASET/" directory.
2. Splits the data into training, validation, and test sets.
3. Tunes hyperparameters for both Random Forest and XGBoost using Optuna.
4. Trains both models using the best hyperparameters found.
5. Evaluates the models on validation and test sets, showing the following metrics:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R2 Score (how well the model fits the data)

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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def objective_rf(trial, X_train, y_train, X_val, y_val):
    """Objective function for tuning Random Forest."""
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)


def objective_xgb(trial, X_train, y_train, X_val, y_val):
    """Objective function for tuning XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)


def hyperparameter_tuning(X_train, X_val, y_train, y_val):
    print("\n Hyperparameter Tuning in Progress...\n")

    # Random Forest Tuning
    print("Tuning Random Forest...")
    rf_study = optuna.create_study(direction="minimize")
    rf_study.optimize(lambda trial: objective_rf(trial, X_train, y_train, X_val, y_val), n_trials=10)
    best_rf_params = rf_study.best_params
    print(f"\nBest Random Forest Params: {best_rf_params}")

    # XGBoost Tuning
    print("\nTuning XGBoost...")
    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val), n_trials=10)
    best_xgb_params = xgb_study.best_params
    print(f"\nBest XGBoost Params: {best_xgb_params}")

    return best_rf_params, best_xgb_params


def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train models with best parameters and evaluate on validation and test sets."""
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

    results = {}
    for model_name, model in tqdm(models.items(), desc="Training Progress", unit="model"):
        print(f"\n Training {model_name}...")

        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        metrics = {
            "MSE": mean_squared_error(y_val, y_pred_val),
            "MAE": mean_absolute_error(y_val, y_pred_val),
            "R2 Score": r2_score(y_val, y_pred_val),
        }

        test_metrics = {
            "MSE": mean_squared_error(y_test, y_pred_test),
            "MAE": mean_absolute_error(y_test, y_pred_test),
            "R2 Score": r2_score(y_test, y_pred_test),
        }

        results[model_name] = {"Validation Metrics": metrics, "Test Metrics": test_metrics}

        print(f"{model_name} - Validation: {metrics}")
        print(f"{model_name} - Test: {test_metrics}")

    print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")
    return results


def main():
    datadir = "./DATASET/"

    X = np.load(datadir + "feature_vector_full_normalized.npy")
    y = np.load(datadir + "y_normalized.npy").reshape(-1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    print(f"Train Set: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation Set: X={X_val.shape}, y={y_val.shape}")
    print(f"Test Set: X={X_test.shape}, y={y_test.shape}")

    train_and_evaluate_models(X_train=X_train,
                               X_val=X_val,
                               X_test=X_test,
                               y_train=y_train,
                               y_val=y_val,
                               y_test=y_test)


if __name__ == '__main__':
    main()
