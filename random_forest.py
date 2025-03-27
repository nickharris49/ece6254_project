import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import time
from tqdm import tqdm
import optuna

def check_outliers(X_train, y_train):
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    outliers = np.where(cooks_d > 4 / len(X_train))[0]

    print(f"Found {len(outliers)} potential outliers.")
    if len(outliers) > 0:
        X_train_cleaned = np.delete(X_train, outliers, axis=0)
        y_train_cleaned = np.delete(y_train, outliers, axis=0)
        return X_train_cleaned, y_train_cleaned
    else:
        return X_train, y_train

def objective_rf(trial, X_train, y_train, X_val, y_val):
    """ Objective function for tuning Random Forest """
    n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    """ Objective function for tuning XGBoost """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    
    model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)

def hyperparameter_tuning(X_train, X_val, y_train, y_val):
    print("\n Hyperparameter Tuning in Progress...\n")

    # Random Forest Tuning
    print("Tuning Random Forest...")
    rf_study = optuna.create_study(direction="minimize")
    rf_study.optimize(lambda trial: objective_rf(trial, X_train, y_train, X_val, y_val), n_trials=10, show_progress_bar=True)
    best_rf_params = rf_study.best_params
    print(f"\nBest Random Forest Params: {best_rf_params}")

    # XGBoost Tuning
    print("\nTuning XGBoost...")
    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val), n_trials=10, show_progress_bar=True)
    best_xgb_params = xgb_study.best_params
    print(f"\nBest XGBoost Params: {best_xgb_params}")

    return best_rf_params, best_xgb_params

def train_and_evaluate_models(X_train, X_val, y_train, y_val, best_rf_params, best_xgb_params):
    start_time = time.time()
    print("\nTraining Models...")

    results = {}

    models = {
        "Random Forest": RandomForestRegressor(**best_rf_params, n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(**best_xgb_params, objective="reg:squarederror", n_estimators=100, random_state=42, n_jobs=-1)
    }

    # Train models with tqdm progress bar
    for model_name, model in tqdm(models.items(), desc="Training Progress", unit="model"):
        print(f"\n🔹 Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Compute MSE
        mse = mean_squared_error(y_val, y_pred)
        results[model_name] = mse

        print(f"{model_name} MSE: {mse:.4f}")

    print(f"\n Total training time: {time.time() - start_time:.2f} seconds")
    return results

# Main Function
def main():
    datadir = "DATASET/"
    
    X_path = datadir + "feature_vector_full.npy"
    y_path = datadir + "y.npy"

    X = np.load(X_path)
    y = np.load(y_path)

    # Split data into train/validation sets (use test set only after final selection)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    # Convert y_train, y_val, and y_test to 1D arrays
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()

    print(f"Train set shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test set shapes: X={X_test.shape}, y={y_test.shape}")

    # X_train_cleaned, y_train_cleaned = check_outliers(X_train, y_train)
    best_rf_params, best_xgb_params = hyperparameter_tuning(X_train, X_val, y_train, y_val)

    # Train and Evaluate Models using validation set
    train_and_evaluate_models(X_train, X_val, y_train, y_val,best_rf_params, best_xgb_params)

if __name__ == '__main__':
    main()   