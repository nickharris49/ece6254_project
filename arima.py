import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_subject_series_arima(subject_id, base_dir="./DATASET/"):
    y_path_raw = os.path.join(base_dir, f"ankle_y_{subject_id}.npy")
    y_path_norm = os.path.join(base_dir, f"ankle_y_normalized_{subject_id}.npy")

    y_raw = np.load(y_path_raw).reshape(-1)
    y_norm = np.load(y_path_norm).reshape(-1)

    mean_y = np.mean(y_raw)
    std_y = np.std(y_raw)

    return {
        "subject_id": subject_id,
        "y_raw": y_raw,
        "y_norm": y_norm,
        "mean": mean_y,
        "std": std_y
    }

def plot_forecast(subject_id, model_name, x_train_tail, y_train_tail, x_forecast, y_test_unnorm, forecast_unnorm, results_dir):
    plt.figure(figsize=(12, 5))
    plt.plot(x_train_tail, y_train_tail, label="Train (tail)", color='gray')
    plt.plot(x_forecast, y_test_unnorm, label="Test (actual)", color='blue', alpha=0.6)
    plt.plot(x_forecast, forecast_unnorm, label=f"{model_name} Forecast", color='orange', linewidth=2)
    plt.title(f"{model_name} Forecast - Subject {subject_id}")
    plt.xlabel("Time")
    plt.ylabel("Knee Angle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", f"{model_name.lower()}_subject_{subject_id}_combined.png"))
    plt.close()

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(x_forecast, y_test_unnorm, label="Test (Actual)", color='blue')
    axs[0].set_title("Ground Truth (Test Set)")
    axs[0].set_ylabel("Knee Angle")
    axs[0].grid(True)

    axs[1].plot(x_forecast, forecast_unnorm, label=f"{model_name} Forecast", color='orange')
    axs[1].set_title(f"{model_name} Forecast")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Knee Angle")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", f"{model_name.lower()}_subject_{subject_id}_split.png"))
    plt.close()

def run_arima_forecast(subject_id, forecast_horizon=300, arima_order=(3, 0, 2), context_points=200, results_dir="results/arima"):
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "values"), exist_ok=True)

    data = load_subject_series_arima(subject_id)
    y_norm = data["y_norm"]
    y_raw = data["y_raw"]
    mean, std = data["mean"], data["std"]

    n = len(y_norm)
    train_end = int(n * 0.85)

    y_train = y_norm[:train_end]
    y_test = y_norm[train_end:train_end + forecast_horizon]
    y_test_unnorm = y_raw[train_end:train_end + forecast_horizon]

    model = ARIMA(y_train, order=arima_order)
    model_fit = model.fit()
    forecast_norm = model_fit.forecast(steps=forecast_horizon)
    forecast_unnorm = forecast_norm * std + mean

    x_train_tail = np.arange(train_end - context_points, train_end)
    y_train_tail = y_raw[train_end - context_points:train_end]
    x_forecast = np.arange(train_end, train_end + forecast_horizon)

    plot_forecast(subject_id, "ARIMA", x_train_tail, y_train_tail, x_forecast, y_test_unnorm, forecast_unnorm, results_dir)

    mse = mean_squared_error(y_test, forecast_norm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, forecast_norm)
    r2 = r2_score(y_test, forecast_norm)

    print(f"\nARIMA Metrics for Subject {subject_id}")
    print(f"Test MSE:  {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAE:  {mae:.3f}")
    print(f"Test R2:  {r2:.3f}")

    return {"Subject": subject_id, "Model": "ARIMA", "Test MSE": mse, "Test RMSE": rmse, "Test MAE": mae, "Test R2": r2}

def run_sarima_rolling_forecast(subject_id, forecast_horizon=300, context_points=200, m=25, results_dir="results/arima"):
    data = load_subject_series_arima(subject_id)
    y_raw = data["y_raw"][:20000]
    y_norm = data["y_norm"][:20000]
    mean, std = data["mean"], data["std"]

    n = len(y_norm)
    train_end = int(n * 0.85)

    y_train = y_norm[:train_end]
    y_test = y_norm[train_end:train_end + forecast_horizon]
    y_test_unnorm = y_raw[train_end:train_end + forecast_horizon]

    print(f"Fitting SARIMA(2,0,2)(1,0,0)[{m}] for Subject {subject_id}...")
    base_model = SARIMAX(
        y_train,
        order=(2, 0, 2),
        seasonal_order=(1, 0, 0, m),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = base_model.fit(disp=False)

    predictions_norm = []
    history = results

    for t in range(forecast_horizon):
        forecast = history.forecast(steps=1)[0]
        predictions_norm.append(forecast)
        history = history.append(endog=[y_test[t]], refit=False)

    forecast_unnorm = np.array(predictions_norm) * std + mean

    x_forecast = np.arange(train_end, train_end + forecast_horizon)
    x_train_tail = np.arange(train_end - context_points, train_end)
    y_train_tail = y_raw[train_end - context_points:train_end]

    plot_forecast(subject_id, "SARIMA", x_train_tail, y_train_tail, x_forecast, y_test_unnorm, forecast_unnorm, results_dir)

    mse = mean_squared_error(y_test, predictions_norm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions_norm)
    r2 = r2_score(y_test, predictions_norm)

    print(f"\nSARIMA Metrics for Subject {subject_id}")
    print(f"Test MSE:  {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAE:  {mae:.3f}")
    print(f"Test R2:  {r2:.3f}")
    
    return {"Subject": subject_id, "Model": "SARIMA", "Test MSE": mse, "Test RMSE": rmse, "Test MAE": mae, "Test R2": r2}

def main():
    subjects = [1, 3, 4, 5, 6, 7, 8, 11]
    results_dir = "results/arima"
    all_results = []

    for subject_id in subjects:
        print(f"\nRunning ARIMA for Subject {subject_id}...")
        result_arima = run_arima_forecast(subject_id=subject_id, forecast_horizon=300, arima_order=(3, 0, 2), context_points=200, results_dir=results_dir)
        result_sarima = run_sarima_rolling_forecast(subject_id=subject_id, forecast_horizon=300, context_points=200, m=25, results_dir=results_dir)
        all_results.extend([result_arima, result_sarima])

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(results_dir, "values", "arima_results.csv"), index=False)
    print("\n ARIMA & SARIMA results saved for all subjects.")

if __name__ == '__main__':
    main()
