import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------- Load Subject Data ----------
def load_subject_series_arima(subject_id, base_dir="./DATASET/"):
    y_path_raw = os.path.join(base_dir, f"y_{subject_id}.npy")
    y_path_norm = os.path.join(base_dir, f"y_normalized_{subject_id}.npy")

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

# ---------- ARIMA Forecasting for Subject ----------
def run_arima_forecast(subject_id=11, forecast_horizon=300, arima_order=(3, 0, 2), context_points=200):
    data = load_subject_series_arima(subject_id)
    y_norm = data["y_norm"]
    y_raw = data["y_raw"]
    mean, std = data["mean"], data["std"]

    n = len(y_norm)
    train_end = int(n * 0.85)

    y_train = y_norm[:train_end]
    y_test = y_norm[train_end:train_end + forecast_horizon]
    y_test_unnorm = y_raw[train_end:train_end + forecast_horizon]

    # Fit ARIMA model
    model = ARIMA(y_train, order=arima_order)
    model_fit = model.fit()

    print(model_fit.summary())  # Optional: Inspect AIC/BIC

    # Forecast
    forecast_norm = model_fit.forecast(steps=forecast_horizon)
    forecast_unnorm = forecast_norm * std + mean

    # Plot: context from training + forecast
    x_train_tail = np.arange(train_end - context_points, train_end)
    y_train_tail = y_raw[train_end - context_points:train_end]
    x_forecast = np.arange(train_end, train_end + forecast_horizon)

    plt.figure(figsize=(12, 5))
    plt.plot(x_train_tail, y_train_tail, label="Train (tail)", color='gray')
    plt.plot(x_forecast, y_test_unnorm, label="Test (actual)", color='blue', alpha=0.6)
    plt.plot(x_forecast, forecast_unnorm, label="ARIMA Forecast", color='orange', linewidth=2)
    plt.title(f"ARIMA({arima_order[0]},{arima_order[1]},{arima_order[2]}) Forecast - Subject {subject_id}")
    plt.xlabel("Time")
    plt.ylabel("Knee Angle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Metrics
    mse = mean_squared_error(y_test_unnorm, forecast_unnorm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unnorm, forecast_unnorm)

    print(f"\n Metrics for Subject {subject_id}")
    print(f"Test MSE:  {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAE:  {mae:.3f}")

# ---------- Run ----------
# run_arima_forecast(subject_id=11)

def run_sarima_rolling_forecast(subject_id=11, forecast_horizon=300, context_points=200, m=25):
    # Load and trim data
    data = load_subject_series_arima(subject_id)
    y_raw = data["y_raw"][:20000]
    y_norm = data["y_norm"][:20000]
    mean, std = data["mean"], data["std"]

    n = len(y_norm)
    train_end = int(n * 0.85)

    y_train = y_norm[:train_end]
    y_test = y_norm[train_end:train_end + forecast_horizon]
    y_test_unnorm = y_raw[train_end:train_end + forecast_horizon]

    # Fit once
    print("Fitting SARIMA(2,0,2)(1,0,0)[25] once...")
    base_model = SARIMAX(
        y_train,
        order=(2, 0, 2),
        seasonal_order=(1, 0, 0, m),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = base_model.fit(disp=False)

    # Rolling forecast
    predictions_norm = []
    history = results

    for t in range(forecast_horizon):
        forecast = history.forecast(steps=1)[0]
        predictions_norm.append(forecast)
        history = history.append(endog=[y_test[t]], refit=False)  # update without re-fitting

    # forecast_norm = results.forecast(steps=forecast_horizon)
    # Unnormalize forecast
    forecast_unnorm = np.array(predictions_norm) * std + mean

    # X-axis setup
    x_forecast = np.arange(train_end, train_end + forecast_horizon)
    x_train_tail = np.arange(train_end - context_points, train_end)
    y_train_tail = y_raw[train_end - context_points:train_end]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(x_train_tail, y_train_tail, label="Train (tail)", color='gray')
    plt.plot(x_forecast, y_test_unnorm, label="Test (actual)", color='blue', alpha=0.6)
    plt.plot(x_forecast, forecast_unnorm, label="SARIMA Forecast", color='orange', linewidth=2)
    plt.title(f"Rolling SARIMA Forecast - Subject {subject_id}")
    plt.xlabel("Time")
    plt.ylabel("Knee Angle")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs[0].plot(x_forecast, y_test_unnorm, label="Test (Actual)", color='blue')
    axs[0].set_title("Ground Truth (Test Set)")
    axs[0].set_ylabel("Knee Angle")
    axs[0].grid(True)

    axs[1].plot(x_forecast, forecast_unnorm, label="SARIMA Forecast", color='orange')
    axs[1].set_title("SARIMA Forecast")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Knee Angle")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # Metrics
    mse = mean_squared_error(y_test_unnorm, forecast_unnorm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unnorm, forecast_unnorm)

    print(f"\n Rolling Forecast Metrics for Subject {subject_id}")
    print(f"Test MSE:  {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test MAE:  {mae:.3f}")

# Run
run_sarima_rolling_forecast(subject_id=1, forecast_horizon=300, m=25)