import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt
from LSTM.make_LSTM_windows import generate_lstm_windows

def generate_lstm_windows_magnZ(window_size=50, 
                                features=['knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance'],
                                output="knee_angle_l",
                                subject="11"):
    if os.getlogin() == 'nicholasharris':
        base_path = '/Users/nicholasharris/OneDrive - Georgia Institute of Technology/BioZ Dataset'

    full_path = 'DATASET/'
    
    window_frequency = 2

    data_file = "knee_base_" + subject + ".npy"

    data = np.load(full_path + data_file)

    # LSTM takes data of size (n_windows x window_size x num_features)
    ## knee angle L (y output)
    y_output = data[output]
    y_output = (y_output - np.mean(y_output)) / np.std(y_output)

    # Calculate impedance magnitude
    impedance_5k = np.sqrt(data['knee_bioz_5k_resistance']**2 + data['knee_bioz_5k_reactance']**2)
    impedance_100k = np.sqrt(data['knee_bioz_100k_resistance']**2 + data['knee_bioz_100k_reactance']**2)

    x_input = np.stack([impedance_5k, impedance_100k], axis=2)
    x_input = (x_input - np.mean(x_input, axis=(0,1))) / np.std(x_input, axis=(0,1))
    
    x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size=0.25, random_state=42)

    x_train_windows = []
    y_train_windows = []
    for i in range(np.shape(x_train)[0]):
        for j in range(np.shape(x_train)[1] - window_size):
            # if j % window_frequency:
            #     continue
            x_train_windows.append(x_train[i, j:j+window_size])
            y_train_windows.append(y_train[i, j+window_size])

    x_train_windows = np.stack(x_train_windows, axis=0)
    y_train_windows = np.stack(y_train_windows, axis=0)
    y_train_windows = np.expand_dims(y_train_windows, axis=-1)

    x_test_windows = []
    y_test_windows = []
    for i in range(np.shape(x_test)[0]):
        for j in range(np.shape(x_test)[1] - window_size):
            # if j % window_frequency:
            #     continue
            x_test_windows.append(x_test[i, j:j+window_size])
            y_test_windows.append(y_test[i, j+window_size])

    x_test_windows = np.stack(x_test_windows, axis=0)
    y_test_windows = np.stack(y_test_windows, axis=0)
    y_test_windows = np.expand_dims(y_test_windows, axis=-1)

    return x_train_windows, x_test_windows, y_train_windows, y_test_windows

def main():
    start_time = time.time()

    features = ['knee_bioz_5k_resistance', 'knee_bioz_5k_reactance', 'knee_bioz_100k_resistance', 'knee_bioz_100k_reactance']
    output_feature = "knee_angle_l"
    subject = "11"

    print("feature vector")
    print(features)
    print("output feature")
    print(output_feature)
    print("subject")
    print(subject)

    train_x, test_x, train_y, test_y = generate_lstm_windows_magnZ(WINDOW_SIZE, features, output_feature, subject)
    # train_x, test_x, train_y, test_y = generate_lstm_windows(WINDOW_SIZE, features, output_feature, subject)


    # Debugging: Print the shape of train_x
    print("Shape of train_x before reshaping:", train_x.shape)

    # Reshape train_x to 2D array
    n_windows, window_size, num_features = train_x.shape
    train_x_reshaped = train_x.reshape(n_windows, window_size * num_features)

    # Debugging: Print the shape of train_x after reshaping
    print("Shape of train_x after reshaping:", train_x_reshaped.shape)

    # Create column names for the reshaped DataFrame
    reshaped_features = [f"impedance_5k_t{i}" for i in range(window_size)] + [f"impedance_100k_t{i}" for i in range(window_size)]

    # Combine features and output into a single DataFrame
    train_data = pd.DataFrame(train_x_reshaped, columns=reshaped_features)
    train_data[output_feature] = train_y

    # Fit the VAR model
    model = VAR(train_data)
    fitted_model = model.fit(maxlags=10)

    model_time = time.time()

    # Print the summary of the model
    print(fitted_model.summary())

    # Window-wise forecasting
    n_forecast = 10  # Number of steps to forecast in each window
    forecast_windows = []

    for i in range(0, len(test_y) - n_forecast + 1, n_forecast):
        forecast = fitted_model.forecast(train_data.values[-fitted_model.k_ar:], steps=n_forecast)
        forecast_windows.append(forecast)

    # Combine forecasts
    forecast_combined = np.vstack(forecast_windows)

    forecast_time = time.time()

    # Compare forecast with actual test data
    actual = np.array(test_y[:len(forecast_combined)])
    mse = mean_squared_error(actual, forecast_combined[:, -1])
    print(f"Mean Squared Error: {mse}")

    # Calculate the index for the last half of the signal
    half_index = len(train_y) // 2

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the last half of the actual train signal
    plt.plot(np.arange(half_index, len(train_y)), train_y[half_index:], label='Actual Train')

    # Plot the last half of the actual test signal
    plt.plot(np.arange(len(train_y), len(train_y) + len(forecast_combined)), actual, label='Actual Test')

    # Plot the forecast for the last half of the signal
    plt.plot(np.arange(len(train_y), len(train_y) + len(forecast_combined)), forecast_combined[:, -1], label='Forecast')

    plt.legend()
    plt.title('VAR Model Window-wise Forecast vs Actual (Last Half)')
    plt.show()

    # Print the elapsed time
    elapsed_time = model_time - start_time
    elapsed_min = elapsed_time/60
    print(f"Time taken to generate the model: {elapsed_min:.2f} minutes")
    elapsed_time = forecast_time - start_time
    elapsed_min = elapsed_time/60
    print(f"Time taken to forecast: {elapsed_min:.2f} minutes")

if __name__ == "__main__":
    WINDOW_SIZE = 500
    main()


## EXAMPLE code

# ### Step 1: Generate Fake Signal
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.api import VAR
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # Generate synthetic data
# np.random.seed(42)
# n_obs = 200
# time = np.arange(n_obs)
# signal1 = np.sin(0.1 * time) + np.random.normal(scale=0.1, size=n_obs)
# signal2 = np.cos(0.1 * time) + np.random.normal(scale=0.1, size=n_obs)
# signal3 = np.sin(0.1 * time + np.pi/4) + np.random.normal(scale=0.1, size=n_obs)

# data = pd.DataFrame({'signal1': signal1, 'signal2': signal2, 'signal3': signal3})

# # Plot the synthetic data
# plt.figure(figsize=(10, 6))
# plt.plot(data['signal1'], label='Signal 1')
# plt.plot(data['signal2'], label='Signal 2')
# plt.plot(data['signal3'], label='Signal 3')
# plt.legend()
# plt.title('Synthetic Signals')
# plt.show()

# ### Step 2: Create Windows for Training and Testing
# def create_windows(data, window_size=10, window_frequency=1):
#     x_input = []
#     y_output = []

#     for i in range(len(data) - window_size):
#         if i % window_frequency == 0:
#             x_input.append(data.iloc[i:i+window_size].values)
#             y_output.append(data.iloc[i+window_size].values)

#     x_input = np.array(x_input)
#     y_output = np.array(y_output)

#     return x_input, y_output

# window_size = 10
# window_frequency = 1

# x_input, y_output = create_windows(data, window_size, window_frequency)

# # Split into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size=0.25, random_state=42)

# # Reshape train_x to 2D array
# n_windows, window_size, num_features = x_train.shape
# x_train_reshaped = x_train.reshape(n_windows, window_size * num_features)

# # Create column names for the reshaped DataFrame
# reshaped_features = [f"signal{i}_t{j}" for i in range(1, num_features+1) for j in range(window_size)]

# # Combine features and output into a single DataFrame
# train_data = pd.DataFrame(x_train_reshaped, columns=reshaped_features)
# train_data[['signal1', 'signal2', 'signal3']] = y_train

# ### Step 3: Fit VAR Model
# # Fit the VAR model
# model = VAR(train_data)
# fitted_model = model.fit(maxlags=2)

# # Print the summary of the model
# print(fitted_model.summary())

# # Forecasting the entire test signal
# forecast = fitted_model.forecast(train_data.values[-fitted_model.k_ar:], steps=len(y_test))

# # Combine forecast with actual data for plotting
# forecast_index = np.arange(len(train_data), len(train_data) + len(y_test))
# forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=reshaped_features + ['signal1', 'signal2', 'signal3'])

# # Plot the forecast
# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(len(data)), data['signal1'], label='Actual Signal 1')
# plt.plot(forecast_index, forecast_df['signal1'], label='Forecast Signal 1')
# plt.plot(np.arange(len(data)), data['signal2'], label='Actual Signal 2')
# plt.plot(forecast_index, forecast_df['signal2'], label='Forecast Signal 2')
# plt.plot(np.arange(len(data)), data['signal3'], label='Actual Signal 3')
# plt.plot(forecast_index, forecast_df['signal3'], label='Forecast Signal 3')
# plt.legend()
# plt.title('VAR Model Forecast')
# plt.show()

# # Compare forecast with actual test data
# actual = np.array(y_test)
# mse = mean_squared_error(actual, forecast[:, -3:])
# print(f"Mean Squared Error: {mse}")

# ### Step 5: Impulse Response Functions (IRFs)
# # Impulse Response Functions
# irf = fitted_model.irf(10)
# irf.plot()
# plt.title('Impulse Response Functions')
# plt.show()

# ### Step 6: Forecast Error Variance Decomposition (FEVD)
# # Forecast Error Variance Decomposition
# fevd = fitted_model.fevd(10)
# fevd.plot()
# plt.title('Forecast Error Variance Decomposition')
# plt.show()
