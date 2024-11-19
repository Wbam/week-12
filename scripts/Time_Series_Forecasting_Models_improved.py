import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Function to load pre-processed data
def load_preprocessed_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, index_col="Date", parse_dates=True)

# Function to clean data
def clean_data(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors='coerce')  # Convert non-numeric to NaN
    series = series.fillna(series.mean())  # Replace NaNs with the mean value
    return series

# Function to split data into train and test
def split_data(series: pd.Series, test_size: float = 0.2):
    split_idx = int(len(series) * (1 - test_size))
    train_data = series[:split_idx]
    test_data = series[split_idx:]
    return train_data, test_data

# Function to train ARIMA model
def train_arima(train_data: pd.Series, order=(5, 1, 0)):
    model = ARIMA(train_data, order=order)
    return model.fit()

# Function to train SARIMA model
def train_sarima(train_data: pd.Series, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    return model.fit()

# Function to train LSTM model
def prepare_lstm_data(series: pd.Series, time_steps: int = 60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))  # Scaling data
    
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be [samples, time_steps, features] for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

def create_lstm_model(time_steps: int = 60):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(time_steps, 1)))
    model.add(Dense(units=1))  # Output layer with 1 neuron for prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(train_data: pd.Series, time_steps: int = 60, epochs: int = 10, batch_size: int = 32):
    X_train, y_train, scaler = prepare_lstm_data(train_data, time_steps)
    
    model = create_lstm_model(time_steps)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model, scaler

# Function to forecast with LSTM
def forecast_lstm(model, X_input, n_steps):
    forecast = []
    current_input = X_input[-1].reshape(1, -1, 1)  # Take the last available input
    
    for _ in range(n_steps):
        predicted_value = model.predict(current_input)
        forecast.append(predicted_value[0, 0])
        
        # Update the input for the next prediction
        current_input = np.append(current_input[:, 1:, :], predicted_value.reshape(1, 1, 1), axis=1)
    
    return forecast

# Function to evaluate model performance
def evaluate_model(model, test_data: pd.Series, forecast):
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# Function to plot forecast vs actual values
def plot_forecast(data: pd.DataFrame, forecast, test_data: pd.Series, ticker: str):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(test_data):], test_data, label='Actual', color='blue')
    plt.plot(data.index[-len(test_data):], forecast, label='Forecast', color='orange')
    plt.title(f"{ticker} Forecast vs Actual")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Function to save ARIMA model
def save_arima_model(model, ticker):
    with open(f"{ticker}_arima_model.pkl", 'wb') as file:
        pickle.dump(model, file)
    print(f"ARIMA model for {ticker} saved!")

# Function to save SARIMA model
def save_sarima_model(model, ticker):
    with open(f"{ticker}_sarima_model.pkl", 'wb') as file:
        pickle.dump(model, file)
    print(f"SARIMA model for {ticker} saved!")

# Function to save LSTM model
def save_lstm_model(model, ticker):
    model.save(f"{ticker}_lstm_model.h5")
    print(f"LSTM model for {ticker} saved!")
