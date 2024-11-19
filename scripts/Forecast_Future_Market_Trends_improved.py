import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import load_model

# Function to load and preprocess the data 
def load_and_preprocess_data(ticker):
    filename = f"{ticker}_processed.csv"  
    data = pd.read_csv(filename, index_col='Date', parse_dates=True)
    
    # Ensure the 'Close' column is numeric and drop any rows with NaN values
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data.dropna(subset=['Close'], inplace=True)
    
    return data

# ARIMA Model - Forecasting with ARIMA
def arima_forecast(data, forecast_steps):
    arima_model = ARIMA(data['Close'], order=(5,1,0)) 
    arima_results = arima_model.fit()
    arima_forecast = arima_results.forecast(steps=forecast_steps)
    return arima_forecast

# SARIMA Model - Forecasting with SARIMA
def sarima_forecast(data, forecast_steps):
    sarima_model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  
    sarima_results = sarima_model.fit()
    sarima_forecast = sarima_results.get_forecast(steps=forecast_steps)
    return sarima_forecast

# LSTM Model - Forecasting with LSTM
def lstm_forecast(data, forecast_steps, model_path):
    lstm_model = load_model(model_path)  # Load the pre-trained LSTM model
    last_data_point = data['Close'].values[-1]
    lstm_forecast = []

    # Forecast for the next 6 months (252 trading days)
    for _ in range(forecast_steps):
        input_sequence = np.array([last_data_point])
        input_sequence = input_sequence.reshape((1, 1, 1))  # Reshape for LSTM input
        next_prediction = lstm_model.predict(input_sequence)
        lstm_forecast.append(next_prediction[0, 0])
        last_data_point = next_prediction[0, 0]  
    
    return lstm_forecast

# Plotting forecasted data
def plot_forecast(data, forecast_dates, arima_forecast, sarima_forecast, lstm_forecast, ticker, name):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label="Historical Stock Prices")
    plt.plot(forecast_dates, arima_forecast, label="ARIMA Forecast", color='orange')
    plt.title(f"ARIMA Forecast for {name}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label="Historical Stock Prices")
    plt.plot(forecast_dates, sarima_forecast.predicted_mean, label="SARIMA Forecast", color='red')
    plt.title(f"SARIMA Forecast for {name}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label="Historical Stock Prices")
    plt.plot(forecast_dates, lstm_forecast, label="LSTM Forecast", color='green')
    plt.title(f"LSTM Forecast for {name}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

# Save forecast data to CSV
def save_forecast_data(forecast_dates, arima_forecast, sarima_forecast, lstm_forecast, ticker):
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'ARIMA Forecast': arima_forecast,
        'SARIMA Forecast': sarima_forecast.predicted_mean,
        'LSTM Forecast': lstm_forecast
    })
    forecast_df.to_csv(f"{ticker}_forecast_combined.csv", index=False)

# Save forecast analysis to CSV
def save_forecast_analysis(forecast_dates, arima_forecast, sarima_forecast, lstm_forecast, volatility_range, trend, opportunity_risk, ticker):
    analysis_df = pd.DataFrame({
        'Date': forecast_dates,
        'ARIMA Forecast': arima_forecast,
        'SARIMA Forecast': sarima_forecast.predicted_mean,
        'LSTM Forecast': lstm_forecast,
        'Volatility': volatility_range,
        'Trend': trend,
        'Opportunity/Risk': opportunity_risk
    })
    analysis_df.to_csv(f"{ticker}_forecast_analysis_combined.csv", index=False)
