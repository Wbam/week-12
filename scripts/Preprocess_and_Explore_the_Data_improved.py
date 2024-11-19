"""# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

# Set plot style and figure size
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Function to download and clean data
def load_and_clean_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df['Daily Return'] = df['Close'].pct_change()
    return df

# Function to perform EDA
def perform_eda(data: pd.DataFrame, ticker: str):
    figures = []  # List to store the figures
    
    # Plot closing price
    fig1, ax1 = plt.subplots()
    data['Close'].plot(ax=ax1, title=f"{ticker} Closing Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    figures.append(fig1)
    
    # Plot daily percentage change
    fig2, ax2 = plt.subplots()
    data['Daily Return'].plot(ax=ax2, title=f"{ticker} Daily Returns")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Daily Return (%)")
    figures.append(fig2)
    
    # Plot rolling statistics for volatility analysis
    rolling_mean = data['Close'].rolling(window=30).mean()
    rolling_std = data['Close'].rolling(window=30).std()
    fig3, ax3 = plt.subplots()
    ax3.plot(data['Close'], label="Closing Price")
    ax3.plot(rolling_mean, label="30-Day Rolling Mean", color='orange')
    ax3.plot(rolling_std, label="30-Day Rolling Std Dev", color='red')
    ax3.legend()
    ax3.set_title(f"{ticker} Rolling Statistics")
    figures.append(fig3)
    
    # Seasonal decomposition
    decomposition = seasonal_decompose(data['Close'], model='additive', period=252)
    fig4 = decomposition.plot()
    figures.append(fig4)

    return figures

# Function to detect outliers
def detect_outliers(data: pd.DataFrame, column: str, z_threshold: float = 3.0) -> pd.DataFrame:
    data['Z-Score'] = (data[column] - data[column].mean()) / data[column].std()
    data['Outlier'] = np.abs(data['Z-Score']) > z_threshold
    return data

# Function to save data
def save_data(data: pd.DataFrame, filename: str):
    data.to_csv(filename, index=True)
    print(f"Data saved to {filename}")

# Function to plot outliers
def plot_outliers(data: pd.DataFrame, column: str, ticker: str):
    plt.figure(figsize=(12, 6))
    plt.plot(data[column], label='Closing Price', alpha=0.6)
    outliers = data[data['Outlier']]
    plt.scatter(outliers.index, outliers[column], color='red', label='Outliers', zorder=5)
    plt.title(f"{ticker} Outliers in {column}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

# Function for interactive visualization
def interactive_plot(data, column, ticker):
    y_data = data[column]
    if isinstance(y_data, pd.DataFrame):
        y_data = y_data.squeeze()  
    fig = px.line(
        x=data.index,
        y=y_data,
        title=f"{ticker} {column} Over Time",
        labels={"x": "Date", "y": column},
        markers=True
    )
    fig.update_traces(hovertemplate="%{x}: %{y:.2f}")
    fig.show() """

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

# Set plot style and figure size
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Function to download and clean data
def load_and_clean_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    df['Daily Return'] = df['Close'].pct_change()
    df.reset_index(inplace=True)  # Ensure that the Date column is a column, not an index
    return df

# Function to perform EDA
def perform_eda(data: pd.DataFrame, ticker: str):
    figures = []  # List to store the figures
    
    # Plot closing price
    fig1, ax1 = plt.subplots()
    data['Close'].plot(ax=ax1, title=f"{ticker} Closing Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    figures.append(fig1)
    
    # Plot daily percentage change
    fig2, ax2 = plt.subplots()
    data['Daily Return'].plot(ax=ax2, title=f"{ticker} Daily Returns")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Daily Return (%)")
    figures.append(fig2)
    
    # Plot rolling statistics for volatility analysis
    rolling_mean = data['Close'].rolling(window=30).mean()
    rolling_std = data['Close'].rolling(window=30).std()
    fig3, ax3 = plt.subplots()
    ax3.plot(data['Close'], label="Closing Price")
    ax3.plot(rolling_mean, label="30-Day Rolling Mean", color='orange')
    ax3.plot(rolling_std, label="30-Day Rolling Std Dev", color='red')
    ax3.legend()
    ax3.set_title(f"{ticker} Rolling Statistics")
    figures.append(fig3)
    
    # Seasonal decomposition
    decomposition = seasonal_decompose(data['Close'], model='additive', period=252)
    fig4 = decomposition.plot()
    figures.append(fig4)

    return figures

# Function to detect outliers
def detect_outliers(data: pd.DataFrame, column: str, z_threshold: float = 3.0) -> pd.DataFrame:
    data['Z-Score'] = (data[column] - data[column].mean()) / data[column].std()
    data['Outlier'] = np.abs(data['Z-Score']) > z_threshold
    return data

# Function to save data
def save_data(data: pd.DataFrame, filename: str):
    data.to_csv(filename, index=False)  # Save without the index as a column
    print(f"Data saved to {filename}")

# Function to plot outliers
def plot_outliers(data: pd.DataFrame, column: str, ticker: str):
    plt.figure(figsize=(12, 6))
    plt.plot(data[column], label='Closing Price', alpha=0.6)
    outliers = data[data['Outlier']]
    plt.scatter(outliers.index, outliers[column], color='red', label='Outliers', zorder=5)
    plt.title(f"{ticker} Outliers in {column}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

# Function for interactive visualization
def interactive_plot(data, column, ticker):
    y_data = data[column]
    if isinstance(y_data, pd.DataFrame):
        y_data = y_data.squeeze()  
    fig = px.line(
        x=data['Date'],  # Explicitly using 'Date' column for x-axis
        y=y_data,
        title=f"{ticker} {column} Over Time",
        labels={"x": "Date", "y": column},
        markers=True
    )
    fig.update_traces(hovertemplate="%{x}: %{y:.2f}")
    fig.show()


