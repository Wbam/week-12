# Portfolio Optimization and Forecasting for Tesla and ETFs

This repository contains a series of tasks aimed at analyzing stock market data, forecasting future trends, and optimizing portfolio performance. The tasks are structured as follows:

1. **Task-1: Data Cleaning and Outlier Detection**
2. **Task-2: Time Series Forecasting**
3. **Task-3: Forecast Future Market Trends**
4. **Task-4: Optimize Portfolio Performance**

Each task focuses on different aspects of stock market analysis, from data preparation to advanced forecasting and portfolio optimization.

---

## Table of Contents

- [Task-1: Data Cleaning and Outlier Detection](#task-1)
- [Task-2: Time Series Forecasting](#task-2)
- [Task-3: Forecast Future Market Trends](#task-3)
- [Task-4: Optimize Portfolio Performance](#task-4)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Task-1: Data Cleaning and Outlier Detection

In **Task-1**, we:

- Loaded historical stock data for **Tesla (TSLA)**, **Vanguard Total Bond Market ETF (BND)**, and **S&P 500 ETF (SPY)**.
- Cleaned the data by handling missing values, formatting issues, and ensuring consistency.
- Detected outliers using Z-scores and visualized them to understand their impact on the data.

### Improvements:
- Refined the data cleaning process to address edge cases.
- Implemented a systematic outlier detection approach using statistical methods (Z-scores).

---

## Task-2: Time Series Forecasting

**Task-2** focused on building time series models to forecast Tesla's future stock prices. 

- We explored different forecasting models including **ARIMA**, **SARIMA**, and **LSTM**.
- We split the data into training and testing sets.
- The task involved training the models and optimizing their parameters using evaluation metrics such as **MAE**, **RMSE**, and **MAPE**.
- The forecast results were visualized, and confidence intervals were included to provide insights into price trends and uncertainty.

### Improvements:
- Optimized model selection to achieve better accuracy for Tesla's stock price predictions.
- Enhanced the evaluation process by using multiple metrics to compare model performance.

---

## Task-3: Forecast Future Market Trends

In **Task-3**, we extended the work from Task-2 by forecasting **6-12 months** of future stock prices for **Tesla (TSLA)**.

- The model (ARIMA, SARIMA, or LSTM) trained in Task-2 was used to generate forecasts.
- Confidence intervals were added to visualize the range of potential future stock prices.
- The trends were analyzed for upward, downward, or stable patterns, and the volatility was assessed.

### Improvements:
- Added forecast analysis with confidence intervals to capture future uncertainties.
- Focused on interpreting the long-term trends and potential risks associated with Tesla’s stock.

---

## Task-4: Optimize Portfolio Performance

**Task-4** focused on optimizing the performance of a portfolio containing **Tesla (TSLA)**, **Vanguard Total Bond Market ETF (BND)**, and **S&P 500 ETF (SPY)**.

- **Historical data** was used to calculate the **mean returns** and **covariance matrix**.
- The **mean-variance optimization** approach was employed to find the optimal portfolio weights that maximize the **Sharpe ratio**.
- A **Monte Carlo simulation** was run to generate random portfolios and visualize their returns and risk levels.
- The optimized portfolio was saved to a CSV file along with performance metrics like return, volatility, and Sharpe ratio.

### Improvements:
- Introduced mean-variance optimization to maximize risk-adjusted returns.
- Implemented a Monte Carlo simulation to provide a broader understanding of portfolio possibilities.

---

# GMF Investments Portfolio Management System

The **GMF Investments Portfolio Management System** is designed to optimize financial portfolios for clients by utilizing advanced time series forecasting, machine learning models, and real-time portfolio adjustments. This system helps the **Guide Me in Finance (GMF) Investments** firm manage diverse portfolios efficiently by analyzing market data, predicting trends, and ensuring optimal asset allocation. The system provides actionable insights to enhance the risk-return trade-off for client portfolios.

## Table of Contents

1. [Introduction](#introduction)
2. [Task 5: Portfolio Optimization](#task-5-portfolio-optimization)
3. [Task 6: Dynamic Portfolio Rebalancing](#task-6-dynamic-portfolio-rebalancing)
4. [Task 7: Stress Testing and Scenario Analysis](#task-7-stress-testing-and-scenario-analysis)
5. [Task 8: Machine Learning-Based Market Predictions](#task-8-machine-learning-based-market-predictions)
6. [Task 9: Client-Specific Portfolio Recommendations](#task-9-client-specific-portfolio-recommendations)

---

## Introduction

GMF Investments specializes in creating personalized investment strategies for its clients. The core of this project is to provide a robust system for portfolio management using historical financial data for key assets, such as **Tesla (TSLA)**, **Vanguard Total Bond Market ETF (BND)**, and **S&P 500 ETF (SPY)**. The system uses data-driven insights to predict future market trends, optimize asset allocation, and enhance portfolio performance, ensuring clients can meet their financial objectives while minimizing risks.

---

## Task 5: Portfolio Optimization

### Overview

Portfolio optimization is a key step in financial portfolio management. The goal of this task is to allocate capital optimally among multiple assets to maximize returns while minimizing risk. The **Sharpe Ratio** is used as the optimization criterion, which measures the risk-adjusted return of a portfolio. By maximizing the Sharpe Ratio, the system ensures that the portfolio performs well even under varying market conditions.

### Key Features

- **Historical Data Analysis**: Collects historical stock data for **TSLA**, **BND**, and **SPY**.
- **Risk-Return Optimization**: Maximizes the Sharpe Ratio by optimizing the portfolio’s asset weights.
- **Optimal Weights Calculation**: Returns the optimal asset allocation to maximize risk-adjusted returns.

### Potential Use Cases

- **Portfolio Management**: Helps financial analysts in the firm make decisions about the optimal distribution of assets.
- **Risk Reduction**: Reduces the risk of underperformance by maintaining the optimal balance of assets.

---

## Task 6: Dynamic Portfolio Rebalancing

### Overview

Dynamic portfolio rebalancing involves adjusting the asset weights in a portfolio at regular intervals or when market conditions change. This task ensures that the portfolio continues to align with the desired risk-return profile over time. As market conditions fluctuate, the portfolio's composition needs to be optimized dynamically, ensuring that it remains in an optimal state.

### Key Features

- **Rebalancing Logic**: The system regularly rebalances the portfolio to maintain the optimal weights calculated during the portfolio optimization process.
- **Real-time Adjustment**: Takes into account new market data and adjusts the portfolio in real time.
- **Visualization**: Provides a graphical representation of the portfolio’s performance, including individual assets and the rebalanced portfolio.

### Potential Use Cases

- **Portfolio Tracking**: Investors can track how the portfolio evolves over time with the real-time rebalancing mechanism.
- **Adapt to Market Conditions**: Ensures that the portfolio reacts to market conditions and maintains its optimal balance.

---

## Task 7: Stress Testing and Scenario Analysis

### Overview

Stress testing and scenario analysis are used to evaluate how a portfolio will perform under extreme market conditions. This task simulates different market scenarios such as a market crash or high volatility, to understand how these events will affect the portfolio. The goal is to identify potential risks and ensure the portfolio is resilient against such conditions.

### Key Features

- **Scenario Simulation**: Simulates different market scenarios, such as a market crash or high volatility, to see how the portfolio performs under stress.
- **Risk Assessment**: Provides insights into the potential loss under adverse conditions, helping analysts make more informed decisions about risk management.
- **Portfolio Resilience**: Tests the resilience of the portfolio and adjusts strategies accordingly.

### Potential Use Cases

- **Risk Management**: Helps GMF Investments assess the resilience of portfolios to unexpected market events.
- **Decision Support**: Provides actionable insights to adjust portfolios to ensure clients’ investments are protected during adverse market conditions.

---

## Task 8: Machine Learning-Based Market Predictions

### Overview

Machine learning models are used to predict future market trends based on historical data. This task leverages models like **Long Short-Term Memory (LSTM)** networks to forecast stock prices for **TSLA**, **BND**, and **SPY**. By using these predictions, financial analysts can make more accurate decisions about portfolio adjustments and asset allocation.

### Key Features

- **Market Prediction**: Utilizes machine learning models to predict future prices for selected assets.
- **LSTM Model**: Uses LSTM, a type of recurrent neural network, to analyze time series data and predict future prices.
- **Data Preparation**: Data is preprocessed, scaled, and reshaped to fit the model’s requirements.

### Potential Use Cases

- **Forecasting Market Trends**: Helps predict future stock prices, allowing GMF Investments to make proactive portfolio adjustments.
- **Data-Driven Investment Decisions**: Enhances decision-making by providing more accurate predictions of future market movements.

---

## Task 9: Client-Specific Portfolio Recommendations

### Overview

In this task, personalized portfolio recommendations are generated based on a client’s risk profile. Different clients may have varying risk tolerances, and their portfolios should be adjusted accordingly. This task takes a client's risk profile (e.g., **conservative**, **balanced**, **aggressive**) and suggests an optimal portfolio allocation for them.

### Key Features

- **Risk Profile Analysis**: Takes into account the client’s risk preferences to determine the appropriate asset allocation.
- **Personalized Allocation**: Suggests portfolio weights for **TSLA**, **BND**, and **SPY** that align with the client’s investment goals.
- **Portfolio Customization**: Offers flexibility to adjust the portfolio based on client preferences and financial objectives.

### Potential Use Cases

- **Client Relationship Management**: GMF Investments can provide personalized investment advice to clients based on their specific financial goals and risk tolerance.
- **Customized Portfolio Creation**: Offers tailored portfolio recommendations that help clients achieve their desired returns while managing risks.

---

## Conclusion

This project provides GMF Investments with a comprehensive tool to manage and optimize client portfolios using advanced financial analysis techniques, including **portfolio optimization**, **dynamic rebalancing**, **stress testing**, **machine learning-based predictions**, and **personalized recommendations**. By leveraging historical market data and real-time adjustments, GMF Investments can ensure clients' portfolios are aligned with their financial objectives while managing risk.

---

## Requirements

- **Python 3.x**
- **Libraries**: `yfinance`, `numpy`, `pandas`, `matplotlib`, `scipy`, `tensorflow`, etc.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
