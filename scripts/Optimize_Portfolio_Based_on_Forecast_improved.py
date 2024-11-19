import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf

def load_and_preprocess_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (portfolio_return - risk_free_rate) / portfolio_volatility

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    initial_guess = np.ones(num_assets) / num_assets
    bounds = [(0.0, 1.0)] * num_assets
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    
    result = minimize(negative_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def monte_carlo_simulation(mean_returns, cov_matrix, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = (portfolio_return - 0.0) / portfolio_volatility
    return results
