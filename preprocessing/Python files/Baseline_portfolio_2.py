import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os

current_directory = os.getcwd()

portfolio_2 = pd.read_csv(current_directory + '/Portfolio_2-Baseline.csv')
portfolio_2 = portfolio_2.drop(columns='Date')

returns = np.log(portfolio_2 / portfolio_2.shift(1))
returns = returns.dropna()

cov_matrix = returns.cov() * 252

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, returns):
    return np.sum(returns.mean())

def sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

risk_free_rate = .04

def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.234) for i in range(20)]
initial_weights = [1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20]

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)

optimal_weights2=optimized_results.x
optimal_weights2 = pd.DataFrame(optimal_weights2)

weight_baseline2 = np.round(optimal_weights2, decimals=4)

df_weights = pd.DataFrame(weight_baseline2)

# Save to CSV
df_weights.to_csv("Baseline_weights2.csv", index=False)
