import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# 1. Data Retrieval
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # List of stocks
start_date = '2020-01-01'
end_date = '2024-01-01'

# Attempt to download data up to 3 times in case of failure
for i in range(3):
    try:
        data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
        break
    except Exception as e:
        print(f"Attempt {i+1} failed: {e}")
        time.sleep(5)  # Pause before retrying

# Calculate daily returns and drop missing values
returns = data.pct_change(fill_method=None).dropna()

# 2. Financial Metrics Calculation
mean_returns = returns.mean() * 252  # Annualized return
cov_matrix = returns.cov() * 252  # Annualized covariance matrix

# 3. Minimization Function (Negative Sharpe Ratio)
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Calculates the portfolio performance in terms of return, volatility, and Sharpe ratio.
    
    :param weights: Portfolio allocation weights
    :param mean_returns: Expected annual returns of assets
    :param cov_matrix: Annualized covariance matrix of assets
    :param risk_free_rate: Risk-free rate (default 2%)
    :return: Negative Sharpe ratio (for minimization)
    """
    returns = np.dot(weights, mean_returns)  # Expected return
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio volatility
    sharpe_ratio = (returns - risk_free_rate) / volatility  # Sharpe ratio
    return -sharpe_ratio  # Negative for minimization

# Constraints and Bounds
num_assets = len(assets)
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Sum of weights must be 1
bounds = tuple((0, 1) for _ in range(num_assets))  # Each weight between 0 and 1
initial_guess = num_assets * [1. / num_assets]  # Equal allocation initial guess

# Optimization Process
optimal = minimize(portfolio_performance, initial_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = optimal.x  # Extract optimal weights

# 4. Display Results
plt.figure(figsize=(10, 6))
num_portfolios = 10000  # Number of simulated portfolios
results = np.zeros((3, num_portfolios))  # Store return, volatility, and Sharpe ratio

# Monte Carlo Simulation to generate random portfolios
for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(num_assets))  # Generate random weights
    portfolio_return = np.dot(weights, mean_returns)  # Compute return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Compute volatility
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = (portfolio_return - 0.02) / portfolio_volatility  # Compute Sharpe ratio

# Plot the Efficient Frontier
plt.gcf().canvas.manager.set_window_title("Efficient Frontier - Portfolio Optimization")
plt.scatter(results[1], results[0], c=results[2], cmap='viridis', marker='o', alpha=0.3)
plt.colorbar(label='Sharpe Ratio')  # Color bar for Sharpe Ratio
plt.scatter(np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))), np.dot(optimal_weights, mean_returns), c='red', marker='*', s=200, label='Optimal Portfolio')  # Mark the optimal portfolio
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier of Portfolios')
plt.legend()
plt.show()

# Display optimal weights as a DataFrame
df_weights = pd.DataFrame({'Stocks': assets, 'Optimal Weights': optimal_weights})
print("\nOptimal Portfolio Weights:")
print(df_weights)