import numpy as np
import matplotlib.pyplot as plt
from src.portfolio_optimizer import calculate_portfolio_performance

def plot_efficient_frontier(mean_returns, cov_matrix, optimal_weights, risk_free_rate=0.04, num_portfolios=10000):
    print(f"Running Monte Carlo Simulation with {num_portfolios} portfolios...")

    # --- 1. PREPARE DATA STORAGE ---
    results = np.zeros((3, num_portfolios)) # 0: Return, 1: Volatility, 2: Sharpe Ratio

    # --- 2. MONTE CARLO LOOP ---
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)

        p_ret, p_vol, p_sharpe = calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = p_ret
        results[1, i] = p_vol
        results[2, i] = p_sharpe

    # --- 3. PLOTTING ---
    plt.figure(figsize = (10, 6))
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='.', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')

    opt_ret, opt_vol, opt_sharpe = calculate_portfolio_performance(optimal_weights, mean_returns, cov_matrix, risk_free_rate)
    plt.scatter(opt_vol, opt_ret, c= 'red', s = 300, marker = '*', label = f'Optimal (Sharpe: {opt_sharpe:.2f})')

    plt.title('Efficient Frontier: Monte Carlo Simulation')
    plt.xlabel('Volatility (Annual Standard Deviation)')
    plt.ylabel('Expected Annual Return')
    plt.legend()
    plt.grid(True, alpha=0.3)


    filename = 'results/efficient_frontier.png'
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Chart saved as '{filename}'")
    
    plt.show()