import datetime
import numpy as np
from src.data_loader import get_stock_data
from src.portfolio_optimizer import optimize_portfolio
from src.visualizer import plot_efficient_frontier
from src.risk_manager import monte_carlo_simulation, plot_simulation

TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA']
START_DATE = '2020-01-01'
END_DATE = datetime.datetime.today().strftime('%Y-%m-%d')
RISK_FREE_RATE = 0.044

def main():
    print("\n--- Portfolio Optimization Engine ---")
    df = get_stock_data(TICKERS, START_DATE, END_DATE)
    returns = df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    print("\n[Optimization] Finding Maximum Sharpe Ratio...")
    result = optimize_portfolio(mean_returns, cov_matrix, RISK_FREE_RATE)
    optimal_weights = result.x

    print("\n----------------------------------------------------------------")
    print(f"OPTIMAL PORTFOLIO ALLOCATION (Max Sharpe: {-result.fun:.2f})")
    print("----------------------------------------------------------------")

    portfolio = sorted(zip(TICKERS, optimal_weights), key=lambda x: x[1], reverse=True)

    for ticker, weight in portfolio:
        if weight > 0.0001:
            print(f"{ticker:<5}: {weight*100:.2f}%")

    plot_efficient_frontier(mean_returns, cov_matrix, optimal_weights, RISK_FREE_RATE)
    print("\n--- Process Complete ---")

    paths, final_values = monte_carlo_simulation(mean_returns, cov_matrix, result.x)
    plot_simulation(paths, final_values, initial_investment=10000)

if __name__ == "__main__":
    main()