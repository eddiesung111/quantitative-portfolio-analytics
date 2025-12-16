import numpy as np
import matplotlib.pyplot as plt
def monte_carlo_simulation(mean_returns, cov_matrix, weights, initial_investment=10000, days=500, simulations=10000):
    '''
    Perform Monte Carlo simulation to project portfolio value over time.
    '''
    weights = np.array(weights)
    daily_return = mean_returns / 252
    daily_cov = cov_matrix / 252
    # Return = \sum return_i * weight_i
    # Volatility = sqrt(weights.T * Covariance_Matrix * weights)
    portfolio_return =  np.sum(daily_return * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_cov, weights)))

    # Simulate price paths by normal distribution, N(return, volatility)
    daily_return = np.random.normal(portfolio_return, portfolio_volatility, (days, simulations))
    price_paths = initial_investment * np.cumprod(1 + daily_return, axis=0)

    # Add initital investment row
    start_row = np.full((1, simulations), initial_investment)
    price_paths = np.vstack((start_row, price_paths))

    final_values = price_paths[-1]

    # Calculate statistics
    expected_return = np.mean(final_values)
    var_95_value = np.percentile(final_values, 5)
    profit_or_loss = var_95_value - initial_investment

    # Print results
    print("\n----------------------------------------------------------------")
    print(f"MONTE CARLO SIMULATION RESULTS ({simulations} runs / {days} days)")
    print("----------------------------------------------------------------")
    print(f"Initial Investment:   ${initial_investment:,.2f}")
    print(f"Expected Final Value: ${expected_return:,.2f}")
    print(f"Worst Case (95%):     ${var_95_value:,.2f}")
    if profit_or_loss < 0:
        print(f"Risk (VaR 95%):       We risk losing ${abs(profit_or_loss):,.2f}")
    else:
        print(f"Risk (VaR 95%):       Even in the worst 5%, we GAIN ${profit_or_loss:,.2f}")
    print("----------------------------------------------------------------")

    return price_paths, final_values

def plot_simulation(price_paths, final_values, initial_investment):
    # Plot the first 50 simulations results
    plt.figure(figsize=(10,6))
    plt.plot(price_paths[:, :50], alpha = 0.5, linewidth = 1)
    plt.title(f"Projected Portfolio Paths (First 50 Simulations)")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.axhline(initial_investment, color='black', linestyle='--', label="Start Value")

    file_name = 'results/monte_carlo_simulation.png'
    plt.savefig(file_name)
    print(f"Chart saved as '{file_name}'")
    
    plt.show()

    # Plot distribution of final portfolio values
    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=50, alpha=0.6, color='blue')
    var_95 = np.percentile(final_values, 5)
    plt.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f"95% VaR: ${var_95:.0f}")
    plt.title("Distribution of Final Portfolio Values")
    plt.xlabel("Final Value ($)")
    plt.ylabel("Frequency")
    plt.legend()
    
    filename = 'results/final_value_distribution.png'
    plt.savefig(filename)
    print(f"Chart saved as '{filename}'")

    plt.show()
