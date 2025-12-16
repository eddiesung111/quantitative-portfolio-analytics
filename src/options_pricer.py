import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    '''
    Calculate the option price using Black-Scholes formula
    :param S: Stock Price
    :param K: Strike Price
    :param T: Time to Maturity (in years)
    :param r: Risk-free Interest Rate
    :param sigma: Volatility of the underlying stock
    :param option_type: 'call' or 'put'
    '''
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    return price

def plot_3d_option_surface(K, T, r, option_type='call'):
    '''
    Plot the 3D surface of option prices over stock price and volatility
    '''
    # --- 1. PREPARE GRID ---
    S_range = np.linspace(50, 150, 50)
    vol_range = np.linspace(0.1, 1.0, 50)

    # Each (i, j) has the combination of the stock price and volatility
    # Eg. S = [10, 20] and vol = [0.1, 0.2, 0.3] will produce:
    # X = [10, 20]
    #     [10, 20]
    #     [10, 20]
    # Y = [0.1, 0.1]
    #     [0.2, 0.2]
    #     [0.3, 0.3]
    X, Y = np.meshgrid(S_range, vol_range)
    Z = np.zeros_like(X)

    # --- 2. CALCULATE OPTION PRICES ---
    # For each combination of stock price and volatility, calculate the option price
    for i in range(len(vol_range)):
        for j in range(len(S_range)):
            Z[i, j] = black_scholes(X[i, j], K, T, r, Y[i, j], option_type)
     
    # --- 3. PLOTTING ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_title(f'Black-Scholes {option_type} Option Price Surface\nStrike: ${K}, Time: {T} year')
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Volatility (sigma)')
    ax.set_zlabel('Option Price ($)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    

    filename = f'results/black_scholes_{option_type}_surface.png'
    plt.savefig(filename)
    print(f"Chart saved as '{filename}'")

    plt.show()

if __name__ == "__main__":
    plot_3d_option_surface(K=100, T=1, r=0.05, option_type='call')
    plot_3d_option_surface(K=100, T=1, r=0.05, option_type='put')