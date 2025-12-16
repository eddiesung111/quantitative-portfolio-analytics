# Quantitative Portfolio Optimization Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-Pytest-yellow)

## 📌 Overview
This project is a comprehensive quantitative finance toolkit that combines **Portfolio Optimization**, **Risk Management**, and **Derivatives Pricing**.

It uses Modern Portfolio Theory (MPT) to construct optimal asset allocations, employs **Monte Carlo Simulations** to quantify tail risk (VaR), and implements the **Black-Scholes Model** to generate 3D volatility surfaces for option valuation.

## 🚀 Key Features
* **Portfolio Optimization:** Uses `scipy.optimize` (SLSQP) to maximize the **Sharpe Ratio** subject to constraints ($\sum w_i = 1$) and long-only bounds.
* **Risk Management:** Runs a **10,000-iteration Monte Carlo Simulation** using Geometric Brownian Motion (GBM) to calculate the **95% Value at Risk (VaR)**.
* **Derivatives Pricing:** Implements the Black-Scholes formula to price European Options and generates a **3D Volatility Surface** to visualize the relationship between Spot Price, Volatility, and Option Value.
* **Automated Data:** Fetches Adjusted Close prices dynamically using `yfinance`.

## 🛠️ Tech Stack
* **Python 3**: Core Logic
* **NumPy / Pandas**: Vectorized calculations and time-series manipulation.
* **SciPy**: Constrained non-linear optimization (SLSQP) and statistical functions.
* **Matplotlib / Seaborn**: 2D Data visualization (Histograms, Scatter Plots).
* **Pytest**: Unit testing framework.

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/eddiesung111/portfolio-optimizer.git](https://github.com/eddiesung111/portfolio-optimizer.git)
   ```
   
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🛡️ Module 1: Portfolio Optimization & Risk
This module fetches historical data, optimizes asset allocation for the Sharpe Ratio, and runs a Monte Carlo simulation to stress-test the portfolio.
### 💻 How to Run
```bash
python main.py
```
### 📊 Methodology & Results

#### Optimization (Efficient Frontier)

The project solves the following optimization problem:
   
$$\text{Maximize  } S_p = \frac{E[R_p] - R_f}{\sigma_p}$$
     
Subject to:

   - Unity Constraint: $\sum_{i=1}^{N} w_i = 1$

   - Long-Only Constraint: $0 \leq w_i \leq 1$

Sample Output:
```text
OPTIMAL PORTFOLIO ALLOCATION (Max Sharpe: 1.32)
------------------------------------------------
TSLA : 68.97%
NVDA : 26.01%
GOOG : 5.01%
```

- Efficient Frontier Visualization: Maps the risk-return profile of random portfolios vs. the optimal allocation.
![Efficient Frontier](results/efficient_frontier.png)

#### Monte Carlo Simulation (Projected Paths)
Uses Geometric Brownian Motion (GBM) with Cholesky Decomposition to model correlated asset paths over a 2-year horizon (10,000 iterations).
![Monte Carlo Paths](results/monte_carlo_simulation.png)

#### Risk Analysis (Value at Risk)
A histogram of final portfolio values showing the 95% VaR threshold (the red dashed line).
![VaR Distribution](results/final_value_distribution.png)

## 📉 Module 2: Derivatives Pricing
This module focuses on individual options pricing using the Black-Scholes-Merton model, visualizing how option premiums react to market variables.

### 💻 How to Run
```bash
python src/options_pricer.py
```
Opens an interactive 3D plot window.

### ⚙️ 📊 Methodology & Results

#### The Math
Prices European Call/Put options based on Stock Price ($S$), Strike ($K$), Time ($T$), Risk-free Rate ($r$), and Volatility.

1. Call Option Volatility Surface
Visualizes the relationship between Underlying Price, Volatility, and Call Price. Note how higher volatility increases the option value (Vega).
![Call Surface](results/black_scholes_call_surface.png)

2. Put Option Volatility Surface
Visualizes the Put Price surface. Note the inverse relationship with stock price compared to the Call option.
![Put Surface](results/black_scholes_put_surface.png)

## 🧪 Testing
The project uses `pytest` to ensure mathematical accuracy (e.g., weights summing to 1.0, Put-Call Parity).
To run the full test suite:
```bash
pytest
```

## 📂 Project Structure
```text
portfolio-optimization-engine/
├── src/
│   ├── __init__.py           # Package marker
│   ├── data_loader.py        # Yfinance fetcher
│   ├── portfolio_optimizer.py# Mean-Variance Solver
│   ├── risk_manager.py       # Monte Carlo Engine
│   └── options_pricer.py     # Black-Scholes & 3D Plotting
├── tests/
│   ├── test_optimizer.py     # Optimizer tests
│   └── test_pricer.py        # Black-Scholes tests
├── results/
│   ├── monte_carlo_simulation.png
│   ├── final_value_distribution.png
│   └── option_surface.png
├── main.py                   # Orchestrator script
├── requirements.txt          # Dependencies
└── README.md                 # Project Documentation      
```

## ⚠️ Disclaimer
This software is for educational purposes only. Past performance is not indicative of future results.
