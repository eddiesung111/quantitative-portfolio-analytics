# Quantitative Portfolio Optimization Engine 

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview
This project is a **Mean-Variance Portfolio Optimizer** that uses Modern Portfolio Theory (MPT) to construct an optimal asset allocation strategy. 

It fetches historical market data, calculates the annualized covariance matrix, and employs the **SLSQP (Sequential Least Squares Programming)** algorithm to maximize the **Sharpe Ratio**. Additionally, it runs a **Monte Carlo Simulation** (10,000 iterations) to visualize the Efficient Frontier and the feasible set of portfolios.

## 🚀 Key Features
* **Automated Data Pipeline:** Fetches Adjusted Close prices dynamically using `yfinance`.
* **Mathematical Optimization:** Uses `scipy.optimize` to minimize Negative Sharpe Ratio subject to constraints ($\sum w_i = 1$) and bounds ($0 \le w_i \le 1$).
* **Risk Modeling:** Calculates Annualized Volatility and Covariance Matrix ($\Sigma$) to quantify inter-asset correlation.
* **Visualization:** Generates an Efficient Frontier scatter plot mapping Risk ($\sigma$) vs. Expected Return ($E[r]$).

## 🛠️ Tech Stack
* **Python 3**: Core Logic
* **NumPy / Pandas**: Vectorized calculations and Time-series manipulation.
* **SciPy**: Constrained non-linear optimization (SLSQP).
* **Matplotlib**: Data visualization.

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

## 📊 Usage
Run the main orchestrator:
```bash
python main.py
```

## Sample Output
```text
OPTIMAL PORTFOLIO ALLOCATION (Max Sharpe: 1.32)
------------------------------------------------
TSLA : 68.97%
NVDA : 26.01%
GOOG : 5.01%
```

## 📈 Results
After running the engine, the Efficient Frontier chart is saved to the `results/` folder.
Figure 1: Monte Carlo Simulation (n=10,000) showing the Efficient Frontier and Optimal Portfolio (Red Star).

## 🧪 Testing
The project includes comprehensive unit tests to ensure the mathematical solver adheres to constraints (e.g., weights summing to 1.0).
To run tests:
```bash
python -m unittest discover tests
```

## 📂 Project Structure
```text
portfolio-optimization-engine/
├── src/
│   ├── data_loader.py      
│   ├── portfolio_optimizer.py 
│   └── visualizer.py     
├── tests/
│   └── test_optimizer.py   
├── results/
│   └── efficient_frontier.png
├── main.py                  
├── requirements.txt         
└── README.md              
```

## 📐 Mathematical Methodology
The project solves the following optimization problem:

$$\text{Maximize  }  S_p = \frac{E[R_p] - R_f}{\sigma_p}$$

Subject to:
1. Unity Constraint: $\sum_{i=1}^{N} w_i = 1$
2. Long-Only Constraint: $0 \leq w_i \leq 1$

Where:
1. $R_f$ is the Risk-Free Rate (proxied by 10-Year Treasury Yield).
2. $\sigma_p = \sqrt{w^T \Sigma w}$ is the Portfolio Volatility.



## ⚠️ Disclaimer
This software is for educational purposes only. Past performance is not indicative of future results.
