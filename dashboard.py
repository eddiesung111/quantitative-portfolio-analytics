import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from src.data_loader import get_stock_data
from src.portfolio_optimizer import optimize_portfolio
from src.risk_manager import monte_carlo_simulation
from src.options_pricer import black_scholes

# --- Page Configuration ---
st.set_page_config(page_title="Quant Portfolio Engine", layout="wide")
st.title("💰 Quantitative Portfolio Optimization Engine")

# --- Cached ---
@st.cache_data
def get_data_cached(tickers, start, end):
    return get_stock_data(tickers, start, end)

@st.cache_data
def optimize_cached(mean_rets, cov_mat, rate):
    return optimize_portfolio(mean_rets, cov_mat, rate)

# --- User Configuration ---
st.sidebar.header("User Configuration")
tickers_input = st.sidebar.text_input("Enter Tickers (comma separated)", "AAPL, MSFT, GOOG, AMZN, TSLA, NVDA")
start_date = st.sidebar.date_input("Start Date", '2020-01-01')
end_date = st.sidebar.date_input("End Date", datetime.now())
rf_rate = st.sidebar.number_input("Risk-Free Rate (Decimal)", value=0.04, step = 0.01)

# Tickers in the Correct Format
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Portfolio Optimizer", "🛡️ Risk Management", "📉 Options Pricer"])

# --- Tab 1 ---
with tab1:
    st.header("Mean-Variance Optimization")
    st.write("Calculates the asset allocation that maximizes the Sharpe Ratio.")
    if st.button("Run Optimization", type = 'primary'):
        if len(tickers) < 2:
            st.error("⚠️ Please enter at least 2 tickers to optimize a portfolio.")
        else:
            with st.spinner("Fetching market data & optimizing..."):
                df = get_data_cached(tickers, start_date, end_date)
                daily_returns = df.pct_change().dropna()
                mean_returns = daily_returns.mean() * 252
                cov_matrix = daily_returns.cov() * 252

                results = optimize_cached(mean_returns, cov_matrix, rf_rate)
                weights = results.x
                max_sharpe = -results.fun

                portfolio_return = np.sum(mean_returns * weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                # Store in session state for later use
                st.session_state['mean_returns'] = mean_returns
                st.session_state['cov_matrix'] = cov_matrix
                st.session_state['weights'] = weights
                st.session_state['opt_done'] = True
                
                # Display Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Annual Return", f"{portfolio_return:.2%}")
                col2.metric("Annual Volatility", f"{portfolio_volatility:.2%}")
                col3.metric("Sharpe Ratio", f"{max_sharpe:.2f}")
                st.divider()

                # Display Weights and Pie Chart
                c1, c2 = st.columns([1, 1])

                # Display Optimal Weights
                with c1:
                    st.subheader("Optimal Weights")
                    alloc_df = pd.DataFrame({
                        'Asset': tickers,
                        'Weight': weights
                    })
                    alloc_df = alloc_df.sort_values(by='Weight', ascending=False)
                    st.dataframe(
                        alloc_df.style.format({"Weight": "{:.2%}"}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Display Pie Chart
                with c2:
                    st.subheader("Allocation Breakdown")
                    fig_pie = px.pie(
                            alloc_df, 
                            values='Weight', 
                            names='Asset',
                            hole=0.4,
                            color_discrete_sequence=px.colors.sequential.RdBu
                        )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("👈 Enter tickers in the sidebar and click 'Run Optimization' to start.")

# --- Tab 2 ---
with tab2:
    st.header("Monte Carlo Simulation")
    st.write("Simulates future portfolio performance using Geometric Brownian Motion (GBM).")
    if 'opt_done' not in st.session_state or not st.session_state['opt_done']:
        st.warning("⚠️ Please run the Portfolio Optimizer first.")
    else:
        # User Configuration
        col1, col2 = st.columns(2)
        with col1:
            sims = st.slider("Number of Simulations", 100, 5000, 2000, step=100)
            initial_investment = st.number_input("Initial Investment Amount ($)", min_value=5000, value=10000, step=1000)
        with col2:
            days = st.slider("Time Horizon (Days)", 30, 500, 250, step = 10)
        
        if st.button("Run Simulation", type = 'primary'):
            with st.spinner("Running Monte Carlo Simulations..."):
                mean_returns = st.session_state['mean_returns']
                cov_matrix = st.session_state['cov_matrix']
                weights = st.session_state['weights']

                price_paths, final_values = monte_carlo_simulation(
                    mean_returns, cov_matrix, weights, initial_investment, days, sims)
                
                var_95 = np.percentile(final_values, 5)
                st.subheader("Projected Portfolio Value Paths")
                fig_paths = go.Figure()
                
                # Plot first 50 simulations
                for i in range(50):
                    fig_paths.add_trace(go.Scatter(
                        y=price_paths[:, i],
                        mode='lines',
                        opacity=0.3,
                        showlegend=False
                    ))
                fig_paths.update_layout(xaxis_title="Days", yaxis_title="Portfolio Value ($)")
                st.plotly_chart(fig_paths, use_container_width=True)

                # Display VaR Analysis
                st.subheader("Value at Risk (VaR) Analysis")
                fig_hist = px.histogram(
                    final_values, 
                    nbins=50, 
                    title="Distribution of Final Portfolio Values",
                    labels={'value': 'Final Portfolio Value ($)'},
                    color_discrete_sequence=['#636EFA']
                )
                fig_hist.add_vline(x=var_95, line_width=3, line_dash="dash", line_color="red")
                st.plotly_chart(fig_hist, use_container_width=True)
                st.metric("95% Value at Risk (VaR)", f"${var_95:,.2f}")

# --- Tab 3 ---
with tab3:
    st.header("Black-Scholes Option Pricing")
    st.write("Calculate the theoretical price of European options and visualize sensitivity.")

    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Market Parameters")
        current_price = st.number_input("Stock Price ($)", value=100.0, step=1.0)
        strike_price = st.number_input("Strike Price ($)", value=100.0, step=1.0)
        time_to_expiry = st.number_input("Time to Expiry (Years)", value=1.0, step=0.1)
    with col2:
        st.subheader("Risk Parameters")
        volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
        rf_rate_opt = st.number_input("Risk-Free Rate (Decimal)", value=0.04, step=0.001, key="rf_opt")
    st.divider()

    # Calculate Option Prices
    call_price = black_scholes(current_price, strike_price, time_to_expiry, rf_rate_opt, volatility, 'call')
    put_price = black_scholes(current_price, strike_price, time_to_expiry, rf_rate_opt, volatility, 'put')

    # Display Results
    m1, m2, m3 = st.columns(3)
    m1.metric("CALL Option Price", f"${call_price:.2f}", delta=None)
    m2.metric("PUT Option Price", f"${put_price:.2f}", delta=None)
    with m3:
        choice = st.selectbox("Select Option Type for Sensitivity Analysis", options=['call', 'put'])

    st.subheader("Interactive Volatility Surface")
    st.write(f"Visualizing how **{choice} Option Price** changes with Stock Price and Volatility.")

    with st.spinner("Generating 3D Surface..."):
        spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 20)
        vol_range = np.linspace(0.05, 0.8, 20)

        X, Y = np.meshgrid(spot_range, vol_range)
        Z = np.array([
            [black_scholes(s, strike_price, time_to_expiry, rf_rate_opt, v, choice) for s in spot_range]
            for v in vol_range
        ])

        fig_3d = go.Figure(data=[go.Surface(
            z=Z, x=X, y=Y,
            colorscale='Viridis',
            opacity=0.9
        )])

        fig_3d.update_layout(
            title="Call Price Sensitivity",
            scene=dict(
                xaxis_title='Stock Price ($)',
                yaxis_title='Volatility (σ)',
                zaxis_title='Option Price ($)'
            ),
            width=900, height=600,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(fig_3d, use_container_width=True)