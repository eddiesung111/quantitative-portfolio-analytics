import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

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

# --- Helper Functions ---
def get_stressed_parameters(base_means, base_cov, tickers, scenario, custom_params):
    pass


# --- User Configuration ---
st.sidebar.header("User Configuration")
tickers_input = st.sidebar.text_input("Enter Tickers (comma separated)", "MSFT", "JPM", "JNJ", "KO", "XOM", "TLT")
start_date = st.sidebar.date_input("Start Date", '2020-01-01')
end_date = st.sidebar.date_input("End Date", '2025-12-31')
rf_rate = st.sidebar.number_input("Risk-Free Rate (Decimal)", value=0.04, step = 0.01)

# Tickers in the Correct Format
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio Optimizer", "🛡️ Risk Management", "📉 Options Pricer"])

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
    st.caption("Simulate portfolio performance under various market regimes using Geometric Brownian Motion.")
    if 'opt_done' not in st.session_state or not st.session_state['opt_done']:
        st.warning("⚠️ Please go to the 'Optimization' tab and generate a portfolio first.")
    else:
        # User Configuration
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### ⚙️ Configuration")
            scenario = st.selectbox(
                "Select Market Scenario", 
                options=["Normal Market", "2008 Crash", "Tech Bubble", "Custom Stress Test"]
            )
            
            # --- CUSTOM MODE SLIDERS (Conditional) ---
            custom_params = {}
            if scenario == "Custom Stress Test":
                st.info("🛠️ Custom Stress Parameters")
                custom_params['vol_mult'] = st.slider("Volatility Multiplier", 1.0, 5.0, 1.5, 0.1)
                custom_params['corr_force'] = st.slider("Force Correlation", 0.0, 1.0, 0.0, 0.1)
                custom_params['drift'] = st.slider("Annual Drift Override", -0.5, 0.5, -0.1, 0.05)
            
            st.divider()
            sims = st.slider("Number of Simulations", 1000, 50000, 5000, step=1000)
            
        with col2:
            st.markdown("### 📊 Simulation Parameters")
            c1, c2 = st.columns(2)
            with c1:
                initial_investment = st.number_input("Initial Investment ($)", value=10000, step=1000)
                confidence_level = st.selectbox("VaR Confidence Level", [95, 99])
            with c2:
                days = st.slider("Time Horizon (Days)", 30, 365, 252)

        # --- Run Simulation ---
        if st.button("Run Simulation", type = 'primary', use_container_width=True):
            with st.spinner(f"Simulating {scenario} conditions..."):
                base_means = st.session_state['mean_returns']
                base_cov = st.session_state['cov_matrix']
                weights = st.session_state['weights']

                str_means, str_cov = get_stressed_parameters(base_means, base_cov, tickers, scenario, custom_params)

                price_paths, final_values = monte_carlo_simulation(
                    str_means, str_cov, weights, initial_investment, days, sims
                )
                
                var_percentile = np.percentile(final_values, 100 - confidence_level)
                var_loss = initial_investment - var_percentile

        # --- Display Results ---
                st.subheader("Projected Portfolio: {scenario}")
                path_color = 'red' if "Crash" in scenario or "Bubble" in scenario else 'royalblue'
                fig_paths = go.Figure()
                
                indices = np.random.choice(price_paths.shape[1], 100, replace=False)
                for i in indices:
                    fig_paths.add_trace(go.Scatter(
                        y=price_paths[:, i],
                        mode='lines',
                        line=dict(color=path_color, width=1),
                        opacity=0.1,
                        showlegend=False
                    ))

                mean_path = np.mean(price_paths, axis=1)
                fig_paths.add_trace(go.Scatter(
                    y = mean_path, 
                    mode = 'lines'),
                    name = 'Average Path',
                    line = dict(color='black', width=3, dash = "dash")
                )

                fig_paths.update_layout(xaxis_title="Days", yaxis_title="Portfolio Value ($)", height = 400, margin = dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_paths, use_container_width=True)

                c_res1, c_res2 = st.columns([2, 1])
                with c_res1:
                    st.subheader("Final Value Distribution")
                    fig_hist = px.histogram(
                        x=final_values, nbins=50, 
                        color_discrete_sequence=[path_color], opacity=0.7
                    )
                    fig_hist.add_vline(x=var_percentile, line_width=3, line_dash="dash", line_color="orange")
                    fig_hist.add_annotation(x=var_percentile, y=10, text=f"VaR {confidence_level}%", showarrow=True, arrowhead=1)
                    fig_hist.update_layout(xaxis_title="Final Portfolio Value ($)", yaxis_title="Frequency", height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                with c_res2:
                    st.markdown("### Risk Metrics")
                    st.metric(label=f"Value at Risk ({confidence_level}%)", value=f"${initial_investment - var_percentile:,.2f}", delta_color="inverse")
                    st.metric(label="Worst Case (Min)", value=f"${np.min(final_values):,.2f}")
                    st.metric(label="Success Rate (> Initial)", value=f"{np.mean(final_values > initial_investment)*100:.1f}%")


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