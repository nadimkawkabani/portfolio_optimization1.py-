import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
import cvxpy as cp
import warnings

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Set page configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# Define diversified assets across industries
ASSETS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",    # Technology
    "NFLX", "ADSK", "INTC", "NVDA", "TSM",      # Semiconductors & Tech
    "JPM", "GS", "BAC", "C", "AXP",             # Finance
    "UNH", "LLY", "PFE", "BMY", "MRK"           # Healthcare
]

# Sidebar Filters
with st.sidebar:
    st.header("Portfolio Settings")
    selected_assets = st.multiselect("Select Assets", ASSETS, default=ASSETS)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2017-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    gamma_value = st.slider("Risk Aversion (Gamma)", 0.001, 100.0, 0.1, 0.001)
    max_leverage_value = st.slider("Max Leverage", 1, 5, 1, 1)

# Download market data
try:
    prices_df = yf.download(selected_assets, start=start_date, end=end_date)["Close"]
    returns = prices_df.pct_change().dropna()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Equal Weight Portfolio (Baseline)
portfolio_weights = np.array([1 / len(selected_assets)] * len(selected_assets))
portfolio_returns = returns.dot(portfolio_weights)

st.subheader("Baseline: 1/n Portfolio Performance")
qs.plots.snapshot(portfolio_returns, title="Equal Weighted Portfolio Performance", grayscale=True)
qs.reports.metrics(portfolio_returns, benchmark="SPY", mode="basic")

# Optimization Setup
avg_returns = returns.mean().values
cov_mat = returns.cov().values

weights = cp.Variable(len(selected_assets))
gamma_par = cp.Parameter(nonneg=True)

portf_rtn_cvx = avg_returns @ weights
portf_vol_cvx = cp.quad_form(weights, cov_mat)
objective_function = cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx)
problem = cp.Problem(objective_function, [cp.sum(weights) == 1, weights >= 0])

# Efficient Frontier Calculation
N_POINTS = 25
portf_rtn_cvx_ef = []
portf_vol_cvx_ef = []
gamma_range = np.logspace(-3, 3, num=N_POINTS)

for gamma in gamma_range:
    gamma_par.value = gamma_value
    problem.solve()
    portf_vol_cvx_ef.append(np.sqrt(portf_vol_cvx.value))
    portf_rtn_cvx_ef.append(portf_rtn_cvx.value)

# Convert data for Plotly
df_ef = pd.DataFrame({"Volatility": portf_vol_cvx_ef, "Return": portf_rtn_cvx_ef, "Gamma": np.round(gamma_range, 3)})

# Efficient Frontier Plot
st.subheader("Efficient Frontier (Interactive)")
fig = px.scatter(df_ef, x="Volatility", y="Return", color="Gamma",
                 hover_data={"Gamma": True}, title="Efficient Frontier")

# Add asset points
for i, asset in enumerate(selected_assets):
    fig.add_trace(go.Scatter(
        x=[np.sqrt(cov_mat[i, i])], y=[avg_returns[i]],
        mode="markers", marker=dict(size=10, symbol="circle"),
        name=asset
    ))

st.plotly_chart(fig, use_container_width=True)

# Portfolio Optimization with Leverage
max_leverage = cp.Parameter()
prob_with_leverage = cp.Problem(objective_function, [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage])

portf_vol_l = np.zeros((N_POINTS, 1))
portf_rtn_l = np.zeros((N_POINTS, 1))

for gamma_ind in range(N_POINTS):
    max_leverage.value = max_leverage_value
    gamma_par.value = gamma_range[gamma_ind]
    prob_with_leverage.solve()
    portf_vol_l[gamma_ind, 0] = np.sqrt(portf_vol_cvx.value)
    portf_rtn_l[gamma_ind, 0] = portf_rtn_cvx.value

df_leverage = pd.DataFrame({"Volatility": portf_vol_l[:, 0], "Return": portf_rtn_l[:, 0], "Leverage": max_leverage_value})

st.subheader("Efficient Frontier with Leverage")
fig_leverage = px.scatter(df_leverage, x="Volatility", y="Return", color="Leverage",
                          title="Efficient Frontier with Leverage", hover_data={"Leverage": True})
st.plotly_chart(fig_leverage, use_container_width=True)

# Weight Allocation Table
st.subheader("Optimal Portfolio Weights")
opt_weights = pd.DataFrame(weights.value, index=selected_assets, columns=["Weight"])
st.write(opt_weights.style.format("{:.2%}"))

# Final Remarks
st.markdown("""
## Project Overview  
This project explores **Modern Portfolio Theory (MPT)** and **Efficient Frontier Calculation**.  
We optimize a stock portfolio based on historical returns and volatility, incorporating:
- Equal-weighted baseline portfolios
- Risk aversion sensitivity analysis (Gamma parameter)
- **Interactive efficient frontier visualization**
- Effects of leverage on asset allocation

The **goal** is to find optimal portfolio allocations that maximize returns **for a given level of risk**.
""")
