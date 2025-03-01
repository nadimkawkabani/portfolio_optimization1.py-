import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import quantstats as qs
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page layout
st.set_page_config(layout="wide")

# Sidebar - User Inputs
st.sidebar.header("Portfolio Optimization Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2017-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# Gamma and Leverage sliders
gamma_value = st.sidebar.slider("Risk Aversion (Gamma)", min_value=0.001, max_value=1000.0, value=1.0, log=True)
leverage_value = st.sidebar.slider("Max Leverage", min_value=1, max_value=5, value=2)

# Assets List
ASSETS = ["MSFT", "AAPL", "GOOGL", "AMZN", "META", "NFLX", "ADSK", "INTC", "NVDA", "TSM", "JPM", "GS", "BAC", "C", "AXP", "UNH", "LLY", "PFE", "BMY", "MRK"]
n_assets = len(ASSETS)

# Download stock data
prices_df = yf.download(ASSETS, start=start_date, end=end_date)["Close"].dropna()
returns = prices_df.pct_change().dropna()

# Portfolio Optimization
avg_returns = returns.mean().values
cov_mat = returns.cov().values

weights = cp.Variable(n_assets)
gamma_par = cp.Parameter(nonneg=True)
max_leverage = cp.Parameter()

portf_rtn = avg_returns @ weights
portf_vol = cp.quad_form(weights, cov_mat)
objective = cp.Maximize(portf_rtn - gamma_par * portf_vol)

problem = cp.Problem(objective, [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage])

# Solve for different gamma values
gamma_range = np.logspace(-3, 3, num=25)
portf_vol_l, portf_rtn_l = [], []
for gamma in gamma_range:
    gamma_par.value = gamma
    max_leverage.value = leverage_value
    problem.solve()
    portf_vol_l.append(cp.sqrt(portf_vol).value)
    portf_rtn_l.append(portf_rtn.value)

# Plot Efficient Frontier
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(portf_vol_l, portf_rtn_l, label=f"Leverage {leverage_value}", linewidth=2)
ax.set(title="Efficient Frontier", xlabel="Volatility", ylabel="Expected Returns")
ax.legend(loc="upper right")
sns.despine()
st.pyplot(fig)

# Stock Scatter Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(np.sqrt(np.diag(cov_mat)), avg_returns, color='red', label="Individual Stocks")
for i, txt in enumerate(ASSETS):
    ax.annotate(txt, (np.sqrt(cov_mat[i, i]), avg_returns[i]))
ax.set(title="Stock Risk vs Return", xlabel="Volatility", ylabel="Expected Returns")
st.pyplot(fig)
