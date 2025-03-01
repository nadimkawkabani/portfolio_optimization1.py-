import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cvxpy as cp
import yfinance as yf

st.set_page_config(layout="wide")

# Define assets
ASSETS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",    # Technology
    "NFLX", "ADSK", "INTC", "NVDA", "TSM",      # Semiconductors & Tech
    "JPM", "GS", "BAC", "C", "AXP",            # Finance
    "UNH", "LLY", "PFE", "BMY", "MRK"          # Healthcare
]
n_assets = len(ASSETS)

# Sidebar filters
st.sidebar.title("Filters")
gamma_min, gamma_max = st.sidebar.slider("Gamma Range", 0.001, 1000.0, (0.1, 10.0), step=0.1)
leverage_levels = st.sidebar.multiselect("Leverage Levels", [1, 2, 5], default=[1, 2, 5])

# Fetch stock data
prices_df = yf.download(ASSETS, start="2017-01-01", end="2023-12-31")["Close"]
returns = prices_df.pct_change().dropna()
avg_returns = returns.mean().values
cov_mat = returns.cov().values

# Portfolio optimization setup
weights = cp.Variable(n_assets)
gamma_par = cp.Parameter(nonneg=True)
max_leverage = cp.Parameter()
portf_rtn_cvx = avg_returns @ weights
portf_vol_cvx = cp.quad_form(weights, cov_mat)
objective_function = cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx)
prob_with_leverage = cp.Problem(objective_function, [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage])

# Efficient frontier calculation
N_POINTS = 25
gamma_range = np.logspace(np.log10(gamma_min), np.log10(gamma_max), num=N_POINTS)
len_leverage = len(leverage_levels)
portf_vol_l = np.zeros((N_POINTS, len_leverage))
portf_rtn_l = np.zeros((N_POINTS, len_leverage))

for lev_ind, leverage in enumerate(leverage_levels):
    for gamma_ind in range(N_POINTS):
        max_leverage.value = leverage
        gamma_par.value = gamma_range[gamma_ind]
        prob_with_leverage.solve()
        portf_vol_l[gamma_ind, lev_ind] = cp.sqrt(portf_vol_cvx).value
        portf_rtn_l[gamma_ind, lev_ind] = portf_rtn_cvx.value

# Plot efficient frontier
fig, ax = plt.subplots(figsize=(12, 6))
for leverage_index, leverage in enumerate(leverage_levels):
    ax.plot(portf_vol_l[:, leverage_index], portf_rtn_l[:, leverage_index], label=f"Leverage = {leverage}")
ax.set(title="Efficient Frontier for Different Leverage Levels", xlabel="Volatility", ylabel="Expected Returns")
ax.legend(title="Max Leverage", bbox_to_anchor=(1, 1))
sns.despine()
st.pyplot(fig)
