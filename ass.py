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
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

# Define diversified assets across industries
ASSETS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",    # Technology
    "NFLX", "ADSK", "INTC", "NVDA", "TSM",      # Semiconductors & Tech
    "JPM", "GS", "BAC", "C", "AXP",             # Finance
    "UNH", "LLY", "PFE", "BMY", "MRK"           # Healthcare
]

n_assets = len(ASSETS)

# Streamlit configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

# Streamlit Title
st.title("Portfolio Optimization")

# Place the filters in the sidebar
with st.sidebar:
    selected_assets = st.multiselect("Select Assets", ASSETS, default=ASSETS)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2017-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    gamma_value = st.slider("Risk Aversion (Gamma)", min_value=0.001, max_value=100.0, value=0.1, step=0.001)
    max_leverage_value = st.slider("Max Leverage", min_value=1, max_value=5, value=1, step=1)

# Download market data
try:
    prices_df = yf.download(selected_assets, start=start_date, end=end_date)
    returns = prices_df["Close"].pct_change().dropna()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Portfolio weights (Equal Weight Strategy)
portfolio_weights = np.array([1 / len(selected_assets)] * len(selected_assets))
portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index=returns.index)

# Plot portfolio performance
st.subheader("Portfolio Performance")
qs.plots.snapshot(portfolio_returns, title="1/n Portfolio Performance", grayscale=True)
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
weights_ef = []
gamma_range = np.logspace(-3, 3, num=N_POINTS)

for gamma in gamma_range:
    gamma_par.value = gamma_value  # Use selected gamma value
    problem.solve()
    portf_vol_cvx_ef.append(cp.sqrt(portf_vol_cvx).value)
    portf_rtn_cvx_ef.append(portf_rtn_cvx.value)
    weights_ef.append(weights.value)

weights_df = pd.DataFrame(weights_ef, columns=selected_assets, index=np.round(gamma_range, 3))

# Plot Weight Allocation with Hover Interactivity
st.subheader("Weight Allocation by Risk Aversion")
fig_bar = px.bar(
    weights_df,
    x=weights_df.index,
    y=weights_df.columns,
    labels={"x": "Risk Aversion (Gamma)", "y": "Weight"},
    title="Weights Allocation",
    barmode="stack"
)

# Customize hover data
fig_bar.update_traces(
    hovertemplate="<b>Risk Aversion (Gamma):</b> %{x}<br><b>Weight:</b> %{y:.2f}<br><b>Asset:</b> %{fullData.name}"
)

# Improve layout
fig_bar.update_layout(
    xaxis_title="Risk Aversion (Gamma)",
    yaxis_title="Weight",
    hovermode="x unified",
    legend_title="Assets",
    barmode='stack'
)

st.plotly_chart(fig_bar)

# Plot Efficient Frontier with Asset Points (Interactive)
st.subheader("Efficient Frontier with Asset Points")
fig_frontier = go.Figure()

# Plot Efficient Frontier Line
fig_frontier.add_trace(go.Scatter(
    x=portf_vol_cvx_ef,
    y=portf_rtn_cvx_ef,
    mode="lines",
    name="Efficient Frontier",
    line=dict(color="green", width=2),
    hovertemplate="<b>Volatility:</b> %{x:.2f}<br><b>Return:</b> %{y:.2f}"
))

# Scatter plot for individual assets
for asset_index in range(len(selected_assets)):
    fig_frontier.add_trace(go.Scatter(
        x=[np.sqrt(cov_mat[asset_index, asset_index])],
        y=[avg_returns[asset_index]],
        mode="markers",
        marker=dict(size=10, symbol="circle"),
        name=selected_assets[asset_index],
        hovertemplate="<b>Asset:</b> %{text}<br><b>Volatility:</b> %{x:.2f}<br><b>Return:</b> %{y:.2f}",
        text=[selected_assets[asset_index]]
    ))

# Improve layout
fig_frontier.update_layout(
    title="Efficient Frontier with Asset Points",
    xaxis_title="Volatility",
    yaxis_title="Expected Returns",
    hovermode="x unified",
    legend_title="Assets"
)

st.plotly_chart(fig_frontier)

# Portfolio Optimization with Leverage (Interactive)
max_leverage = cp.Parameter()
prob_with_leverage = cp.Problem(objective_function, [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage])

LEVERAGE_RANGE = [max_leverage_value]
len_leverage = len(LEVERAGE_RANGE)

portf_vol_l = np.zeros((N_POINTS, len_leverage))
portf_rtn_l = np.zeros((N_POINTS, len_leverage))
weights_ef = np.zeros((len_leverage, N_POINTS, len(selected_assets)))

for lev_ind, leverage in enumerate(LEVERAGE_RANGE):
    for gamma_ind in range(N_POINTS):
        max_leverage.value = leverage
        gamma_par.value = gamma_range[gamma_ind]
        prob_with_leverage.solve()
        portf_vol_l[gamma_ind, lev_ind] = cp.sqrt(portf_vol_cvx).value
        portf_rtn_l[gamma_ind, lev_ind] = portf_rtn_cvx.value
        weights_ef[lev_ind, gamma_ind, :] = weights.value

# Plot Efficient Frontier with Leverage (Interactive)
st.subheader("Efficient Frontier with Leverage")
fig_leverage = go.Figure()

for leverage_index, leverage in enumerate(LEVERAGE_RANGE):
    fig_leverage.add_trace(go.Scatter(
        x=portf_vol_l[:, leverage_index],
        y=portf_rtn_l[:, leverage_index],
        mode="lines",
        name=f"Leverage: {leverage}",
        line=dict(width=2),
        hovertemplate="<b>Leverage:</b> %{text}<br><b>Volatility:</b> %{x:.2f}<br><b>Return:</b> %{y:.2f}",
        text=[f"Leverage: {leverage}"] * len(portf_vol_l[:, leverage_index])
    ))

# Improve layout
fig_leverage.update_layout(
    title="Efficient Frontier with Leverage",
    xaxis_title="Volatility",
    yaxis_title="Expected Returns",
    hovermode="x unified",
    legend_title="Leverage"
)

st.plotly_chart(fig_leverage)
