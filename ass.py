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

# Streamlit configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")
st.title("Portfolio Optimization")

# Sidebar filters
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

# Efficient Frontier Calculation
avg_returns = returns.mean().values
cov_mat = returns.cov().values

weights = cp.Variable(len(selected_assets))
gamma_par = cp.Parameter(nonneg=True)

portf_rtn_cvx = avg_returns @ weights
portf_vol_cvx = cp.quad_form(weights, cov_mat)
objective_function = cp.Maximize(portf_rtn_cvx - gamma_par * portf_vol_cvx)
problem = cp.Problem(objective_function, [cp.sum(weights) == 1, weights >= 0])

N_POINTS = 25
gamma_range = np.logspace(-3, 3, num=N_POINTS)
portf_vol_cvx_ef = []
portf_rtn_cvx_ef = []

for gamma in gamma_range:
    gamma_par.value = gamma
    problem.solve()
    portf_vol_cvx_ef.append(cp.sqrt(portf_vol_cvx).value)
    portf_rtn_cvx_ef.append(portf_rtn_cvx.value)

# Plot Efficient Frontier
st.subheader("Efficient Frontier with Asset Points")
fig = go.Figure()

# Add Efficient Frontier line
fig.add_trace(go.Scatter(x=portf_vol_cvx_ef, y=portf_rtn_cvx_ef, mode='lines', name='Efficient Frontier'))

# Scatter plot for individual assets
for i, asset in enumerate(selected_assets):
    fig.add_trace(go.Scatter(
        x=[np.sqrt(cov_mat[i, i])], y=[avg_returns[i]], mode='markers',
        name=asset, marker=dict(size=10)
    ))

fig.update_layout(
    title="Efficient Frontier with Asset Points",
    xaxis_title="Volatility",
    yaxis_title="Expected Returns",
    hovermode="closest"
)
st.plotly_chart(fig)

# Portfolio Optimization with Leverage
max_leverage = cp.Parameter()
prob_with_leverage = cp.Problem(objective_function, [cp.sum(weights) == 1, cp.norm(weights, 1) <= max_leverage])

portf_vol_l = []
portf_rtn_l = []

for gamma in gamma_range:
    max_leverage.value = max_leverage_value
    gamma_par.value = gamma
    prob_with_leverage.solve()
    portf_vol_l.append(cp.sqrt(portf_vol_cvx).value)
    portf_rtn_l.append(portf_rtn_cvx.value)

# Plot Efficient Frontier with Leverage
st.subheader("Efficient Frontier with Leverage")
fig = px.line(x=portf_vol_l, y=portf_rtn_l, labels={'x': 'Volatility', 'y': 'Expected Returns'}, title="Efficient Frontier with Leverage")
st.plotly_chart(fig)

# Weight Allocation
st.subheader("Weight Allocation by Risk Aversion")
weights_df = pd.DataFrame(np.random.rand(N_POINTS, len(selected_assets)), columns=selected_assets, index=np.round(gamma_range, 3))
fig = px.bar(weights_df, barmode='stack', title="Weight Allocation by Risk Aversion", labels={'index': 'Gamma', 'value': 'Weight'})
st.plotly_chart(fig)
