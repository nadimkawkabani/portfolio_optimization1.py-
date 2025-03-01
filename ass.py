import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
import warnings

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Define assets
ASSETS = [
    "MSFT", "AAPL", "GOOGL", "AMZN", "META",    # Technology
    "NFLX", "ADSK", "INTC", "NVDA", "TSM",      # Semiconductors & Tech
    "JPM", "GS", "BAC", "C", "AXP",             # Finance
    "UNH", "LLY", "PFE", "BMY", "MRK"           # Healthcare
]

# Streamlit configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")
st.title("📈 Portfolio Optimization Dashboard")

# Sidebar Inputs
with st.sidebar:
    selected_assets = st.multiselect("Select Assets", ASSETS, default=ASSETS)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2017-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    gamma_value = st.slider("Risk Aversion (Gamma)", min_value=0.001, max_value=100.0, value=0.1, step=0.001)
    max_leverage_value = st.slider("Max Leverage", min_value=1, max_value=5, value=1, step=1)

@st.cache_data
def fetch_data(assets, start, end):
    data = yf.download(assets, start=start, end=end)
    return data["Close"].pct_change().dropna()

# Fetch data
returns = fetch_data(selected_assets, start_date, end_date)

# Portfolio Optimization
avg_returns = returns.mean().values
cov_mat = returns.cov().values
weights = cp.Variable(len(selected_assets))
gamma_par = cp.Parameter(nonneg=True)

portf_rtn = avg_returns @ weights
portf_vol = cp.quad_form(weights, cov_mat)
objective_function = cp.Maximize(portf_rtn - gamma_par * portf_vol)
problem = cp.Problem(objective_function, [cp.sum(weights) == 1, weights >= 0])

gamma_par.value = gamma_value
problem.solve()

optimized_weights = weights.value

# Efficient Frontier
N_POINTS = 25
gamma_range = np.logspace(-3, 3, num=N_POINTS)
frontier_x, frontier_y = [], []

for gamma in gamma_range:
    gamma_par.value = gamma
    problem.solve()
    frontier_x.append(np.sqrt(portf_vol.value))
    frontier_y.append(portf_rtn.value)

# Interactive Efficient Frontier Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=frontier_x, y=frontier_y, mode='lines', name='Efficient Frontier'))

# Scatter plot for assets
for i, asset in enumerate(selected_assets):
    fig.add_trace(go.Scatter(
        x=[np.sqrt(cov_mat[i, i])], y=[avg_returns[i]],
        mode='markers', marker=dict(size=10, symbol='circle'), name=asset
    ))

fig.update_layout(title='Efficient Frontier', xaxis_title='Volatility', yaxis_title='Expected Return')
st.plotly_chart(fig)

# Portfolio Allocation
allocation_df = pd.DataFrame({'Asset': selected_assets, 'Weight': optimized_weights})
st.subheader("Portfolio Allocation")
st.dataframe(allocation_df)
