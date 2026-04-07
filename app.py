"""
Portfolio Optimizer – Max Sharpe · Min Volatility
Built with Streamlit · yfinance · PyPortfolioOpt · Plotly
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting as pf_plotting
import warnings
import datetime

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer · PyPortfolioOpt",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #f8fafc;
    color: #1e293b;
}

#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #475569 !important; font-size: 0.83rem; }
[data-testid="stSidebar"] h2 { color: #1e293b !important; font-size: 1rem; font-weight: 700; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #64748b !important;
    font-size: 0.7rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem !important;
}

/* Button */
.stButton > button {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 0.88rem;
    width: 100%;
    transition: background-color 0.2s;
}
.stButton > button:hover { background-color: #1d4ed8; }

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.5rem;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.1;
}
.metric-value.positive { color: #16a34a; }
.metric-value.neutral  { color: #2563eb; }
.metric-value.accent   { color: #7c3aed; }

/* Section headers */
.section-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #e2e8f0;
}

/* Hero banner */
.hero {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 0.3rem;
}
.hero p { color: #64748b; font-size: 0.875rem; margin: 0; }

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #e2e8f0 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9;
    border-radius: 6px;
    padding: 3px;
    gap: 3px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-weight: 500;
    border-radius: 5px;
    padding: 0.45rem 1.1rem;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
}

/* Objective badge */
.obj-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-left: 0.5rem;
    vertical-align: middle;
}
.obj-badge.sharpe { background: #dbeafe; color: #1d4ed8; border: 1px solid #bfdbfe; }
.obj-badge.minvol { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }

/* Alerts */
.stAlert { border-radius: 6px !important; }

/* Radio */
[data-testid="stSidebar"] .stRadio > div { gap: 0.3rem; }
[data-testid="stSidebar"] .stRadio label {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 0.4rem 0.75rem !important;
    cursor: pointer;
    font-size: 0.82rem !important;
    color: #475569 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    border-color: #2563eb;
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: list[str], years: int = 3) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers."""
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=years * 365)
    raw   = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    prices.dropna(how="all", inplace=True)
    return prices


@st.cache_data(show_spinner=False)
def validate_tickers(tickers: list[str]) -> tuple[list[str], list[str]]:
    """Return (valid_tickers, invalid_tickers) by checking yfinance fast_info."""
    valid, invalid = [], []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            # fast_info has a 'last_price' that is None for unknown symbols
            price = getattr(info, "last_price", None)
            if price is not None and price > 0:
                valid.append(t)
            else:
                invalid.append(t)
        except Exception:
            invalid.append(t)
    return valid, invalid


def run_optimization(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.02,
    objective: str = "Max Sharpe Ratio",
    max_weight: float = 1.0,
    target_return: float | None = None,
):
    """
    Compute Mu, Sigma and run the chosen optimisation objective.
    Objectives: 'Max Sharpe Ratio', 'Minimum Volatility', 'Target Return'
    When objective == 'Target Return', target_return (decimal) is required.
    """
    mu    = expected_returns.mean_historical_return(prices)
    sigma = risk_models.sample_cov(prices)
    ef    = EfficientFrontier(mu, sigma, weight_bounds=(0, max_weight))
    if objective == "Max Sharpe Ratio":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif objective == "Target Return" and target_return is not None:
        # Clamp to feasible range
        ef_mv = EfficientFrontier(mu, sigma, weight_bounds=(0, max_weight))
        ef_mv.min_volatility()
        min_ret, _, _ = ef_mv.portfolio_performance(verbose=False)
        max_ret = float(mu.max())
        clamped = float(np.clip(target_return, min_ret + 1e-4, max_ret * 0.999))
        ef.efficient_return(target_return=clamped)
    else:  # Minimum Volatility (default fallback)
        ef.min_volatility()
    weights_raw = ef.clean_weights()
    perf        = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    return weights_raw, perf, mu, sigma


def build_weights_df(weights_raw: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(weights_raw, orient="index", columns=["Weight"])
    df = df[df["Weight"] > 1e-4].sort_values("Weight", ascending=False)
    df.index.name = "Ticker"
    df["Weight (%)"] = (df["Weight"] * 100).round(2)
    df.drop(columns="Weight", inplace=True)
    df.reset_index(inplace=True)
    return df


def weights_bar_chart(weights_raw: dict) -> go.Figure:
    """Horizontal bar chart of portfolio weights – sorted ascending for readability."""
    active = {k: v for k, v in weights_raw.items() if v > 1e-4}
    tickers = list(active.keys())
    values  = [v * 100 for v in active.values()]
    # Sort ascending so largest bar is at top
    pairs   = sorted(zip(values, tickers))
    values, tickers = zip(*pairs) if pairs else ([], [])

    palette = ["#2563eb","#4f46e5","#7c3aed","#a855f7","#ec4899",
               "#06b6d4","#10b981","#f59e0b","#ef4444","#84cc16",
               "#0ea5e9","#f97316","#14b8a6","#8b5cf6","#fb7185"]
    bar_colors = [palette[i % len(palette)] for i in range(len(tickers))]

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(tickers),
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="Inter", size=12, color="#334155"),
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        width=0.6,
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor ="#f8fafc",
        font=dict(family="Inter", color="#334155"),
        xaxis=dict(
            title="Portfolio Weight (%)",
            showgrid=True, gridcolor="#e2e8f0",
            zeroline=False, tickfont=dict(size=11),
            range=[0, max(values) * 1.22] if values else [0, 100],
        ),
        yaxis=dict(
            tickfont=dict(size=12, family="Inter", color="#334155"),
            showgrid=False,
        ),
        margin=dict(l=10, r=60, t=10, b=10),
        bargap=0.25,
    )
    return fig


def pie_chart(weights_raw: dict) -> go.Figure:
    labels  = list(weights_raw.keys())
    values  = list(weights_raw.values())
    palette = ["#2563eb","#4f46e5","#7c3aed","#a855f7","#ec4899",
               "#06b6d4","#10b981","#f59e0b","#ef4444","#84cc16",
               "#0ea5e9","#f97316","#14b8a6","#8b5cf6","#fb7185"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=palette[:len(labels)], line=dict(color="#0a0e1a", width=2)),
        textinfo="label+percent",
        textfont=dict(family="Inter", size=12, color="#e2e8f0"),
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor ="#f8fafc",
        font=dict(family="Inter", color="#334155"),
        legend=dict(
            orientation="v", x=1.05, y=0.5,
            font=dict(color="#94a3b8", size=11),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=10,r=10,t=10,b=10),
        showlegend=True,
        annotations=[dict(
            text="<b>Portfolio</b>", x=0.5, y=0.5,
            font=dict(size=13, color="#e2e8f0", family="Space Grotesk"),
            showarrow=False,
        )],
    )
    return fig


def correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    corr = prices.pct_change().dropna().corr()
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[
            [0.0,  "#1e1b4b"],
            [0.25, "#2563eb"],
            [0.5,  "#0ea5e9"],
            [0.75, "#34d399"],
            [1.0,  "#a78bfa"],
        ],
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#e2e8f0"),
        hovertemplate="<b>%{x} vs %{y}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor ="#f8fafc",
        font=dict(family="Inter", color="#334155"),
        xaxis=dict(tickfont=dict(size=11), gridcolor="rgba(99,179,237,0.05)"),
        yaxis=dict(tickfont=dict(size=11), gridcolor="rgba(99,179,237,0.05)"),
        margin=dict(l=10,r=10,t=10,b=10),
    )
    return fig


def price_history_chart(prices: pd.DataFrame) -> go.Figure:
    """Normalised cumulative returns (rebased to 100)."""
    rebased = (prices / prices.iloc[0]) * 100
    palette = ["#2563eb","#4f46e5","#7c3aed","#a855f7","#ec4899",
               "#06b6d4","#10b981","#f59e0b","#ef4444","#84cc16",
               "#0ea5e9","#f97316","#14b8a6","#8b5cf6","#fb7185"]
    fig = go.Figure()
    for i, col in enumerate(rebased.columns):
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased[col],
            name=col,
            mode="lines",
            line=dict(color=palette[i % len(palette)], width=1.8),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %d, %Y}}<br>Value: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor ="#f8fafc",
        font=dict(family="Inter", color="#334155"),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                   tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                   tickfont=dict(size=11), title="Rebased Value (100 = Start)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"),
        hovermode="x unified",
        margin=dict(l=10,r=10,t=40,b=10),
    )
    return fig


def efficient_frontier_chart(
    prices: pd.DataFrame,
    rf: float = 0.05,
    max_weight: float = 1.0,
    objective: str = "Max Sharpe Ratio",
) -> go.Figure:
    """
    Plot the Efficient Frontier with:
      • Monte-Carlo backdrop (1,000 random portfolios, colour-coded by Sharpe)
      • True Efficient Frontier curve (50-point return sweep)
      • Max Sharpe Ratio landmark (⭐ amber)
      • Minimum Risk (Min Volatility) landmark (◆ green)
    """
    mu    = expected_returns.mean_historical_return(prices)
    sigma = risk_models.sample_cov(prices)

    # ── 1. Monte-Carlo random portfolios ──
    n_assets = len(mu)
    n_sim    = 1000
    np.random.seed(42)
    sim_rets, sim_vols, sim_sharpe = [], [], []
    for _ in range(n_sim):
        w  = np.random.dirichlet(np.ones(n_assets))
        w  = np.clip(w, 0, max_weight)
        w /= w.sum()
        r  = float(w @ mu.values)
        v  = float(np.sqrt(w @ sigma.values @ w))
        sim_rets.append(r)
        sim_vols.append(v)
        sim_sharpe.append((r - rf) / v)

    # ── 2. Minimum Volatility landmark ──
    ef_minvol = EfficientFrontier(mu, sigma, weight_bounds=(0, max_weight))
    ef_minvol.min_volatility()
    minvol_ret, minvol_vol, minvol_sr = ef_minvol.portfolio_performance(
        verbose=False, risk_free_rate=rf
    )

    # ── 3. Max Sharpe landmark ──
    ef_ms = EfficientFrontier(mu, sigma, weight_bounds=(0, max_weight))
    ef_ms.max_sharpe(risk_free_rate=rf)
    ms_ret, ms_vol, ms_sr = ef_ms.portfolio_performance(
        verbose=False, risk_free_rate=rf
    )

    # ── 4. Efficient Frontier curve (return-sweep) ──
    curve_vols, curve_rets = [], []
    # sweep target returns from just above min-vol return to mu.max()
    ret_lo  = minvol_ret + 1e-4
    ret_hi  = float(mu.max()) * 0.999
    n_curve = 60
    for target_ret in np.linspace(ret_lo, ret_hi, n_curve):
        try:
            ef_c = EfficientFrontier(mu, sigma, weight_bounds=(0, max_weight))
            ef_c.efficient_return(target_return=target_ret)
            r_c, v_c, _ = ef_c.portfolio_performance(verbose=False, risk_free_rate=rf)
            curve_rets.append(r_c)
            curve_vols.append(v_c)
        except Exception:
            pass  # infeasible at this target – skip

    # Sort curve points by volatility so the line doesn't zig-zag
    if curve_vols:
        paired = sorted(zip(curve_vols, curve_rets))
        curve_vols, curve_rets = zip(*paired)

    # ── 5. Build figure ──
    fig = go.Figure()

    # ── 5a. Monte-Carlo scatter (behind everything) ──
    fig.add_trace(go.Scatter(
        x=sim_vols, y=sim_rets,
        mode="markers",
        marker=dict(
            size=5, opacity=0.45,
            color=sim_sharpe,
            colorscale=[
                [0.0,  "#1e1b4b"],
                [0.35, "#2563eb"],
                [0.7,  "#7c3aed"],
                [1.0,  "#a78bfa"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Sharpe Ratio", font=dict(color="#334155", size=11)),
                tickfont=dict(color="#334155", size=10),
                thickness=14, len=0.55, x=1.01,
                outlinewidth=0,
            ),
        ),
        name="Random Portfolios (1,000 simulated)",
        hovertemplate=(
            "<b>Random Portfolio</b><br>"
            "Volatility (Risk): %{x:.2%}<br>"
            "Expected Return: %{y:.2%}<extra></extra>"
        ),
    ))

    # ── 5b. Efficient Frontier curve ──
    if curve_vols:
        fig.add_trace(go.Scatter(
            x=curve_vols, y=curve_rets,
            mode="lines",
            line=dict(color="#1e293b", width=2.5, dash="solid"),
            name="Efficient Frontier",
            hovertemplate=(
                "<b>Efficient Portfolio</b><br>"
                "Volatility (Risk): %{x:.2%}<br>"
                "Expected Return: %{y:.2%}<extra></extra>"
            ),
        ))

    # ── 5c. Minimum Risk Portfolio (green diamond) ──
    fig.add_trace(go.Scatter(
        x=[minvol_vol], y=[minvol_ret],
        mode="markers+text",
        marker=dict(
            size=16, color="#16a34a", symbol="diamond",
            line=dict(width=2, color="#fff"),
        ),
        text=["Min Risk"],
        textposition="bottom right",
        textfont=dict(size=11, color="#15803d", family="Inter"),
        name=f"Min Risk  (SR={minvol_sr:.2f})",
        hovertemplate=(
            f"<b>Minimum Risk Portfolio</b><br>"
            f"Volatility (Risk): {minvol_vol:.2%}<br>"
            f"Expected Return: {minvol_ret:.2%}<br>"
            f"Sharpe Ratio: {minvol_sr:.2f}<extra></extra>"
        ),
    ))

    # ── 5d. Max Sharpe Portfolio (amber star) ──
    ms_active = (objective == "Max Sharpe Ratio")
    fig.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_ret],
        mode="markers+text",
        marker=dict(
            size=20, color="#d97706", symbol="star",
            line=dict(width=2, color="#fff"),
        ),
        text=["Max Sharpe"],
        textposition="top right",
        textfont=dict(size=11, color="#b45309", family="Inter"),
        name=f"Max Sharpe  (SR={ms_sr:.2f})",
        hovertemplate=(
            f"<b>Max Sharpe Portfolio</b><br>"
            f"Volatility (Risk): {ms_vol:.2%}<br>"
            f"Expected Return: {ms_ret:.2%}<br>"
            f"Sharpe Ratio: {ms_sr:.2f}<extra></extra>"
        ),
    ))

    # ── 6. Layout ──
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor ="#f8fafc",
        font=dict(family="Inter", color="#334155"),
        xaxis=dict(
            title=dict(text="Volatility (Risk)", font=dict(size=13, color="#334155")),
            tickformat=".0%",
            showgrid=True, gridcolor="#e2e8f0",
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text="Expected Return", font=dict(size=13, color="#334155")),
            tickformat=".0%",
            showgrid=True, gridcolor="#e2e8f0",
            zeroline=False,
            tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11), bgcolor="rgba(255,255,255,0.8)",
        ),
        hovermode="closest",
        margin=dict(l=10, r=60, t=50, b=10),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    # ── Universe ──
    st.markdown("### Ticker Universe")
    default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, V, UNH, BRK-B, XOM"
    raw_input = st.text_area(
        "Enter 5–15 Stock Tickers (comma-separated)",
        value=default_tickers,
        height=110,
        help="e.g. AAPL, MSFT, TSLA, AMZN, GOOGL — invalid tickers are automatically detected and skipped.",
    )

    history_years = st.slider(
        "Historical Window (years)", min_value=1, max_value=5, value=3,
        help="How many years of daily data to fetch from Yahoo Finance.",
    )

    st.markdown("---")

    # ── Portfolio Strategy ──
    st.markdown("### Portfolio Strategy")
    objective = st.radio(
        "Optimisation Objective",
        options=["Max Sharpe Ratio", "Minimum Volatility", "Target Return"],
        index=0,
        help=(
            "**Max Sharpe** – maximises risk-adjusted return.  \n"
            "**Min Volatility** – finds the least-risky feasible portfolio.  \n"
            "**Target Return** – minimises risk for a user-specified annual return."
        ),
        label_visibility="collapsed",
    )

    target_return: float | None = None
    target_return_pct: float = 15.0  # default; becomes live widget when objective == "Target Return"
    if objective == "Target Return":
        target_return_pct = st.number_input(
            "Target Annual Return (%)",
            min_value=1.0,
            max_value=200.0,
            value=15.0,
            step=0.5,
            help="The portfolio will be constructed to achieve this return at the lowest possible volatility.",
        )
        target_return = target_return_pct / 100.0

    st.markdown("---")

    # ── Risk Parameters ──
    st.markdown("### Risk Parameters")
    risk_free = st.slider(
        "Risk-Free Rate (%)", min_value=0.0, max_value=8.0, value=2.0, step=0.25,
        help="Annualised U.S. 10-yr Treasury yield. Used to calculate the Sharpe Ratio. Default: 2% (long-run average).",
    ) / 100.0

    max_weight_pct = st.slider(
        "Max Weight per Stock (%)", min_value=5, max_value=100, value=100, step=5,
        help="Caps the allocation to any single asset. Lower values enforce diversification.",
    )
    max_weight = max_weight_pct / 100.0

    st.markdown("---")
    run_btn = st.button("Optimize Portfolio")

    st.markdown("---")
    _solver_map = {
        "Max Sharpe Ratio":   "SLSQP · Max Sharpe",
        "Minimum Volatility": "SLSQP · Min Volatility",
        "Target Return":      "SLSQP · Efficient Return",
    }
    solver_name = _solver_map.get(objective, "SLSQP")
    _tr_line = f"• Target return: {target_return_pct:.1f}%<br>" if objective == "Target Return" else ""
    st.markdown(f"""
    <div style='color:#475569;font-size:0.73rem;line-height:1.6'>
    <b style='color:#64748b'>Methodology</b><br>
    • Returns: Mean Historical Return<br>
    • Risk: Sample Covariance Matrix<br>
    • Solver: {solver_name}<br>
    {_tr_line}• Max weight / asset: {max_weight_pct}%<br>
    • Risk-free rate: {risk_free*100:.2f}%<br>
    • Library: PyPortfolioOpt 1.6
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
_badge_map = {
    "Max Sharpe Ratio":   ("sharpe", "Max Sharpe"),
    "Minimum Volatility": ("minvol", "Min Volatility"),
    "Target Return":      ("minvol", f"Target {target_return_pct:.1f}%" if objective == "Target Return" else "Target Return"),
}
_badge_cls, _badge_text = _badge_map.get(objective, ("sharpe", objective))
_constraint_desc = (
    f"Target return: <strong>{target_return_pct:.1f}%</strong> · "
    if objective == "Target Return" else ""
)
st.markdown(f"""
<div class="hero">
  <h1>Portfolio Optimizer
    <span class="obj-badge {_badge_cls}">{_badge_text}</span>
  </h1>
  <p>Objective: <strong>{objective}</strong> · {_constraint_desc}Max <strong>{max_weight_pct}%</strong> per asset
     · RF rate <strong>{risk_free*100:.2f}%</strong>
     · Powered by <strong>PyPortfolioOpt</strong> &amp; <strong>yfinance</strong>.<br>
     Adjust any parameter in the sidebar and click <em>Optimize Portfolio</em>.</p>
</div>
""", unsafe_allow_html=True)

# ── Main Logic ────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("Configure your universe in the sidebar, then click **Optimize Portfolio** to begin.", )
    st.stop()

# ── Parse & validate tickers ─────────────────────────────────────────────────
tickers_raw = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
if len(tickers_raw) < 5:
    st.error("Please enter **at least 5** tickers."); st.stop()
if len(tickers_raw) > 15:
    st.warning(f"You entered {len(tickers_raw)} tickers — truncating to the first 15.")
    tickers_raw = tickers_raw[:15]

# Step 1 – lightweight symbol validation (no full download yet)
with st.spinner(f"Validating {len(tickers_raw)} ticker symbols…"):
    valid_symbols, invalid_symbols = validate_tickers(tickers_raw)

if invalid_symbols:
    st.warning(
        f"Warning: **{len(invalid_symbols)} unrecognised ticker(s) skipped:** "
        f"`{'`, `'.join(invalid_symbols)}` — "
        "check spelling or try the full exchange suffix (e.g. `BRK-B`, `GOOGL`)."
    )

if len(valid_symbols) < 5:
    st.error(
        f"Only **{len(valid_symbols)} valid ticker(s)** found. "
        "Please add more recognisable symbols to run the optimiser."
    )
    st.stop()

tickers = valid_symbols

# Step 2 – download price history for validated tickers only
with st.spinner(f"Fetching {history_years}yr price history for {len(tickers)} ticker(s)…"):
    try:
        prices = fetch_prices(tickers, years=history_years)
    except Exception as e:
        st.error(f"Data fetch failed: {e}"); st.stop()

# Step 3 – drop any tickers that came back with no usable rows
valid = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 60]
if len(valid) < 5:
    st.error("Fewer than 5 tickers returned sufficient data. Check your ticker symbols."); st.stop()
if len(valid) < len(tickers):
    dropped = set(tickers) - set(valid)
    st.warning(f"Dropped tickers with insufficient price history: {', '.join(sorted(dropped))}")
prices = prices[valid]

# ── Optimisation ─────────────────────────────────────────────────────────────
_spin_map = {
    "Max Sharpe Ratio":   "Running Maximum Sharpe Ratio optimisation…",
    "Minimum Volatility": "Running Minimum Volatility optimisation…",
    "Target Return":      f"Running Efficient Return optimisation (target {target_return_pct:.1f}%)…",
}
_spin_label = _spin_map.get(objective, "Running optimisation…")
with st.spinner(_spin_label):
    try:
        weights_raw, (exp_ret, ann_vol, sharpe), mu, sigma = run_optimization(
            prices,
            risk_free_rate=risk_free,
            objective=objective,
            max_weight=max_weight,
            target_return=target_return,
        )
    except Exception as e:
        st.error(
            f"**Optimisation failed:** {e}\n\n"
            "_Tips: (1) Relax the Max Weight constraint — a very tight cap can be infeasible.  "
            "(2) If using Target Return, ensure your target is between the min-volatility return "
            "and the maximum individual asset return._"
        )
        st.stop()

weights_df = build_weights_df(weights_raw)

# ── Performance Summary ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
_sharpe_color = "positive" if sharpe >= 1.0 else ("neutral" if sharpe >= 0.5 else "accent")
_ret_vs_rf    = exp_ret - risk_free
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Expected Annual Return</div>
    <div class="metric-value positive">{exp_ret*100:.2f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Annual Volatility (Risk)</div>
    <div class="metric-value neutral">{ann_vol*100:.2f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Sharpe Ratio (rf={risk_free*100:.2f}%)</div>
    <div class="metric-value {_sharpe_color}">{sharpe:.3f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Excess Return over RF</div>
    <div class="metric-value positive">{_ret_vs_rf*100:+.2f}%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Assets in Universe</div>
    <div class="metric-value">{len(valid)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Active Positions</div>
    <div class="metric-value">{len(weights_df)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Weights", "Efficient Frontier", "Correlations", "Price History"])

# ── Tab 1 – Weights ───────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1.1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Optimal Portfolio Weights</div>', unsafe_allow_html=True)

        # Styled table
        styled = weights_df.style \
            .format({"Weight (%)": "{:.2f}%"}) \
            .bar(subset=["Weight (%)"], color=["#2563eb22", "#2563eb"], vmin=0, vmax=100) \
            .set_properties(**{"font-size": "0.85rem"})

        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Full weights table (incl. zeroes)
        with st.expander("Show all tickers (incl. zero-weight)"):
            all_df = pd.DataFrame.from_dict(weights_raw, orient="index", columns=["Weight (%)"])
            all_df["Weight (%)"] = (all_df["Weight (%)"] * 100).round(4)
            all_df.index.name = "Ticker"
            st.dataframe(all_df.reset_index(), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<div class="section-header">Allocation Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(pie_chart(weights_raw), use_container_width=True, config={"displayModeBar": False})

    # ── Horizontal bar chart — weight distribution ──
    st.markdown('<div class="section-header">Weight Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(
        weights_bar_chart(weights_raw),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Individual stock expected returns table
    st.markdown('<div class="section-header">Individual Stock Expected Returns &amp; Volatility</div>', unsafe_allow_html=True)
    ind_vols = np.sqrt(np.diag(sigma.values))
    ind_df   = pd.DataFrame({
        "Ticker":              mu.index.tolist(),
        "Exp. Annual Return":  (mu.values * 100).round(2),
        "Annual Volatility":   (ind_vols * 100).round(2),
        "Return/Risk Ratio":   (mu.values / ind_vols).round(3),
    })
    ind_df.sort_values("Exp. Annual Return", ascending=False, inplace=True)
    ind_df["Exp. Annual Return"] = ind_df["Exp. Annual Return"].apply(lambda x: f"{x:.2f}%")
    ind_df["Annual Volatility"]  = ind_df["Annual Volatility"].apply(lambda x: f"{x:.2f}%")
    st.dataframe(ind_df, use_container_width=True, hide_index=True)

# ── Tab 2 – Efficient Frontier ────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="section-header">'
        'Efficient Frontier — 1,000 Simulated Portfolios + True Frontier Curve'
        '</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        efficient_frontier_chart(prices, rf=risk_free, max_weight=max_weight, objective=objective),
        use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    st.caption(
        f"[Star] = Max Sharpe Ratio portfolio  |  [Diamond] = Minimum Risk portfolio  |  "
        f"\u2014 = True Efficient Frontier curve  |  "
        f"Dots = 1,000 random long-only portfolios (max {max_weight_pct}% per asset, colour = Sharpe Ratio). "
        f"Hover over any point to see exact Return / Risk values."
    )
    st.markdown("""
    <div style='
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 4px solid #0284c7;
        border-radius: 6px;
        padding: 1rem 1.3rem;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.75;
    '>
    <strong style='color:#0369a1;font-size:0.9rem;'>What is the Efficient Frontier?</strong><br>
    The <strong>Efficient Frontier</strong> (dark curve) represents the set of
    <em>optimal</em> portfolios that offer the <strong>highest possible expected
    return for every level of risk</strong>. Any portfolio lying <em>below or to the right</em> of this
    curve is sub-optimal — it either takes on unnecessary risk or sacrifices return.
    The two key landmark portfolios are:<br><br>
    &nbsp;&nbsp;<strong style='color:#d97706;'>[Star] Max Sharpe Ratio</strong>
    — maximises <em>return per unit of risk</em>, making it the best risk-adjusted portfolio
    for most investors.<br>
    &nbsp;&nbsp;<strong style='color:#16a34a;'>[Diamond] Minimum Risk</strong>
    — the portfolio with the lowest possible volatility, regardless of return.
    Preferred by highly risk-averse investors.
    </div>
    """, unsafe_allow_html=True)

# ── Tab 3 – Correlations ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Pairwise Correlation Matrix (Daily Returns)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(correlation_heatmap(prices), use_container_width=True, config={"displayModeBar": False})
    st.caption("Values close to +1 indicate highly correlated assets; values near 0 or negative reduce portfolio risk.")

# ── Tab 4 – Price History ─────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Normalised Price History (Rebased to 100)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(price_history_chart(prices), use_container_width=True, config={"displayModeBar": True})
    st.caption(f"All tickers rebased to 100 at the start of the {history_years}-year window for comparability.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.75rem;padding:0.5rem'>
  Portfolio Optimizer · Data via <strong>yfinance</strong> · Optimisation via <strong>PyPortfolioOpt</strong> ·
  For educational purposes only – not financial advice.
</div>
""", unsafe_allow_html=True)
