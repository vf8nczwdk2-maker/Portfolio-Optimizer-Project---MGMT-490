# DRIVER Documentation — Portfolio Optimizer

> **DRIVER** = **D**iscover · **R**epresent · **I**mplement · **V**alidate · **E**volve · **R**eflect

---

## Purpose

This application is a **Portfolio Optimizer** built with Python and Streamlit. It allows users to:

- Input a universe of **5–15 stock tickers** and fetch **up to 5 years** of daily price data via `yfinance`
- Compute **expected annual returns** and a **sample covariance risk matrix** using `PyPortfolioOpt`
- Solve for the **optimal long-only portfolio** under three strategies:
  - **Maximum Sharpe Ratio** — best risk-adjusted return
  - **Minimum Volatility** — lowest-risk feasible portfolio
  - **Target Return** — minimum risk for a user-specified annual return target
- Enforce a **per-asset weight cap** to control concentration
- Visualise results through an **Efficient Frontier**, a **weight bar chart**, a **correlation heatmap**, and a **normalised price history chart**

---

## Installation

### Prerequisites

- Python 3.11 or higher
- `pip` (or a virtual environment manager such as `venv` / `conda`)

### Steps

```bash
# 1. Clone or download the project
cd portfolio-optimizer

# 2. (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
streamlit>=1.32.0
yfinance>=0.2.38
pyportfolioopt>=1.5.5
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
```

---

## Running the App

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

To run headlessly (e.g. on a server):

```bash
streamlit run app.py --server.headless true --server.port 8501
```

---

## Portfolio Optimisation Logic

### 1. Discover — Data Sourcing

Daily adjusted closing prices are downloaded from **Yahoo Finance** via `yfinance.download()` for the user-defined ticker universe over 1–5 years. Invalid or unrecognised ticker symbols are detected upfront using `yf.Ticker.fast_info` and surfaced as warnings before any bulk download occurs.

### 2. Represent — Return & Risk Models

| Quantity | Method | Library Call |
|---|---|---|
| Expected Annual Return (μ) | Mean historical return, annualised | `expected_returns.mean_historical_return(prices)` |
| Covariance Matrix (Σ) | Sample covariance, annualised | `risk_models.sample_cov(prices)` |

### 3. Implement — Optimisation Solver

All three objectives use `PyPortfolioOpt`'s `EfficientFrontier` class with the **SLSQP** (Sequential Least Squares Programming) solver:

```python
ef = EfficientFrontier(mu, sigma, weight_bounds=(0, max_weight))

# Objective A — Max Sharpe Ratio
ef.max_sharpe(risk_free_rate=rf)

# Objective B — Minimum Volatility
ef.min_volatility()

# Objective C — Target Return (min risk for a specific return)
ef.efficient_return(target_return=target)
```

The `weight_bounds=(0, max_weight)` constraint enforces long-only positions and a configurable per-asset concentration cap.

### 4. Validate — Efficient Frontier Curve

A 60-point return sweep via `ef.efficient_return(target)` traces the mathematically exact mean-variance frontier independently of the optimised portfolio, providing a ground-truth visual benchmark. Additionally, 1,000 Monte-Carlo randomly weighted portfolios are plotted as a backdrop.

### 5. Evolve — Interactive Controls

Users can dynamically adjust:
- Ticker universe (5–15 symbols)
- Historical window (1–5 years)
- Optimisation objective
- Target return (when using Efficient Return mode)
- Risk-free rate (for Sharpe ratio calculation)
- Max weight per stock (concentration constraint)

### 6. Reflect — Performance Summary

The **Performance Summary** panel displays:
- Expected Annual Return
- Annual Volatility (Risk)
- Sharpe Ratio (annotated with the current risk-free rate)
- Excess Return over the Risk-Free Rate

---

## Project Structure

```
portfolio-optimizer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── DRIVER.md           # This documentation file
└── AI_DISCLOSURE.md    # AI usage disclosure
```

---

## Limitations & Disclaimers

- **Mean historical return** is a simple backward-looking estimator and may not reflect future performance.
- The sample covariance matrix is sensitive to the look-back window; short windows produce noisier estimates.
- This tool is for **educational purposes only** and does not constitute financial advice.
- All data is sourced from Yahoo Finance and may be subject to delays or inaccuracies.
