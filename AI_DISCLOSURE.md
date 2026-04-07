# AI Usage Disclosure

## Project: Portfolio Optimizer — Maximum Sharpe Ratio & Efficient Frontier

**Course / Submission Context:** Financial Technology / Portfolio Management Application

---

## Statement of AI Assistance

This project was developed with the assistance of **Antigravity**, an AI coding assistant powered by Google DeepMind's advanced agentic coding technology.

---

## What the AI Did

The AI assistant (Antigravity) was used to accelerate the development of the following technical components:

| Component | AI Contribution |
|---|---|
| **Application boilerplate** | Generated the initial Streamlit app structure, page configuration, and CSS styling system |
| **Data pipeline** | Wrote the `yfinance` data-fetching and ticker-validation functions |
| **Optimisation logic** | Implemented the `PyPortfolioOpt` interface for Max Sharpe, Min Volatility, and Efficient Return objectives |
| **Visualisations** | Generated Plotly chart code for the Efficient Frontier, weight bar chart, correlation heatmap, and price history chart |
| **Error handling** | Implemented defensive input validation, per-ticker validity checking, and solver exception handling |
| **Documentation** | Drafted the `DRIVER.md` framework documentation |

---

## What I (the Student) Did

While the AI generated implementation code, the intellectual direction, financial strategy decisions, and result verification were performed by me:

| Responsibility | My Contribution |
|---|---|
| **Financial strategy** | Directed the choice of optimisation objectives (Max Sharpe, Min Volatility, Target Return), interpreted the economic meaning of each, and chose the risk-free rate convention |
| **Constraint design** | Defined the per-asset weight cap constraint and validated that it correctly enforces portfolio diversification |
| **Mathematical verification** | Cross-checked the Sharpe Ratio formula `(μ − rf) / σ`, validated that the Efficient Frontier curve is mathematically correct by verifying the monotonicity and shape of the risk-return tradeoff |
| **Rubric alignment** | Determined which features to build, in which order, to satisfy the project rubric requirements |
| **Output interpretation** | Reviewed all optimised weight outputs and confirmed that the results were economically reasonable (e.g., higher-Sharpe assets received larger allocations) |
| **Quality control** | Tested all three optimisation modes, edge cases (infeasible constraints, invalid tickers), and verified the UI behaviour |

---

## Summary

This project represents a **human-directed, AI-assisted** development process. The AI served as a highly capable coding partner that accelerated implementation; all financial reasoning, strategic decisions, and output verification were performed by the student.

The use of AI coding tools was authorised in accordance with the course policy on AI assistance.

---

*Disclosure prepared in accordance with academic integrity guidelines.*  
*Date: April 2026*
