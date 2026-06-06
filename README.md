# Market Finance — Quantitative Projects
**Paul Desferet · ESCP Business School · [LinkedIn](https://www.linkedin.com/)**

A portfolio of quantitative finance projects covering derivatives pricing, portfolio construction,
and systematic trading signals — built with Python, real market data (yfinance), and standard
quant finance libraries (NumPy, SciPy, pandas, matplotlib).

---

## Projects

### [01 · Monte Carlo Simulation & Options Pricing](01_monte_carlo_options/monte_carlo_options.ipynb)
Prices European options via Geometric Brownian Motion (GBM) Monte Carlo simulation and benchmarks
against the Black-Scholes closed-form formula. Covers GBM path generation, terminal price distribution,
VaR/CVaR from simulated paths, option Greeks (Δ, Γ, ν, θ), and convergence analysis.

**Stack:** `numpy` `scipy.stats` `matplotlib` `yfinance`

---

### [02 · Portfolio Optimization & Efficient Frontier](02_portfolio_optimization/portfolio_optimization.ipynb)
Builds a mean-variance optimized multi-asset portfolio using the Markowitz framework on real equity data.
Plots the full efficient frontier (20 000 random portfolios), computes the **Maximum Sharpe** and
**Minimum Variance** portfolios via `scipy.optimize`, and estimates historical VaR and CVaR.

**Stack:** `numpy` `scipy.optimize` `pandas` `matplotlib` `yfinance`

---

### [03 · Technical Analysis & Multi-Factor Trading Signals](03_trading_signals/technical_signals.ipynb)
Implements SMA 30/90, RSI 14, Bollinger Bands, and MACD on real equity data, then aggregates them into
a scored **Buy / Hold / Sell** signal. Includes a simple long-only backtest benchmarked against
buy-and-hold with Sharpe ratio, annualised return, volatility, and max drawdown.

**Stack:** `numpy` `pandas` `matplotlib` `yfinance`

---

### [05 · Options Pricing — OOP Framework](05_options_pricing_oop/options_pricing_oop.ipynb)
Builds an object-oriented options pricing library from scratch: `Option` abstract base class,
`EuropeanOption` (Black-Scholes + all 5 Greeks), `AmericanOption` (CRR binomial tree, 500 steps),
and `OptionPortfolio` for multi-leg strategies. Covers Greeks surfaces (3D), early exercise premium,
straddle/strangle/collar payoff diagrams, and a Newton-Raphson implied volatility solver.

**Stack:** `numpy` `scipy.stats` `matplotlib` (Python OOP — `ABC`, `abstractmethod`, properties)

---

### [06 · FX Analysis & Carry Trade Strategy](06_fx_carry_trade/fx_carry_trade.ipynb)
Analyses 6 major FX pairs (EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CHF, NZD/USD) and implements
a systematic carry trade strategy based on central bank rate differentials. Reports Sharpe, max
drawdown, Calmar ratio, and rolling 90-day correlation with S&P 500 as a risk-on/risk-off proxy.

**Stack:** `numpy` `pandas` `matplotlib` `yfinance`

---

## Upcoming Projects
| # | Topic |
|---|-------|
| 04 | Volatility Analysis — rolling vol, GARCH(1,1), implied vol skew |

---

## Flagship Application
→ **[Market Intelligence App](https://github.com/pauldesferet/shared2)** — a full-stack market
analytics platform (FastAPI + Streamlit) covering equities, FX, indices, portfolio analytics,
Monte Carlo, efficient frontier, trading signals, crypto, and newsletter.

---

## Setup

```bash
pip install yfinance numpy pandas scipy matplotlib
jupyter notebook
```

Python 3.10+ recommended.
