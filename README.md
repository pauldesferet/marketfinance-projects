# Market Finance
**Paul Desferet - ESCP Business School

This repository has been created to show a sample of my Python and VBA projects concerning Market Finance. 
For other projects (poker, kaggle, chess), please see other repositories (if public)

to contact me: paul.desferet@edu.escp.eu
or linkedin: https://www.linkedin.com/in/paul-desferet/

you can also visit my website via: https://shared2.vercel.app/
---

## Projects

### [Project 1: Monte Carlo Simulation & Options Pricing]

---

### [Project 2: Portfolio Optimization & Efficient Frontier]

---

### [Project 03: Technical Analysis & Multi-Factor Trading Signals]

---

### [Project 04: Options Pricing — OOP Framework]

---

### [Project 05: FX Analysis & Carry Trade Strategy]

---

### [Project 06: Xiaomi Event Study — Trump Blacklist (2021)](06_xiaomi_event_study/)

Econometric event study analyzing the impact of Xiaomi's placement on the US "Communist Chinese Military Companies" list (January 14, 2021) on its stock returns.

- Event study with dummy variable & Cumulative Abnormal Return (CAR)
- OLS regressions: Xiaomi ~ HSI, BYD, CSI300
- Diagnostic tests: White, Durbin-Watson, Jarque-Bera, VIF, Huber-White HAC
- Volatility models: ARMA, EWMA (λ=0.94), GARCH(1,1)
- Data sourced automatically via `yfinance`

---

### [Project 07: CAPM & Fama-French — CAC 40](07_capm_factor_model/capm_factor_model.ipynb)

Beta, alpha de Jensen et chargements Fama-French 3 facteurs (Mkt-RF, SMB, HML) appliqués aux 28 valeurs du CAC 40, réparties en 12 secteurs. Facteurs européens quotidiens issus de la bibliothèque Ken French.

- Matrice de corrélation inter-titres et inter-secteurs (heatmaps triées par secteur)
- CAPM : beta par titre, droite de marché (SML), alpha de Jensen par secteur
- Fama-French : chargements β Mkt / β SMB / β HML, comparaison R² CAPM vs FF
- Tableau de bord 6 panneaux (beta, alpha, R², corrélation, chargements, performance)

---

## Setup

```bash
pip install yfinance numpy pandas scipy matplotlib seaborn statsmodels pandas-datareader
jupyter notebook
```

Python 3.10+ recommended.
