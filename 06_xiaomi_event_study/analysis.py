"""
Étude d'événement : Xiaomi et la liste noire américaine (janvier 2021)
----------------------------------------------------------------------
Ce script reproduit l'analyse économétrique complète :
  1. Téléchargement des données via yfinance
  2. Statistiques descriptives et visualisation
  3. Tests de stationnarité (ADF)
  4. Modèles simples (CAPM-like) : Xiaomi ~ HSI, BYD, SHSZ
  5. Étude d'événement avec variable dummy
  6. Modèle multiple : HSI + BYD + dummy
  7. Tests : hétéroscédasticité (White), autocorrélation (DW), multicolinéarité (VIF)
  8. Correction de Huber-White (HAC)
  9. Modèles ARMA sur la volatilité
 10. EWMA et GARCH(1,1)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from arch import arch_model
from scipy import stats
from scipy.stats import jarque_bera
import os

# ── Style global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.color":       "#dee2e6",
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
os.makedirs("figures", exist_ok=True)

# ── Dates clés de l'événement ─────────────────────────────────────────────────
EVENT_DATE   = "2021-01-14"   # annonce Trump blacklist
EVENT_LIFT   = "2021-03-12"   # suspension par tribunal
START        = "2018-07-09"   # IPO Xiaomi
END          = "2026-02-20"

# ═════════════════════════════════════════════════════════════════════════════
# 1. TÉLÉCHARGEMENT DES DONNÉES
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. Téléchargement des données (yfinance)")
print("=" * 60)

TICKERS = {
    "xiaomi":  "1810.HK",
    "byd":     "1211.HK",
    "hsi":     "^HSI",
    "shsz":    "3188.HK",   # ETF CSOP CSI 300 (proxy CSI 300)
    "tesla":   "TSLA",
    "samsung": "005930.KS",
    "sp500":   "^GSPC",
    "oil":     "CL=F",
}

raw = yf.download(
    list(TICKERS.values()), start=START, end=END,
    auto_adjust=True, progress=False
)["Close"]
raw.columns = list(TICKERS.keys())

# Aligner sur les jours de cotation de Xiaomi (HK)
prices = raw.dropna(subset=["xiaomi"]).copy()
prices = prices.ffill().dropna()

print(f"  Période : {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"  Observations : {len(prices)}\n")

# Log-rendements
returns = np.log(prices / prices.shift(1)).dropna()

# ═════════════════════════════════════════════════════════════════════════════
# 2. VISUALISATION : PRIX ET RENDEMENTS
# ═════════════════════════════════════════════════════════════════════════════
print("2. Visualisation des prix et rendements")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Xiaomi (1810.HK) — Prix et Rendements", fontsize=14, fontweight="bold")

# Prix normalisés (base 100)
norm = (prices[["xiaomi", "hsi", "byd", "tesla"]] /
        prices[["xiaomi", "hsi", "byd", "tesla"]].iloc[0] * 100)
ax = axes[0, 0]
for col, color in zip(norm.columns, COLORS):
    ax.plot(norm.index, norm[col], label=col.upper(), color=color, lw=1.2)
ax.axvline(pd.Timestamp(EVENT_DATE), color="red",   ls="--", lw=1, label="Blacklist (14 jan 2021)")
ax.axvline(pd.Timestamp(EVENT_LIFT), color="green", ls="--", lw=1, label="Suspension (12 mar 2021)")
ax.set_title("Indices de prix (base 100)")
ax.legend(fontsize=7)

# Rendements Xiaomi
ax = axes[0, 1]
ax.fill_between(returns.index, returns["xiaomi"], 0,
                where=(returns["xiaomi"] >= 0), color=COLORS[2], alpha=0.5, label="Hausse")
ax.fill_between(returns.index, returns["xiaomi"], 0,
                where=(returns["xiaomi"] < 0),  color=COLORS[1], alpha=0.5, label="Baisse")
ax.axvline(pd.Timestamp(EVENT_DATE), color="red",   ls="--", lw=1)
ax.axvline(pd.Timestamp(EVENT_LIFT), color="green", ls="--", lw=1)
ax.set_title("Rendements log journaliers — Xiaomi")
ax.legend(fontsize=7)

# Distribution des rendements
ax = axes[1, 0]
ax.hist(returns["xiaomi"], bins=80, color=COLORS[0], alpha=0.7, density=True, label="Xiaomi")
x = np.linspace(returns["xiaomi"].min(), returns["xiaomi"].max(), 200)
ax.plot(x, stats.norm.pdf(x, returns["xiaomi"].mean(), returns["xiaomi"].std()),
        "r-", lw=1.5, label="Loi normale")
ax.set_title("Distribution des rendements")
ax.legend(fontsize=7)

# Volatilité réalisée (rolling 21j)
ax = axes[1, 1]
vol = returns["xiaomi"].rolling(21).std() * np.sqrt(252)
ax.plot(vol.index, vol, color=COLORS[4], lw=1)
ax.axvline(pd.Timestamp(EVENT_DATE), color="red",   ls="--", lw=1, label="Blacklist")
ax.axvline(pd.Timestamp(EVENT_LIFT), color="green", ls="--", lw=1, label="Suspension")
ax.set_title("Volatilité annualisée (fenêtre 21j)")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("figures/01_prix_rendements.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/01_prix_rendements.png\n")

# ═════════════════════════════════════════════════════════════════════════════
# 3. TESTS DE STATIONNARITÉ (ADF)
# ═════════════════════════════════════════════════════════════════════════════
print("3. Tests ADF de stationnarité")
print("-" * 45)
print(f"{'Série':<12} {'ADF stat':>10} {'p-value':>10} {'Décision':>15}")
print("-" * 45)

series_to_test = {
    "Xiaomi (prix)": prices["xiaomi"],
    "Xiaomi (ret.)": returns["xiaomi"],
    "HSI (ret.)":    returns["hsi"],
    "BYD (ret.)":    returns["byd"],
    "SHSZ (ret.)":   returns["shsz"],
}
for name, s in series_to_test.items():
    res = adfuller(s.dropna(), autolag="AIC")
    decision = "Stationnaire ✓" if res[1] < 0.05 else "Non-stat. ✗"
    print(f"{name:<20} {res[0]:>8.3f}   {res[1]:>8.4f}   {decision}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 4. MODÈLES SIMPLES (CAPM-like)
# ═════════════════════════════════════════════════════════════════════════════
print("4. Régressions simples")

def ols_summary(y, X_vars, names):
    """OLS avec affichage compact."""
    X = add_constant(X_vars)
    model = OLS(y, X).fit()
    print(f"\n  Modèle : Xiaomi ~ {' + '.join(names)}")
    print(f"  R²={model.rsquared:.4f}  F-stat={model.fvalue:.1f}  p(F)={model.f_pvalue:.4e}")
    for i, n in enumerate(["const"] + names):
        coef = model.params[i]
        pval = model.pvalues[i]
        sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"    {n:<12}  β={coef:+.4f}  p={pval:.4f} {sig}")
    return model

r = returns.copy()
m_hsi  = ols_summary(r["xiaomi"], r[["hsi"]],  ["HSI"])
m_byd  = ols_summary(r["xiaomi"], r[["byd"]],  ["BYD"])
m_shsz = ols_summary(r["xiaomi"], r[["shsz"]], ["SHSZ"])

# Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Régressions simples : Xiaomi vs indices de marché", fontweight="bold")

for ax, (col, label, model) in zip(axes, [
        ("hsi",  "HSI",  m_hsi),
        ("byd",  "BYD",  m_byd),
        ("shsz", "SHSZ", m_shsz)]):
    ax.scatter(r[col], r["xiaomi"], alpha=0.15, s=8, color=COLORS[0])
    xs = np.linspace(r[col].min(), r[col].max(), 100)
    ys = model.params[0] + model.params[1] * xs
    ax.plot(xs, ys, color=COLORS[1], lw=2)
    ax.set_xlabel(f"Rendement {label}")
    ax.set_ylabel("Rendement Xiaomi")
    ax.set_title(f"Xiaomi ~ {label}\nβ={model.params[1]:.3f}  R²={model.rsquared:.3f}")

plt.tight_layout()
plt.savefig("figures/02_regressions_simples.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → figures/02_regressions_simples.png\n")

# ═════════════════════════════════════════════════════════════════════════════
# 5. ÉTUDE D'ÉVÉNEMENT — DUMMY VARIABLE
# ═════════════════════════════════════════════════════════════════════════════
print("5. Étude d'événement avec variable dummy")

# Fenêtre d'événement : 14 jan → 12 mar 2021
r["dummy"] = ((r.index >= EVENT_DATE) & (r.index <= EVENT_LIFT)).astype(int)
print(f"  Jours dans la fenêtre d'événement : {r['dummy'].sum()}")

# Modèle avec dummy
X_event = r[["hsi", "dummy"]]
m_event = ols_summary(r["xiaomi"], X_event, ["HSI", "dummy"])

# Rendement anormal cumulé (CAR)
r_event_window = r.loc[EVENT_DATE:EVENT_LIFT].copy()
r_event_window["predicted"] = m_hsi.params[0] + m_hsi.params[1] * r_event_window["hsi"]
r_event_window["abnormal"]  = r_event_window["xiaomi"] - r_event_window["predicted"]
car = r_event_window["abnormal"].cumsum()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Étude d'événement : Xiaomi sur la liste noire américaine (jan–mar 2021)",
             fontweight="bold")

ax = axes[0]
ax.bar(r_event_window.index, r_event_window["abnormal"],
       color=[COLORS[2] if x >= 0 else COLORS[1] for x in r_event_window["abnormal"]],
       width=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Rendements anormaux journaliers")
ax.set_ylabel("Rendement anormal")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax = axes[1]
ax.plot(car.index, car.values * 100, color=COLORS[0], lw=2)
ax.fill_between(car.index, 0, car.values * 100,
                where=(car.values >= 0), alpha=0.2, color=COLORS[2])
ax.fill_between(car.index, 0, car.values * 100,
                where=(car.values < 0),  alpha=0.2, color=COLORS[1])
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"Rendement Anormal Cumulé (CAR)\nTotal = {car.iloc[-1]*100:.2f}%")
ax.set_ylabel("CAR (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("figures/03_etude_evenement.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  CAR sur la fenêtre : {car.iloc[-1]*100:.2f}%")
print("  → figures/03_etude_evenement.png\n")

# ═════════════════════════════════════════════════════════════════════════════
# 6. MODÈLE MULTIPLE : HSI + BYD + DUMMY
# ═════════════════════════════════════════════════════════════════════════════
print("6. Modèle multiple : Xiaomi ~ HSI + BYD + dummy")

X_multi = r[["hsi", "byd", "dummy"]]
m_multi = ols_summary(r["xiaomi"], X_multi, ["HSI", "BYD", "dummy"])

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = add_constant(r[["hsi", "byd", "dummy"]])
print("\n  Facteurs d'inflation de la variance (VIF) :")
for i, col in enumerate(X_vif.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_vif.values, i)
    print(f"    {col:<10}  VIF = {vif:.2f}  {'⚠️ multicolinéarité' if vif > 10 else '✓ OK'}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. TESTS ÉCONOMÉTRIQUES
# ═════════════════════════════════════════════════════════════════════════════
print("\n7. Tests économétriques")

resid = m_multi.resid
X_fitted = add_constant(r[["hsi", "byd", "dummy"]])

# Test de White
white_stat, white_p, _, _ = het_white(resid, X_fitted)
print(f"\n  Test de White (hétéroscédasticité)")
print(f"    stat = {white_stat:.4f}  p-value = {white_p:.4f}")
print(f"    Décision : {'hétéroscédasticité détectée ✗' if white_p < 0.05 else 'homoscédasticité ✓'}")

# Durbin-Watson
dw = durbin_watson(resid)
print(f"\n  Durbin-Watson (autocorrélation des résidus)")
print(f"    DW = {dw:.4f}  {'→ pas d autocorrélation ✓' if 1.5 < dw < 2.5 else '→ autocorrélation possible ✗'}")

# Jarque-Bera
jb_stat, jb_p = jarque_bera(resid)
print(f"\n  Jarque-Bera (normalité des résidus)")
print(f"    stat = {jb_stat:.2f}  p-value = {jb_p:.4f}")
print(f"    Décision : {'non-normalité ✗' if jb_p < 0.05 else 'normalité ✓'}")

# Correction HAC (Huber-White)
print("\n  → Correction de Huber-White (HAC) :")
m_hac = OLS(r["xiaomi"], add_constant(r[["hsi", "byd", "dummy"]])).fit(
    cov_type="HC3")
for i, name in enumerate(["const", "HSI", "BYD", "dummy"]):
    coef = m_hac.params[i]
    pval = m_hac.pvalues[i]
    sig  = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else "n.s."))
    print(f"    {name:<10}  β={coef:+.5f}  p={pval:.4f} {sig}")

# Graphique des diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Diagnostics du modèle multiple (HSI + BYD + dummy)", fontweight="bold")

ax = axes[0, 0]
ax.scatter(m_multi.fittedvalues, resid, alpha=0.2, s=8, color=COLORS[0])
ax.axhline(0, color="red", lw=1)
ax.set_xlabel("Valeurs ajustées"); ax.set_ylabel("Résidus")
ax.set_title("Résidus vs valeurs ajustées")

ax = axes[0, 1]
sm.qqplot(resid, line="s", ax=ax, alpha=0.3, markersize=3)
ax.set_title("Q-Q plot des résidus")

ax = axes[1, 0]
ax.hist(resid, bins=60, color=COLORS[0], alpha=0.7, density=True)
x = np.linspace(resid.min(), resid.max(), 200)
ax.plot(x, stats.norm.pdf(x, resid.mean(), resid.std()), "r-", lw=2, label="Normale")
ax.set_title("Distribution des résidus")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(resid.index, resid.values, lw=0.6, color=COLORS[4])
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Résidus dans le temps")

plt.tight_layout()
plt.savefig("figures/04_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → figures/04_diagnostics.png\n")

# ═════════════════════════════════════════════════════════════════════════════
# 8. ANALYSE DE VOLATILITÉ
# ═════════════════════════════════════════════════════════════════════════════
print("8. Analyse de volatilité")

vol_proxy = (returns["xiaomi"] ** 2)

# ACF / PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Corrélogramme de la volatilité (r² journalier)", fontweight="bold")
plot_acf(vol_proxy, lags=30, ax=axes[0], title="ACF")
plot_pacf(vol_proxy, lags=30, ax=axes[1], title="PACF")
plt.tight_layout()
plt.savefig("figures/05_acf_pacf_vol.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/05_acf_pacf_vol.png")

# ─── ARMA(1,1) sur la volatilité proxy ────────────────────────────────────
print("\n  ARMA sur la volatilité (proxy = r²)")
for order in [(1, 0, 0), (1, 0, 1), (2, 0, 1)]:
    try:
        fit = ARIMA(vol_proxy, order=order).fit()
        print(f"    ARMA{order[::2]}  AIC={fit.aic:.1f}  BIC={fit.bic:.1f}")
    except Exception:
        pass

# Meilleur modèle : AR(1)
arma_model = ARIMA(vol_proxy, order=(1, 0, 0)).fit()

# Prévisions statique vs dynamique (30 jan → 11 fév 2026)
FCST_START = "2026-01-30"
FCST_END   = "2026-02-11"
fcst_index = pd.bdate_range(FCST_START, FCST_END)

n_steps = len(fcst_index)
fcst_static  = arma_model.get_prediction(
    start=len(vol_proxy) - n_steps, end=len(vol_proxy) - 1
).predicted_mean

fcst_dynamic = arma_model.forecast(steps=n_steps)
fcst_dynamic.index = fcst_index

# ─── EWMA ─────────────────────────────────────────────────────────────────
print("\n  EWMA (λ=0.94)")
lam = 0.94
ewma_var = vol_proxy.copy()
for t in range(1, len(ewma_var)):
    ewma_var.iloc[t] = lam * ewma_var.iloc[t - 1] + (1 - lam) * returns["xiaomi"].iloc[t] ** 2
ewma_vol = np.sqrt(ewma_var) * np.sqrt(252)

# ─── GARCH(1,1) ──────────────────────────────────────────────────────────
print("\n  GARCH(1,1)")
r_scaled = returns["xiaomi"] * 100
garch = arch_model(r_scaled, vol="Garch", p=1, q=1, dist="Normal")
garch_fit = garch.fit(disp="off")
print(f"    α (ARCH) = {garch_fit.params['alpha[1]']:.4f}")
print(f"    β (GARCH) = {garch_fit.params['beta[1]']:.4f}")
print(f"    α+β = {garch_fit.params['alpha[1]'] + garch_fit.params['beta[1]']:.4f} "
      f"(persistance de la volatilité)")
garch_vol = garch_fit.conditional_volatility / 100 * np.sqrt(252)

# Graphique synthèse volatilité
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
fig.suptitle("Modèles de volatilité — Xiaomi", fontweight="bold")

ax = axes[0]
rolling_vol = returns["xiaomi"].rolling(21).std() * np.sqrt(252)
ax.plot(rolling_vol.index, rolling_vol * 100, color="#cbd5e1", lw=0.8, label="Vol. réalisée 21j")
ax.plot(ewma_vol.index, ewma_vol * 100, color=COLORS[1], lw=1, label="EWMA (λ=0.94)", alpha=0.8)
ax.plot(garch_vol.index, garch_vol * 100, color=COLORS[0], lw=1.2, label="GARCH(1,1)")
ax.axvline(pd.Timestamp(EVENT_DATE), color="red", ls="--", lw=1, label="Blacklist")
ax.axvline(pd.Timestamp(EVENT_LIFT), color="green", ls="--", lw=1)
ax.set_ylabel("Volatilité annualisée (%)")
ax.set_title("Comparaison des estimations de volatilité")
ax.legend(fontsize=8)

# Zoom sur la période d'événement
ax = axes[1]
mask = (rolling_vol.index >= "2020-10-01") & (rolling_vol.index <= "2021-06-30")
ax.plot(rolling_vol[mask].index, rolling_vol[mask] * 100,
        color="#cbd5e1", lw=1, label="Vol. réalisée 21j")
ax.plot(ewma_vol[mask].index, ewma_vol[mask] * 100,
        color=COLORS[1], lw=1.5, label="EWMA")
ax.plot(garch_vol[mask].index, garch_vol[mask] * 100,
        color=COLORS[0], lw=1.5, label="GARCH(1,1)")
ax.axvline(pd.Timestamp(EVENT_DATE), color="red", ls="--", lw=1.5, label="Blacklist (14 jan)")
ax.axvline(pd.Timestamp(EVENT_LIFT), color="green", ls="--", lw=1.5, label="Suspension (12 mar)")
ax.set_ylabel("Volatilité annualisée (%)")
ax.set_title("Zoom : période de l'événement (oct 2020 – juin 2021)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/06_volatilite.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → figures/06_volatilite.png\n")

# ─── Comparaison MSE des modèles de prévision ──────────────────────────────
print("9. Comparaison MSE des prévisions de volatilité (30 jan – 11 fév 2026)")

# Proxy réalisé sur la fenêtre de prévision
actual = vol_proxy.reindex(fcst_index).dropna()
if len(actual) > 0:
    naive   = float(vol_proxy.iloc[-len(actual) - 1])
    ar_pred = arma_model.forecast(steps=len(actual)).values

    ewma_pred = []
    v = ewma_var.iloc[-1]
    for ret in returns["xiaomi"].reindex(actual.index).fillna(0):
        v = lam * v + (1 - lam) * ret ** 2
        ewma_pred.append(v)

    mse_naive = np.mean((actual.values - naive) ** 2)
    mse_ar    = np.mean((actual.values - ar_pred[:len(actual)]) ** 2)
    mse_ewma  = np.mean((actual.values - np.array(ewma_pred)) ** 2)

    print(f"  MSE naïf  = {mse_naive:.2e}")
    print(f"  MSE AR(1) = {mse_ar:.2e}")
    print(f"  MSE EWMA  = {mse_ewma:.2e}")

# ═════════════════════════════════════════════════════════════════════════════
# RÉCAPITULATIF
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RÉSULTATS CLÉS")
print("=" * 60)
print(f"  Modèle simple  Xiaomi~HSI  : β={m_hsi.params[1]:.3f}  R²={m_hsi.rsquared:.3f}")
print(f"  Modèle simple  Xiaomi~BYD  : β={m_byd.params[1]:.3f}  R²={m_byd.rsquared:.3f}")
print(f"  Modèle event   (dummy HSI) : γ={m_event.params[2]:.5f}  p={m_event.pvalues[2]:.4f}")
print(f"  Modèle multiple R²         : {m_multi.rsquared:.4f}")
print(f"  CAR fenêtre événement      : {car.iloc[-1]*100:.2f}%")
print(f"  GARCH α+β                  : {garch_fit.params['alpha[1]'] + garch_fit.params['beta[1]']:.4f}")
print(f"\n  Figures sauvegardées dans ./figures/")
print("=" * 60)
