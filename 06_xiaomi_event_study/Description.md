# Étude d'événement : Xiaomi et la liste noire américaine (2021)

Projet d'économétrie financière — ESCP Business School

## Contexte

Le **14 janvier 2021**, l'administration Trump place Xiaomi (1810.HK) sur la liste des *Communist Chinese Military Companies*, obligeant les investisseurs américains à céder leurs positions avant le 11 novembre 2021. Un tribunal américain **suspend cette décision le 12 mars 2021**.

Ce projet analyse l'**impact de cet événement sur le cours de Xiaomi** via une étude d'événement économétrique, en contrôlant les mouvements de marché.

---

## Méthodologie

### 1. Données
- **Source** : Yahoo Finance via `yfinance`
- **Période** : juillet 2018 (IPO Xiaomi) → février 2026
- **Variables** :
  - `1810.HK` — Xiaomi (action principale)
  - `^HSI` — Hang Seng Index (risque de marché)
  - `1211.HK` — BYD (risque sectoriel — véhicules électriques / tech chinoise)
  - `3188.HK` — ETF CSI 300 (marché chinois continental)
  - `TSLA` — Tesla (benchmark sectoriel international)
  - `^GSPC` — S&P 500
- **Rendements** : log-rendements journaliers `r_t = ln(P_t / P_{t-1})`

### 2. Tests de stationnarité
Test ADF (Augmented Dickey-Fuller) sur les prix et rendements.

### 3. Régressions simples (CAPM étendu)

$$R_{Xiaomi,t} = \alpha + \beta \cdot R_{X,t} + \varepsilon_t$$

avec X ∈ {HSI, BYD, CSI300}

### 4. Étude d'événement — variable dummy

$$R_{Xiaomi,t} = \alpha + \beta \cdot R_{HSI,t} + \gamma \cdot D_t + \varepsilon_t$$

- `D_t = 1` pendant la fenêtre d'événement (14 jan → 12 mar 2021)
- `γ` mesure la **performance anormale** de Xiaomi pendant la période, contrôlée du marché
- **CAR** (Cumulative Abnormal Return) = rendement anormal cumulé sur la fenêtre

### 5. Modèle multiple

$$R_{Xiaomi,t} = \alpha + \beta_1 R_{HSI,t} + \beta_2 R_{BYD,t} + \gamma D_t + \varepsilon_t$$

### 6. Tests économétriques
| Test | Objectif |
|------|----------|
| White | Hétéroscédasticité |
| Durbin-Watson | Autocorrélation des résidus |
| Jarque-Bera | Normalité des résidus |
| VIF | Multicolinéarité |
| HAC (Huber-White) | Inférence robuste à l'hétéroscédasticité |

### 7. Analyse de volatilité
- **Proxy** : rendement au carré `r_t²`
- **ARMA(1,0)** sur la volatilité proxy (sélection par critère de Schwarz)
- **EWMA** avec λ = 0.94 (RiskMetrics)
- **GARCH(1,1)** — estimation de la persistance de la volatilité
- Comparaison des MSE de prévision (30 jan → 11 fév 2026)

---

## Résultats principaux

| Indicateur | Valeur |
|---|---|
| β (Xiaomi ~ HSI) | ~0.15 (risque systématique faible) |
| β (Xiaomi ~ BYD) | ~0.13 (corrélation sectorielle) |
| CAR sur la fenêtre d'événement | **−5.8 %** |
| Dummy (corrigée HC3) | Non significative (p > 0.05) |
| GARCH α + β | ~0.99 (forte persistance de la volatilité) |
| Meilleur modèle de vol. (MSE) | EWMA |

> **Interprétation** : Bien qu'un rendement anormal cumulé de −5.8 % soit observé sur la fenêtre d'événement, la variable dummy **n'est pas statistiquement significative** une fois corrigée l'hétéroscédasticité (HAC). Le marché semble avoir partiellement anticipé ou rapidement absorbé l'événement.




## Visualisations

### Prix et rendements
![Prix et rendements](figures/01_prix_rendements.png)

### Régressions simples
![Régressions simples](figures/02_regressions_simples.png)

### Étude d'événement — CAR
![Étude d'événement](figures/03_etude_evenement.png)

### Diagnostics du modèle
![Diagnostics](figures/04_diagnostics.png)

### Modèles de volatilité
![Volatilité](figures/06_volatilite.png)
