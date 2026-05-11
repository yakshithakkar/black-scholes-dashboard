# Black-Scholes Option Pricing Model

An interactive dashboard for pricing European call and put options on **NSE-listed Indian stocks** using the Black-Scholes model. Built with Python and Streamlit. Pulls live stock prices via Yahoo Finance and calculates option prices, all five Greeks, payoff diagrams, sensitivity heatmaps, and compares theoretical model prices against actual market prices.

---

## Live Demo

> Run locally with `streamlit run app.py`

---

## Screenshots

### Pricing — TCS.NS
![Pricing](screenshots/landing_page.png)

### Call Option Price Heatmap
![Heatmap](screenshots/heatmap.png)

### Option Payoff Diagram
![Payoff](screenshots/payoff.png)

### Option Greeks
![Greeks](screenshots/greeks.png)

### Delta vs Spot Price
![Delta](screenshots/delta_vs_spot_price.png)

### TCS.NS Historical Price Chart (6 months)
![Stocks Chart](screenshots/stocks_chart.png)

### Black-Scholes vs Market Price Comparison
![Market Comparison](screenshots/market_comparision.png)

---

## What It Does

| Tab | Description |
|-----|-------------|
| **Pricing** | Calculates European call and put option prices using Black-Scholes. Displays P&L at expiry. |
| **Heatmap** | Sensitivity heatmap of call option prices across a grid of spot prices × volatility values |
| **Payoff** | Option payoff diagram at expiry for call and put, with strike price marked |
| **Greeks** | All five Greeks — Delta, Gamma, Theta, Vega, Rho — for both call and put |
| **Stocks Chart** | 6-month historical price chart for the selected NSE ticker via yfinance |
| **Market Comparison** | Enter an actual NSE market option price and compare it to the model's theoretical price with Overpriced / Underpriced signal |

---

## Sample Output — TCS.NS

```
Ticker:          TCS.NS
Live Spot Price: ₹2,394.40
Strike Price:    ₹99.96
Time to Expiry:  2.00 years
Risk-Free Rate:  13% (India 10Y Government Bond)
Volatility (σ):  0.20

Call Option Price:  ₹2,317.33
Put Option Price:   ₹0.00

Greeks:
  Call Delta:   1.0000      Put Delta:   0.0000
  Call Theta:  -0.0275/day  Put Theta:   0.0000/day
  Call Rho:     1.5415      Put Rho:    -0.0000
  Gamma:        0.0000
  Vega:         0.0000

Market Comparison (TCS.NS):
  Black-Scholes Price:  ₹2,317.33
  Actual Market Price:  ₹10.00
  Difference:           ₹2,307.33  →  Overpriced
```

---

## The Black-Scholes Formula

```
C = S·N(d1) - K·e^(-rT)·N(d2)       Call option price
P = K·e^(-rT)·N(-d2) - S·N(-d1)     Put option price

d1 = [ln(S/K) + (r + σ²/2)·T] / σ√T
d2 = d1 - σ√T
```

Where:

| Variable | Meaning |
|----------|---------|
| S | Current stock price |
| K | Strike price |
| T | Time to expiry (years) |
| r | Risk-free interest rate |
| σ | Volatility (annualised) |
| N() | Cumulative standard normal distribution |

---

## The Greeks

| Greek | Measures | Call Value | Put Value |
|-------|----------|------------|-----------|
| **Delta (Δ)** | Rate of change of option price w.r.t. spot price | 1.0000 | 0.0000 |
| **Gamma (Γ)** | Rate of change of Delta w.r.t. spot price | 0.0000 | 0.0000 |
| **Theta (Θ)** | Time decay per day | -0.0275 | 0.0000 |
| **Vega (V)** | Sensitivity to 1% change in volatility | 0.0000 | 0.0000 |
| **Rho (ρ)** | Sensitivity to interest rate change | 1.5415 | -0.0000 |

---

## Model Inputs (Sidebar)

| Input | Description | Example |
|-------|-------------|---------|
| Stock Price (S) | Auto-fetched live from NSE via yfinance | ₹2,394.40 |
| Strike Price (K) | The price at which the option can be exercised | ₹99.96 |
| Time to Expiry (T) | In years — e.g. 0.5 = 6 months | 2.00 |
| Risk-Free Rate (r) | India 10Y Government Bond rate | 0.13 |
| Volatility (σ) | Annualised implied/historical volatility | 0.20 |
| Purchase Price | Optional — used to calculate P&L at expiry | ₹100.00 |

Supported NSE tickers: `TCS.NS`, `RELIANCE.NS`, `HDFCBANK.NS`, `INFY.NS`, `NIFTY50.NS`, and any valid NSE ticker with `.NS` suffix.

---

## Why Indian Market Data

Most Black-Scholes implementations use US stocks with a ~5% risk-free rate. This dashboard uses:

- **NSE-listed stocks** via yfinance `.NS` suffix
- **13% risk-free rate** (India 10Y Government Bond), reflecting the actual cost of risk-free capital in the Indian market
- **Live spot prices** fetched at runtime — not hardcoded

This makes the model output meaningful for the market it's intended to analyse.

---

## Limitations of Black-Scholes

Understanding where the model breaks down is as important as knowing how to use it:

- Assumes **constant volatility** — real markets exhibit the volatility smile and skew
- Assumes **log-normal returns** — real returns have fat tails (leptokurtosis)
- Assumes **continuous trading** with no transaction costs
- Assumes **no dividends** on the underlying
- Only valid for **European options** (cannot be exercised early)

The large difference between the model price and market price in the comparison tab reflects these assumptions — particularly the strike being deep in-the-money relative to the spot price, which causes Delta to approach 1 and Gamma/Vega to approach 0.

---

## Tech Stack

```
Python          Core language
NumPy           Numerical calculations
SciPy           Normal distribution functions (norm.cdf, norm.pdf)
yfinance        Live NSE stock data via Yahoo Finance API
Streamlit       Interactive web dashboard
Matplotlib      Charts — payoff diagram, Delta vs Spot, historical price
Seaborn         Heatmap visualisation
```

---

## Installation & Setup

```bash
# Clone the repo
git clone https://github.com/yakshithakkar/black-scholes-pricer.git
cd black-scholes-pricer

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

**Requirements:**
```
numpy
scipy
yfinance
streamlit
matplotlib
seaborn
```

---

## Project Structure

```
black-scholes-pricer/
├── app.py              # Streamlit dashboard — all tabs and UI
├── black_scholes.py    # Core model — pricing class and Greeks
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── screenshots/        # App screenshots for README
    ├── landing_page.png
    ├── heatmap.png
    ├── payoff.png
    ├── greeks.png
    ├── delta_vs_spot_price.png
    ├── stocks_chart.png
    └── market_comparision.png
```

---

## What I Learned Building This

- The relationship between moneyness (S/K ratio) and the Greeks — when a call is deep ITM, Delta approaches 1 and Gamma/Vega collapse toward 0 because the option behaves almost identically to holding the stock outright
- Why traders care about Vega more than Delta in near-expiry options — Vega captures vol risk which dominates short-dated options
- The gap between Black-Scholes theoretical prices and NSE market prices reflects liquidity, supply/demand, and the implied volatility surface — not model error per se
- India's higher risk-free rate (13% vs US 5%) materially changes the time-value-of-money component of option pricing

---

## Author

**Yakshi Thakkar**  
B.Tech Data Science · SVKM's NMIMS University, Mumbai  
Targeting: Quantitative Finance  
GitHub: [yakshithakkar](https://github.com/yakshithakkar)  
Email: thakkaryakshi@gmail.com

> Part of a quantitative finance project series. Next: Portfolio Optimiser + Efficient Frontier.