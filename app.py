"""
Black-Scholes Options Pricing Dashboard — NSE Edition
Redesigned for clarity, creativity, and educational depth.
"""
 
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BS Options · NSE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────────────────────
# DESIGN SYSTEM — Amber/Obsidian editorial theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,700&family=JetBrains+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');
 
:root {
  --ink:      #0c0c0d;
  --paper:    #111214;
  --card:     #17191c;
  --lift:     #1e2126;
  --border:   rgba(255,255,255,0.06);
  --border2:  rgba(255,255,255,0.11);
  --gold:     #f5a623;
  --gold2:    #ffcc6b;
  --gold-dim: rgba(245,166,35,0.12);
  --teal:     #00c9a7;
  --teal-dim: rgba(0,201,167,0.12);
  --rose:     #ff5f7e;
  --rose-dim: rgba(255,95,126,0.12);
  --sky:      #4db8ff;
  --sky-dim:  rgba(77,184,255,0.12);
  --lav:      #b39ddb;
  --text:     #e8eaec;
  --muted:    #7a8290;
  --faint:    #3a404a;
  --font-serif: 'Playfair Display', Georgia, serif;
  --font-mono:  'JetBrains Mono', 'Courier New', monospace;
  --font-sans:  'Outfit', system-ui, sans-serif;
}
 
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.main, section.main,
[data-testid="stMainBlockContainer"] {
  background-color: var(--ink) !important;
  color: var(--text) !important;
}
 
#MainMenu, footer, header[data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
 
[data-testid="stSidebar"] {
  background: var(--paper) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font-sans) !important; }
[data-testid="stSidebar"] label {
  color: var(--muted) !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: var(--lift) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 13px !important;
  border-radius: 6px !important;
}
 
.block-container {
  padding: 2rem 2.5rem !important;
  max-width: 100% !important;
  font-family: var(--font-sans) !important;
}
 
.js-plotly-plot, .plot-container { background: transparent !important; }
 
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1rem !important;
}
[data-testid="stMetricLabel"] > div {
  color: var(--muted) !important;
  font-size: 11px !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-family: var(--font-sans) !important;
}
[data-testid="stMetricValue"] > div {
  font-family: var(--font-mono) !important;
  font-size: 22px !important;
  color: var(--text) !important;
}
 
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--card) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: var(--font-sans) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  border-radius: 8px !important;
  padding: 8px 18px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: var(--lift) !important;
  color: var(--gold) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabsContent"] { background: transparent !important; border: none !important; padding: 0 !important; }
 
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--font-sans) !important;
  font-size: 13px !important;
  color: var(--muted) !important;
}
 
/* ── Custom components ── */
.page-eyebrow {
  font-family: var(--font-mono);
  font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--gold); margin-bottom: 6px;
}
.page-title {
  font-family: var(--font-serif);
  font-size: clamp(32px,4vw,52px); font-weight: 700; font-style: italic;
  line-height: 1.05; color: var(--text); letter-spacing: -0.5px; margin: 0 0 10px;
}
.page-title em { color: var(--gold); }
.page-sub { font-family: var(--font-sans); font-size: 14px; color: var(--muted); max-width: 600px; line-height: 1.6; }
 
.price-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.5rem; position: relative; overflow: hidden;
}
.price-card::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; }
.price-card.call::before { background: var(--teal); }
.price-card.put::before  { background: var(--rose); }
.price-card-type { font-family: var(--font-mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px; }
.price-card.call .price-card-type { color: var(--teal); }
.price-card.put  .price-card-type { color: var(--rose); }
.price-card-value { font-family: var(--font-serif); font-size: 42px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.price-card.call .price-card-value { color: var(--teal); }
.price-card.put  .price-card-value { color: var(--rose); }
.price-card-label { font-family: var(--font-sans); font-size: 12px; color: var(--muted); }
 
.greek-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin: 1rem 0; }
.greek-tile {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px 12px; text-align: center;
}
.greek-sym { font-family:'Playfair Display',Georgia,serif; font-style:italic; font-size:26px; line-height:1.1; margin-bottom:4px; }
.greek-name { font-family:var(--font-mono); font-size:9px; text-transform:uppercase; letter-spacing:0.1em; color:var(--faint); margin-bottom:8px; }
.greek-row { font-family:var(--font-mono); font-size:12px; font-weight:500; display:flex; justify-content:space-between; padding:3px 0; }
.greek-lbl { font-size:9px; color:var(--faint); text-transform:uppercase; letter-spacing:0.06em; }
 
.formula-block {
  background: var(--lift); border: 1px solid var(--border); border-radius: 10px;
  padding: 1.25rem 1.5rem; font-family: var(--font-mono); font-size: 13px; line-height: 2.1;
  color: var(--text); margin: 12px 0;
}
.formula-block .var { color: var(--gold); }
.formula-block .fn  { color: var(--teal); }
.formula-block .comment { color: var(--faint); font-style: italic; }
 
.callout { border-radius: 10px; padding: 14px 16px; margin: 10px 0; font-family: var(--font-sans); font-size: 13px; line-height: 1.65; }
.callout.gold  { background: var(--gold-dim);  border-left: 3px solid var(--gold);  color: #c8941a; }
.callout.teal  { background: var(--teal-dim);  border-left: 3px solid var(--teal);  color: #00a88c; }
.callout.rose  { background: var(--rose-dim);  border-left: 3px solid var(--rose);  color: #d94d65; }
.callout.sky   { background: var(--sky-dim);   border-left: 3px solid var(--sky);   color: #3da0e0; }
.callout b { color: var(--text); }
 
.learn-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.5rem; height: 100%; margin-bottom: 14px;
}
.learn-card-icon { font-family:'Playfair Display',serif; font-style:italic; font-size:32px; margin-bottom:12px; }
.learn-card-title { font-family:var(--font-sans); font-size:15px; font-weight:600; color:var(--text); margin-bottom:8px; }
.learn-card-body  { font-family:var(--font-sans); font-size:13px; color:var(--muted); line-height:1.7; }
.learn-example {
  margin-top:12px; padding:10px 12px; background:var(--lift);
  border-left:2px solid var(--gold); border-radius:0 6px 6px 0;
  font-family:var(--font-mono); font-size:11px; color:var(--gold2); line-height:1.65;
}
 
.summary-bar {
  display:flex; background:var(--card); border:1px solid var(--border);
  border-radius:12px; overflow:hidden; margin-bottom:1.5rem;
}
.summary-item { flex:1; padding:14px 18px; border-right:1px solid var(--border); }
.summary-item:last-child { border-right:none; }
.summary-key { font-family:var(--font-mono); font-size:9px; text-transform:uppercase; letter-spacing:0.1em; color:var(--faint); margin-bottom:6px; }
.summary-val { font-family:var(--font-mono); font-size:16px; font-weight:500; color:var(--text); }
 
.section-rule { height:1px; background:linear-gradient(to right,var(--border2),transparent); margin:2rem 0; }
.section-label { font-family:var(--font-mono); font-size:10px; text-transform:uppercase; letter-spacing:0.14em; color:var(--gold); margin-bottom:12px; }
 
.param-table { width:100%; border-collapse:collapse; font-family:var(--font-sans); }
.param-table tr { border-bottom:1px solid var(--border); }
.param-table tr:last-child { border-bottom:none; }
.param-table td { padding:10px 0; font-size:13px; vertical-align:top; }
.param-table td:first-child { color:var(--muted); width:38%; }
.param-table td:last-child { color:var(--text); line-height:1.5; }
.param-table strong { color:var(--gold); }
 
hr { border:none; border-top:1px solid var(--border); margin:1.5rem 0; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--paper); }
::-webkit-scrollbar-thumb { background:var(--lift); border-radius:3px; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
# MATH CORE
# ─────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0), max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return call, put
 
 
def bs_greeks(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return {}
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf1  = norm.pdf(d1)
    Nd1   = norm.cdf(d1);  Nd2  = norm.cdf(d2)
    Nnd1  = norm.cdf(-d1); Nnd2 = norm.cdf(-d2)
    eqT   = np.exp(-q * T); erT = np.exp(-r * T)
    sqrtT = np.sqrt(T)
    delta_c =  eqT * Nd1
    delta_p = -eqT * Nnd1
    gamma   =  eqT * pdf1 / (S * sigma * sqrtT)
    vega    =  S * eqT * pdf1 * sqrtT / 100
    theta_c = (-(S * eqT * pdf1 * sigma) / (2 * sqrtT) - r * K * erT * Nd2  + q * S * eqT * Nd1)  / 365
    theta_p = (-(S * eqT * pdf1 * sigma) / (2 * sqrtT) + r * K * erT * Nnd2 - q * S * eqT * Nnd1) / 365
    rho_c   =  K * T * erT * Nd2  / 100
    rho_p   = -K * T * erT * Nnd2 / 100
    return dict(delta_c=delta_c, delta_p=delta_p, gamma=gamma, vega=vega,
                theta_c=theta_c, theta_p=theta_p, rho_c=rho_c, rho_p=rho_p,
                d1=d1, d2=d2)
 
 
def implied_vol(market_price, S, K, T, r, q, option_type='call'):
    def obj(sigma):
        c, p = bs_price(S, K, T, r, sigma, q)
        return (c if option_type == 'call' else p) - market_price
    try:
        return brentq(obj, 1e-6, 5.0, xtol=1e-6, maxiter=500)
    except Exception:
        return None
 
 
def dark_layout(fig, height=320, title=""):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(family="Playfair Display", size=15,
                                         color="#e8eaec"), x=0) if title else None,
        plot_bgcolor="#17191c",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#7a8290", size=11),
        margin=dict(l=12, r=12, t=36 if title else 12, b=12),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#7a8290")),
        hovermode="x unified",
    )
    return fig
 
 
# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='padding:1rem 0 0.5rem;'>
  <div style='font-family:"Playfair Display",serif;font-style:italic;font-size:22px;color:#f5a623;'>Black-Scholes</div>
  <div style='font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;color:#3a404a;margin-top:4px;'>NSE Options Pricer</div>
</div>
<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:10px 0;'>
""", unsafe_allow_html=True)
 
    st.markdown("<div style='font-family:Outfit;font-size:11px;color:#f5a623;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Quick Preset</div>", unsafe_allow_html=True)
    preset = st.selectbox("Quick Preset", ["Custom", "NIFTY (ATM)", "BANKNIFTY (ATM)", "RELIANCE", "TCS", "INFY"], label_visibility="collapsed")
 
    PRESETS = {
        "NIFTY (ATM)":     dict(S=24500.0, K=24500.0, T=7,  sigma=14.5, r=6.5, q=0.0),
        "BANKNIFTY (ATM)": dict(S=52000.0, K=52000.0, T=7,  sigma=16.2, r=6.5, q=0.0),
        "RELIANCE":        dict(S=2950.0,  K=2950.0,  T=30, sigma=22.0, r=6.5, q=0.4),
        "TCS":             dict(S=4100.0,  K=4100.0,  T=30, sigma=18.5, r=6.5, q=0.3),
        "INFY":            dict(S=1850.0,  K=1850.0,  T=30, sigma=24.0, r=6.5, q=0.6),
        "Custom":          dict(S=24500.0, K=24500.0, T=7,  sigma=14.5, r=6.5, q=0.0),
    }
    p = PRESETS[preset]
 
    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:10px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Outfit;font-size:11px;color:#f5a623;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Model Parameters</div>", unsafe_allow_html=True)
    S      = st.number_input("Spot Price (S) ₹",     value=float(p["S"]),     step=50.0,  format="%.2f")
    K      = st.number_input("Strike Price (K) ₹",   value=float(p["K"]),     step=50.0,  format="%.2f")
    T_days = st.number_input("Days to Expiry (t)",   value=int(p["T"]),       step=1, min_value=1, max_value=365)
    sigma  = st.number_input("Volatility σ (%)",     value=float(p["sigma"]), step=0.5,   format="%.2f",
                              help="India VIX is the standard NSE proxy")
    r      = st.number_input("Risk-Free Rate r (%)", value=float(p["r"]),     step=0.1,   format="%.2f",
                              help="NSE uses 6.5% (91-day T-Bill)")
    q      = st.number_input("Dividend Yield q (%)", value=float(p["q"]),     step=0.1,   format="%.2f",
                              help="0 for index options")
 
    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:10px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Outfit;font-size:11px;color:#f5a623;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>P&L Analysis</div>", unsafe_allow_html=True)
    purchase_price_c = st.number_input("Call Purchase Price ₹", value=0.0, step=1.0, format="%.2f", help="Set > 0 to show P&L")
    purchase_price_p = st.number_input("Put Purchase Price ₹",  value=0.0, step=1.0, format="%.2f")
 
    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:10px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Outfit;font-size:11px;color:#f5a623;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Implied Volatility Solver</div>", unsafe_allow_html=True)
    market_price_input = st.number_input("Market Option Price ₹", value=0.0, step=1.0, format="%.2f")
    iv_type = st.selectbox("Option Type for IV", ["Call", "Put"])
 
    st.markdown("""
<div style='margin-top:1.5rem;padding:12px;background:rgba(245,166,35,0.07);border-radius:8px;border:1px solid rgba(245,166,35,0.15);'>
  <div style='font-family:"JetBrains Mono",monospace;font-size:10px;color:#f5a623;letter-spacing:0.06em;'>NSE DEFAULTS</div>
  <div style='font-family:Outfit;font-size:11px;color:#7a8290;margin-top:6px;line-height:1.6;'>
    r = 6.5% (91-day T-Bill)<br>q = 0% for index options<br>td = 365 days/year (Zerodha)<br>σ = India VIX when blank
  </div>
</div>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────────────────────
T_yr      = T_days / 365.0
sigma_dec = sigma  / 100.0
r_dec     = r      / 100.0
q_dec     = q      / 100.0
 
call_price, put_price = bs_price(S, K, T_yr, r_dec, sigma_dec, q_dec)
greeks = bs_greeks(S, K, T_yr, r_dec, sigma_dec, q_dec)
 
moneyness       = S / K
intrinsic_call  = max(S - K, 0.0)
intrinsic_put   = max(K - S, 0.0)
tv_call         = call_price - intrinsic_call
tv_put          = put_price  - intrinsic_put
parity_rhs      = S * np.exp(-q_dec * T_yr) - K * np.exp(-r_dec * T_yr)
parity_diff     = abs((call_price - put_price) - parity_rhs)
 
iv_result = None
if market_price_input > 0:
    iv_result = implied_vol(market_price_input, S, K, T_yr, r_dec, q_dec, iv_type.lower())
 
 
# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
col_head, col_iv_box = st.columns([3, 1])
with col_head:
    st.markdown("""
<div class="page-eyebrow">NSE · European Options · Black-Scholes Model</div>
<h1 class="page-title">Options <em>Pricing</em><br>Intelligence</h1>
<p class="page-sub">Model-driven pricing for NSE derivatives using the Black-Scholes framework.
Compute fair value, Greeks, sensitivity surfaces, and implied volatility — instantly.</p>
""", unsafe_allow_html=True)
 
with col_iv_box:
    if iv_result is not None:
        st.markdown(f"""
<div style='background:var(--card);border:1px solid rgba(245,166,35,0.3);border-radius:14px;
padding:1.25rem;text-align:center;margin-top:1.5rem;'>
  <div style='font-family:"JetBrains Mono",monospace;font-size:10px;text-transform:uppercase;
  letter-spacing:0.12em;color:#f5a623;margin-bottom:8px;'>Implied Volatility</div>
  <div style='font-family:"Playfair Display",serif;font-style:italic;font-size:40px;
  font-weight:700;color:#f5a623;line-height:1;'>{iv_result*100:.2f}%</div>
  <div style='font-family:Outfit;font-size:12px;color:#7a8290;margin-top:6px;'>
  from ₹{market_price_input:.2f} {iv_type}</div>
</div>
""", unsafe_allow_html=True)
    else:
        moneyness_label = ("In-the-Money" if moneyness > 1.01 else
                           "Out-of-the-Money" if moneyness < 0.99 else "At-the-Money")
        mc = "#00c9a7" if moneyness > 1.01 else "#ff5f7e" if moneyness < 0.99 else "#f5a623"
        st.markdown(f"""
<div style='background:var(--card);border:1px solid var(--border);border-radius:14px;
padding:1.25rem;text-align:center;margin-top:1.5rem;'>
  <div style='font-family:"JetBrains Mono",monospace;font-size:10px;text-transform:uppercase;
  letter-spacing:0.12em;color:#7a8290;margin-bottom:8px;'>Moneyness (S/K)</div>
  <div style='font-family:"Playfair Display",serif;font-style:italic;font-size:32px;
  font-weight:700;color:{mc};line-height:1;'>{moneyness:.4f}</div>
  <div style='font-family:Outfit;font-size:12px;color:{mc};margin-top:8px;padding:3px 12px;
  background:rgba(245,166,35,0.08);border-radius:20px;display:inline-block;'>{moneyness_label}</div>
</div>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
# SUMMARY BAR
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="summary-bar">
  <div class="summary-item"><div class="summary-key">Spot S</div><div class="summary-val">₹{S:,.2f}</div></div>
  <div class="summary-item"><div class="summary-key">Strike K</div><div class="summary-val">₹{K:,.2f}</div></div>
  <div class="summary-item"><div class="summary-key">Expiry</div><div class="summary-val">{T_days}d</div></div>
  <div class="summary-item"><div class="summary-key">Vol σ</div><div class="summary-val">{sigma:.2f}%</div></div>
  <div class="summary-item"><div class="summary-key">Rate r</div><div class="summary-val">{r:.2f}%</div></div>
  <div class="summary-item"><div class="summary-key">Div q</div><div class="summary-val">{q:.2f}%</div></div>
</div>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
# PRICE CARDS
# ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="price-card call">
  <div class="price-card-type">Call Premium</div>
  <div class="price-card-value">₹{call_price:,.2f}</div>
  <div class="price-card-label">Intrinsic ₹{intrinsic_call:.2f} · Time Value ₹{tv_call:.2f}</div>
</div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="price-card put">
  <div class="price-card-type">Put Premium</div>
  <div class="price-card-value">₹{put_price:,.2f}</div>
  <div class="price-card-label">Intrinsic ₹{intrinsic_put:.2f} · Time Value ₹{tv_put:.2f}</div>
</div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="price-card" style="border-top:2px solid #f5a623;">
  <div class="price-card-type" style="color:#f5a623;">Put-Call Sum</div>
  <div class="price-card-value" style="color:#f5a623;">₹{call_price+put_price:,.2f}</div>
  <div class="price-card-label">C + P · parity check</div>
</div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="price-card" style="border-top:2px solid #b39ddb;">
  <div class="price-card-type" style="color:#b39ddb;">Parity Error</div>
  <div class="price-card-value" style="color:#b39ddb;">₹{parity_diff:.4f}</div>
  <div class="price-card-label">C − P vs S·e⁻ᵍᵀ − K·e⁻ʳᵀ</div>
</div>""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
# GREEKS PANEL
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Option Greeks — Sensitivity Dashboard</div>", unsafe_allow_html=True)
 
if greeks:
    gdefs = [
        ("Δ","Delta", greeks["delta_c"], greeks["delta_p"], "#00c9a7","#ff5f7e","Price sensitivity per ₹1 move in spot"),
        ("Γ","Gamma", greeks["gamma"],   greeks["gamma"],   "#4db8ff","#4db8ff","Rate of change of Delta per ₹1 move"),
        ("Θ","Theta", greeks["theta_c"], greeks["theta_p"], "#ff5f7e","#00c9a7","Daily time decay (₹ per day)"),
        ("ν","Vega",  greeks["vega"],    greeks["vega"],    "#f5a623","#f5a623","Sensitivity per 1% change in volatility"),
        ("ρ","Rho",   greeks["rho_c"],   greeks["rho_p"],   "#b39ddb","#b39ddb","Sensitivity per 1% change in rate"),
    ]
    tiles = '<div class="greek-grid">'
    for sym, name, cv, pv, cc, pc, desc in gdefs:
        tiles += f"""
<div class="greek-tile" title="{desc}">
  <div class="greek-sym" style="color:{cc}">{sym}</div>
  <div class="greek-name">{name}</div>
  <div style="border-top:1px solid rgba(255,255,255,0.05);margin:6px 0;"></div>
  <div class="greek-row"><span class="greek-lbl">Call</span><span style="color:{cc}">{cv:+.4f}</span></div>
  <div class="greek-row"><span class="greek-lbl">Put</span><span style="color:{pc}">{pv:+.4f}</span></div>
</div>"""
    tiles += '</div>'
    st.markdown(tiles, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
tab_sens, tab_pnl, tab_payoff, tab_learn = st.tabs([
    "📊  Sensitivity Analysis",
    "💰  P&L Simulation",
    "📈  Payoff Diagrams",
    "📚  Learn the Model",
])
 
 
# ═══════════════════════
# TAB 1 — SENSITIVITY
# ═══════════════════════
with tab_sens:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="callout gold">
  <b>Sensitivity Heatmaps</b> show how call and put premiums change as
  <b>Spot Price</b> and <b>Volatility</b> vary simultaneously.
  Darker = higher premium. The dashed amber line marks the current ATM strike.
  Set a purchase price in the sidebar to switch to P&L view.
</div>""", unsafe_allow_html=True)
 
    spot_range = np.linspace(S * 0.80, S * 1.20, 30)
    vol_range  = np.linspace(max(sigma * 0.40, 2), sigma * 2.0, 30)
    call_m = np.zeros((len(vol_range), len(spot_range)))
    put_m  = np.zeros((len(vol_range), len(spot_range)))
    for i, v in enumerate(vol_range):
        for j, s in enumerate(spot_range):
            c, p = bs_price(s, K, T_yr, r_dec, v/100, q_dec)
            call_m[i,j] = c - purchase_price_c if purchase_price_c > 0 else c
            put_m[i,j]  = p - purchase_price_p if purchase_price_p > 0 else p
 
    xlbls = [f"₹{s:,.0f}" for s in spot_range]
    ylbls = [f"{v:.1f}%" for v in vol_range]
    atm_x = xlbls[len(xlbls)//2]
 
    h1, h2 = st.columns(2)
    for col, z, cscale, title in [
        (h1, call_m,
         [[0,"#ff5f7e"],[0.5,"#17191c"],[1,"#00c9a7"]] if purchase_price_c > 0 else "Greens",
         "Call P&L (₹)" if purchase_price_c > 0 else "Call Premium (₹)"),
        (h2, put_m,
         [[0,"#ff5f7e"],[0.5,"#17191c"],[1,"#00c9a7"]] if purchase_price_p > 0 else "Reds",
         "Put P&L (₹)"  if purchase_price_p > 0 else "Put Premium (₹)"),
    ]:
        with col:
            fig = go.Figure(go.Heatmap(
                z=z, x=xlbls, y=ylbls, colorscale=cscale,
                hoverongaps=False,
                hovertemplate="Spot: %{x}<br>Vol: %{y}<br>Value: ₹%{z:.2f}<extra></extra>",
                colorbar=dict(title="₹", tickfont=dict(color="#7a8290",size=10)),
            ))
            # make sure atm_x is valid
        if atm_x is not None:
         try:
         atm_x = float(atm_x)

         fig.add_vline(
            x=atm_x,
            line_color="rgba(245,166,35,0.5)",
            line_dash="dash",
            annotation_text="ATM",
            annotation_font_color="#f5a623",
            annotation_font_size=10
          )
         except:
         pass
            dark_layout(fig, 320, title)
            fig.update_layout(
                xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
 
    st.markdown("<div class='section-label' style='margin-top:1rem;'>Greeks Across Spot Prices</div>", unsafe_allow_html=True)
    st.markdown("""<div class="callout teal">
  Each curve shows how a Greek changes as spot moves ±20% from current level.
  The <b>vertical amber line = current spot</b>. Gamma peaks at ATM — that's where Delta changes fastest.
</div>""", unsafe_allow_html=True)
 
    spot_arr = np.linspace(S * 0.80, S * 1.20, 100)
    dc, dp, gam, veg, tc, tp = [], [], [], [], [], []
    for s in spot_arr:
        g = bs_greeks(s, K, T_yr, r_dec, sigma_dec, q_dec)
        if g:
            dc.append(g["delta_c"]); dp.append(g["delta_p"])
            gam.append(g["gamma"]*1000); veg.append(g["vega"])
            tc.append(g["theta_c"]);  tp.append(g["theta_p"])
 
    fig_gk = make_subplots(rows=1, cols=3,
                           subplot_titles=["Delta (Δ)", "Gamma (Γ) ×1000", "Theta (Θ)/day"],
                           horizontal_spacing=0.06)
    for ci, (y1, y2, n1, n2, c1c, c2c) in enumerate([
        (dc, dp, "Call Δ", "Put Δ", "#00c9a7", "#ff5f7e"),
        (gam, gam, "Γ×1000","", "#4db8ff",""),
        (tc, tp, "Call Θ", "Put Θ", "#00c9a7", "#ff5f7e"),
    ], 1):
        fig_gk.add_trace(go.Scatter(x=spot_arr, y=y1, name=n1,
                                    line=dict(color=c1c,width=2)), row=1, col=ci)
        if n2:
            fig_gk.add_trace(go.Scatter(x=spot_arr, y=y2, name=n2,
                                        line=dict(color=c2c,width=2,dash="dot")), row=1, col=ci)
        fig_gk.add_vline(x=S, line_color="rgba(245,166,35,0.4)", line_dash="dash", row=1, col=ci)
    dark_layout(fig_gk, 300)
    fig_gk.update_layout(showlegend=False)
    for i in range(1,4):
        fig_gk.update_xaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9), row=1, col=i)
        fig_gk.update_yaxes(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9), row=1, col=i)
    for ann in fig_gk.layout.annotations:
        ann.font.color = "#7a8290"; ann.font.size = 11
    st.plotly_chart(fig_gk, use_container_width=True, config={"displayModeBar": False})
 
 
# ═══════════════════════
# TAB 2 — P&L
# ═══════════════════════
with tab_pnl:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="callout gold">
  <b>P&L at Expiry</b> shows profit or loss across all possible spot prices when the option expires.
  Set your <b>purchase price</b> in the sidebar to personalise. Green zone = profit, red = loss.
</div>""", unsafe_allow_html=True)
 
    pp_c = purchase_price_c if purchase_price_c > 0 else call_price
    pp_p = purchase_price_p if purchase_price_p > 0 else put_price
    se = np.linspace(S * 0.65, S * 1.35, 300)
    pnl_c = np.maximum(se - K, 0) - pp_c
    pnl_p = np.maximum(K - se, 0) - pp_p
 
    fig_pnl = go.Figure()
    for y, clr, nm in [(pnl_c,"#00c9a7","Call P&L"), (pnl_p,"#ff5f7e","Put P&L")]:
        fig_pnl.add_trace(go.Scatter(x=se, y=y, name=nm,
                                     line=dict(color=clr,width=2.5),
                                     fill="tozeroy", fillcolor=clr.replace("#","rgba(").replace("c9a7","201,167,0.07)").replace("5f7e","95,126,0.07)"),
                                     hovertemplate="Spot: ₹%{x:,.0f}<br>P&L: ₹%{y:,.2f}<extra></extra>"))
    fig_pnl.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
    fig_pnl.add_vline(x=S, line_color="rgba(245,166,35,0.5)", line_dash="dash",
                      annotation_text="Current Spot", annotation_font_color="#f5a623", annotation_font_size=10)
    fig_pnl.add_vline(x=K, line_color="rgba(255,255,255,0.2)", line_dash="dot",
                      annotation_text="Strike", annotation_font_color="#7a8290", annotation_font_size=10)
    dark_layout(fig_pnl, 360, "P&L at Expiry vs Spot Price")
    fig_pnl.update_layout(xaxis_title="Spot at Expiry (₹)", yaxis_title="Profit / Loss (₹)")
    st.plotly_chart(fig_pnl, use_container_width=True, config={"displayModeBar": False})
 
    be_c = K + pp_c; be_p = K - pp_p
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Call Purchase Price", f"₹{pp_c:.2f}")
    m2.metric("Put Purchase Price",  f"₹{pp_p:.2f}")
    m3.metric("Call Breakeven", f"₹{be_c:,.2f}", f"{((be_c/S)-1)*100:+.2f}% from spot")
    m4.metric("Put Breakeven",  f"₹{be_p:,.2f}", f"{((be_p/S)-1)*100:+.2f}% from spot")
 
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Time Decay — How Value Erodes</div>", unsafe_allow_html=True)
    st.markdown("""<div class="callout rose">
  <b>Theta decay accelerates near expiry.</b> Assuming spot and volatility stay constant,
  this shows how your option's theoretical value falls day by day.
  Long option buyers fight this curve every session.
</div>""", unsafe_allow_html=True)
 
    days_arr = np.linspace(T_days, 0.5, 120)
    tv_c_arr = [bs_price(S, K, d/365, r_dec, sigma_dec, q_dec)[0] for d in days_arr]
    tv_p_arr = [bs_price(S, K, d/365, r_dec, sigma_dec, q_dec)[1] for d in days_arr]
 
    fig_td = go.Figure()
    fig_td.add_trace(go.Scatter(x=days_arr, y=tv_c_arr, name="Call Value",
                                line=dict(color="#00c9a7",width=2),
                                hovertemplate="Days: %{x:.0f}<br>₹%{y:.2f}<extra></extra>"))
    fig_td.add_trace(go.Scatter(x=days_arr, y=tv_p_arr, name="Put Value",
                                line=dict(color="#ff5f7e",width=2),
                                hovertemplate="Days: %{x:.0f}<br>₹%{y:.2f}<extra></extra>"))
    fig_td.add_vline(x=T_days, line_color="rgba(245,166,35,0.5)", line_dash="dash",
                    annotation_text="Today", annotation_font_color="#f5a623", annotation_font_size=10)
    fig_td.update_xaxes(autorange="reversed")
    dark_layout(fig_td, 300, "Option Value vs Days to Expiry")
    fig_td.update_layout(xaxis_title="Days to Expiry", yaxis_title="Theoretical Value (₹)")
    st.plotly_chart(fig_td, use_container_width=True, config={"displayModeBar": False})
 
 
# ═══════════════════════
# TAB 3 — PAYOFF
# ═══════════════════════
with tab_payoff:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="callout sky">
  <b>Payoff diagrams</b> show net profit/loss at expiry for different strategies.
  Green zone = profit territory. Red zone = loss territory.
  Choose a strategy below to visualise its risk profile.
</div>""", unsafe_allow_html=True)
 
    strategy = st.selectbox("Choose Strategy", [
        "Long Call", "Long Put", "Short Call", "Short Put",
        "Bull Call Spread", "Bear Put Spread", "Long Straddle", "Long Strangle",
    ])
 
    K2   = K * 1.05
    c2_p = bs_price(S, K2, T_yr, r_dec, sigma_dec, q_dec)[0]
    p2_p = bs_price(S, K2, T_yr, r_dec, sigma_dec, q_dec)[1]
    spe  = np.linspace(S * 0.60, S * 1.40, 300)
 
    payoffs = {
        "Long Call":        np.maximum(spe - K, 0) - call_price,
        "Long Put":         np.maximum(K - spe, 0) - put_price,
        "Short Call":       call_price - np.maximum(spe - K, 0),
        "Short Put":        put_price  - np.maximum(K - spe, 0),
        "Bull Call Spread": (np.maximum(spe-K,0) - np.maximum(spe-K2,0) - (call_price - c2_p)),
        "Bear Put Spread":  (np.maximum(K2-spe,0) - np.maximum(K-spe,0) - (p2_p - put_price)),
        "Long Straddle":    (np.maximum(spe-K,0) + np.maximum(K-spe,0) - call_price - put_price),
        "Long Strangle":    (np.maximum(spe-K2,0) + np.maximum(K-spe,0) - c2_p - put_price),
    }
    pay = payoffs[strategy]
 
    fig_pay = go.Figure()
    fig_pay.add_trace(go.Scatter(x=spe, y=np.where(pay>=0,pay,0),
                                 fill="tozeroy", fillcolor="rgba(0,201,167,0.10)",
                                 line=dict(width=0), showlegend=False))
    fig_pay.add_trace(go.Scatter(x=spe, y=np.where(pay<0,pay,0),
                                 fill="tozeroy", fillcolor="rgba(255,95,126,0.10)",
                                 line=dict(width=0), showlegend=False))
    fig_pay.add_trace(go.Scatter(x=spe, y=pay, name=strategy,
                                 line=dict(color="#f5a623",width=2.5),
                                 hovertemplate="Spot: ₹%{x:,.0f}<br>P&L: ₹%{y:,.2f}<extra></extra>"))
    fig_pay.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
    fig_pay.add_vline(x=K, line_color="rgba(245,166,35,0.5)", line_dash="dash",
                      annotation_text=f"K₁ ₹{K:,.0f}", annotation_font_color="#f5a623", annotation_font_size=10)
    if "Spread" in strategy or "Strangle" in strategy:
        fig_pay.add_vline(x=K2, line_color="rgba(179,157,219,0.5)", line_dash="dash",
                          annotation_text=f"K₂ ₹{K2:,.0f}", annotation_font_color="#b39ddb", annotation_font_size=10)
    fig_pay.add_vline(x=S, line_color="rgba(255,255,255,0.2)", line_dash="dot",
                      annotation_text="Spot", annotation_font_color="#7a8290", annotation_font_size=10)
    dark_layout(fig_pay, 400, f"{strategy} — Payoff at Expiry")
    fig_pay.update_layout(xaxis_title="Spot at Expiry (₹)", yaxis_title="Net P&L (₹)")
    st.plotly_chart(fig_pay, use_container_width=True, config={"displayModeBar": False})
 
    max_profit = pay.max(); max_loss = pay.min()
    be_pts = spe[np.where(np.diff(np.sign(pay)))[0]]
    be_str = "  ·  ".join([f"₹{b:,.0f}" for b in be_pts]) if len(be_pts) else "N/A"
    st.markdown(f"""
<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.5rem;">
  <div class="price-card" style="flex:1;min-width:140px;border-top:2px solid #00c9a7;">
    <div class="price-card-type" style="color:#00c9a7;">Max Profit</div>
    <div class="price-card-value" style="color:#00c9a7;font-size:28px;">
      {'Unlimited' if max_profit>9e4 else f'₹{max_profit:,.2f}'}</div>
  </div>
  <div class="price-card" style="flex:1;min-width:140px;border-top:2px solid #ff5f7e;">
    <div class="price-card-type" style="color:#ff5f7e;">Max Loss</div>
    <div class="price-card-value" style="color:#ff5f7e;font-size:28px;">
      {'Unlimited' if max_loss<-9e4 else f'₹{max_loss:,.2f}'}</div>
  </div>
  <div class="price-card" style="flex:2;min-width:200px;border-top:2px solid #f5a623;">
    <div class="price-card-type" style="color:#f5a623;">Breakeven(s)</div>
    <div class="price-card-value" style="color:#f5a623;font-size:22px;">{be_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)
 
 
# ═══════════════════════
# TAB 4 — LEARN
# ═══════════════════════
with tab_learn:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
<div style='font-family:"Playfair Display",serif;font-style:italic;font-size:28px;
font-weight:700;color:#e8eaec;margin-bottom:6px;'>The Black-Scholes Model</div>
<p style='font-family:Outfit;font-size:14px;color:#7a8290;max-width:700px;line-height:1.7;margin-bottom:1.5rem;'>
Everything behind this dashboard — from the ₹{call_price:.0f} call premium to the {greeks.get("delta_c",0):.2f} Delta
— flows from a single elegant set of equations developed in 1973.
Here's what they mean and how to read every number.
</p>
""", unsafe_allow_html=True)
 
    st.markdown("<div class='section-label'>The Formula</div>", unsafe_allow_html=True)
    st.markdown("""
<div class="formula-block">
<span style="color:#3a404a;font-style:italic;">// Black-Scholes European Option Pricing</span><br><br>
<span class="var">d₁</span> = <span class="fn">[ ln(S/K) + (r − q + σ²/2) × T ]</span>
  / <span class="fn">( σ × √T )</span><br>
<span class="var">d₂</span> = <span class="var">d₁</span> − <span class="fn">σ × √T</span><br><br>
<span style="color:#3a404a;font-style:italic;">// Call & Put prices</span><br>
<span class="var">C</span> = <span class="fn">S × e^(−qT) × N(d₁)</span>
  − <span class="fn">K × e^(−rT) × N(d₂)</span><br>
<span class="var">P</span> = <span class="fn">K × e^(−rT) × N(−d₂)</span>
  − <span class="fn">S × e^(−qT) × N(−d₁)</span><br><br>
<span style="color:#3a404a;font-style:italic;">// N(·) = cumulative standard normal CDF</span>
</div>
""", unsafe_allow_html=True)
 
    if greeks:
        d1v = greeks.get("d1", 0); d2v = greeks.get("d2", 0)
        st.markdown(f"""<div class="callout teal">
  <b>With your current inputs:</b><br>
  d₁ = {d1v:.4f} &nbsp;·&nbsp; d₂ = {d2v:.4f}<br>
  N(d₁) = {norm.cdf(d1v):.4f} &nbsp;·&nbsp; N(d₂) = {norm.cdf(d2v):.4f}<br>
  Call = <b>₹{call_price:,.2f}</b> &nbsp;·&nbsp; Put = <b>₹{put_price:,.2f}</b>
</div>""", unsafe_allow_html=True)
 
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>What Each Input Means</div>", unsafe_allow_html=True)
    st.markdown(f"""
<table class="param-table">
  <tr>
    <td><strong>S</strong> — Spot Price</td>
    <td>Current market price of the underlying (Nifty index level, stock price in ₹).
    Higher S → higher call value, lower put value.</td>
  </tr>
  <tr>
    <td><strong>K</strong> — Strike Price</td>
    <td>The price at which you can buy (call) or sell (put) the underlying at expiry.
    Chosen when you enter the contract.</td>
  </tr>
  <tr>
    <td><strong>T</strong> — Time to Expiry</td>
    <td>In years (days ÷ 365). More time = more premium because more can happen.
    NSE weekly options expire every Thursday; monthly on last Thursday of the month.</td>
  </tr>
  <tr>
    <td><strong>σ</strong> — Volatility</td>
    <td>Annualised standard deviation of returns. For NSE, <strong>India VIX</strong>
    (currently {sigma:.1f}%) is the standard proxy. This is the most impactful and uncertain input — small changes cause large premium swings.</td>
  </tr>
  <tr>
    <td><strong>r</strong> — Risk-Free Rate</td>
    <td>Continuously-compounded RBI rate. NSE standard is 6.5% (91-day T-Bill).
    Higher rates → higher call prices, lower put prices.</td>
  </tr>
  <tr>
    <td><strong>q</strong> — Dividend Yield</td>
    <td>Continuous dividend yield. Zero for Nifty/BankNifty index options (dividends
    are already priced into the index). Non-zero for individual stock options.</td>
  </tr>
</table>
""", unsafe_allow_html=True)
 
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>The Greeks — Your Risk Sensors</div>", unsafe_allow_html=True)
 
    lc1, lc2 = st.columns(2)
    learn_cards = [
        ("Δ","Delta","#00c9a7",
         "How much the option price moves per ₹1 change in spot. ATM call ≈ 0.50 — "
         "if Nifty rises 100 points, your call gains ₹50. Put Delta is negative: "
         "falls when spot rises.",
         f"Call Δ = {greeks.get('delta_c',0):+.4f} · put Δ = {greeks.get('delta_p',0):+.4f}"),
        ("Γ","Gamma","#4db8ff",
         "Rate of change of Delta per ₹1 move. Peaks for ATM options near expiry. "
         "High Gamma = Delta shifts fast as market moves. "
         "Market makers must rebalance their hedges constantly around high-Gamma positions.",
         f"Γ = {greeks.get('gamma',0):.6f} (identical for call & put)"),
        ("Θ","Theta","#ff5f7e",
         "Daily time decay in ₹ — how much the option loses each day purely from time passing, "
         "everything else held constant. Option buyers pay Theta every session; "
         "sellers collect it. Accelerates sharply inside the last 2 weeks.",
         f"Call Θ = ₹{greeks.get('theta_c',0):.4f}/day · Put Θ = ₹{greeks.get('theta_p',0):.4f}/day"),
        ("ν","Vega","#f5a623",
         "Sensitivity to 1% change in implied volatility. Long options always have positive Vega. "
         "If India VIX rises from "
         f"{sigma:.0f}% → {sigma+1:.0f}%, your option gains Vega (₹) in value.",
         f"ν = ₹{greeks.get('vega',0):.4f} per 1% vol change"),
        ("ρ","Rho","#b39ddb",
         "Sensitivity to 1% change in the risk-free rate. Calls have positive Rho, "
         "puts have negative Rho. Typically the smallest Greek for short-dated NSE options — "
         "but matters for long-dated LEAPS or when RBI makes surprise rate moves.",
         f"Call ρ = ₹{greeks.get('rho_c',0):.4f} · Put ρ = ₹{greeks.get('rho_p',0):.4f} per 1% rate"),
    ]
    for i, (sym, name, color, desc, example) in enumerate(learn_cards):
        with (lc1 if i % 2 == 0 else lc2):
            st.markdown(f"""
<div class="learn-card" style="border-top:2px solid {color};">
  <div class="learn-card-icon" style="color:{color};">{sym}</div>
  <div class="learn-card-title">{name}</div>
  <div class="learn-card-body">{desc}</div>
  <div class="learn-example">📍 {example}</div>
</div>
""", unsafe_allow_html=True)
 
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Intrinsic Value vs Time Value</div>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">
  <div class="callout teal">
    <b>Intrinsic Value</b> = immediate exercise value.<br>
    Call: max(S−K, 0) = max(₹{S:,.0f}−₹{K:,.0f}, 0) = <b>₹{intrinsic_call:,.2f}</b><br>
    Put:  max(K−S, 0) = max(₹{K:,.0f}−₹{S:,.0f}, 0) = <b>₹{intrinsic_put:,.2f}</b>
  </div>
  <div class="callout gold">
    <b>Time Value</b> = premium above intrinsic — what you pay for future possibility.<br>
    Call TV = ₹{call_price:,.2f} − ₹{intrinsic_call:,.2f} = <b>₹{tv_call:,.2f}</b><br>
    Put TV  = ₹{put_price:,.2f} − ₹{intrinsic_put:,.2f} = <b>₹{tv_put:,.2f}</b>
  </div>
</div>
""", unsafe_allow_html=True)
 
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
 
    with st.expander("Put-Call Parity — the no-arbitrage foundation"):
        st.markdown(f"""
For European options on a dividend-paying stock:
 
    C − P = S·e^(−qT) − K·e^(−rT)
 
With your inputs:
 
    LHS: C − P = ₹{call_price:.4f} − ₹{put_price:.4f} = ₹{call_price-put_price:.4f}
    RHS: S·e^(−qT) − K·e^(−rT) = ₹{S*np.exp(-q_dec*T_yr):.4f} − ₹{K*np.exp(-r_dec*T_yr):.4f} = ₹{parity_rhs:.4f}
    Error: ₹{parity_diff:.6f} (should be ~0)
 
If violated in the real market it creates a **risk-free arbitrage** — buy cheap, sell expensive, lock in the spread.
""")
 
    with st.expander("Model Assumptions & Where Black-Scholes Breaks Down"):
        st.markdown("""
**The model assumes:**
- Continuous trading, no transaction costs or taxes
- Log-normal return distribution (constant σ)
- Constant risk-free rate and dividend yield
- European-style exercise only (no early exercise)
- No price jumps or discontinuities
 
**Where it breaks in NSE practice:**
- **Volatility smile/skew** — real OTM options trade at higher IV than ATM. BS assumes flat vol across strikes.
- **Jumps** — NSE stocks gap on earnings, circuit breakers, macro shocks. BS ignores jumps.
- **Illiquidity** — wide bid-ask on far-OTM or long-dated options; mid-price ≠ fair value.
- **American-style stock options** — NSE equity options can be exercised early. BS prices European only.
- **Calendar vs trading days** — 365 vs 252 matters for weekly options (~15% difference).
 
**More realistic models:** Heston (stochastic vol), SABR, Jump-Diffusion (Merton), Local Volatility.
""")
 
    st.markdown("""
<div class="callout gold" style="margin-top:1rem;">
  <b>NSE-Specific Notes:</b><br>
  • <b>India VIX</b> (published by NSE) = 30-day implied vol of Nifty options. Use it as your σ baseline.<br>
  • NSE uses <b>r = 6.5%</b> (91-day RBI T-Bill) — this matches Zerodha's options calculator.<br>
  • <b>Index options</b> (Nifty, BankNifty, FinNifty) are cash-settled European options → BS applies directly.<br>
  • <b>Stock options</b> are American-style → BS slightly underprices deep-ITM options near ex-dividend.<br>
  • <b>td = 365</b> is Zerodha/NSE convention; some global platforms use 252. Results differ by ~15% on 30-day options.
</div>
""", unsafe_allow_html=True)
 