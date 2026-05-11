import streamlit as st
from black_scholes import BlackScholes
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Page Title
st.title("Black-Scholes Option Pricing Model")

st.write("European Call and Put Option Pricing")

# Stock Ticker Input
ticker = st.text_input("NSE Stock Ticker", "RELIANCE.NS")

# Fetch Live Stock Price
stock = yf.Ticker(ticker)

stock_data = stock.history(period="6mo")

if not stock_data.empty:

    stock_price = stock_data['Close'].iloc[-1]

    st.write(f"Live Stock Price: ₹{stock_price:.2f}")

    # Sidebar Inputs
    st.sidebar.header("Model Inputs")

    S = st.sidebar.number_input(
        "Stock Price (S)",
        value=float(stock_price)
    )

    K = st.sidebar.number_input(
        "Strike Price (K)",
        value=100.0
    )

    T = st.sidebar.number_input(
        "Time to Expiry (Years)",
        value=1.0
    )

    r = st.sidebar.number_input(
        "Risk-Free Interest Rate",
        value=0.05
    )

    sigma = st.sidebar.number_input(
        "Volatility (σ)",
        value=0.2
    )
    purchase_price = st.sidebar.number_input(
        "Your Purchase Price (for P&L)",
        value=0.0
    )
    # Create Black-Scholes Model
    bs = BlackScholes(S, K, T, r, sigma)

    # Calculate Prices
    call_price = bs.call_price()
    put_price = bs.put_price()

    call_delta = bs.call_delta()
    put_delta = bs.put_delta()

    gamma = bs.gamma()

    vega = bs.vega()

    call_theta = bs.call_theta()
    put_theta = bs.put_theta()

    call_rho = bs.call_rho()
    put_rho = bs.put_rho()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Pricing",
        "Heatmap",
        "Payoff",
        "Greeks",
        "Stocks chart",
        "Market Comparison"
    ])

    # Pricing Tab
    with tab1:

        st.subheader("Option Prices")

        st.metric(
            "Call Option Price",
            f"₹{call_price:.2f}"
        )

        st.metric(
            "Put Option Price",
            f"₹{put_price:.2f}"
        )
        
    # Heatmap Section
       # Heatmap Tab
    with tab2:

        st.subheader("Call Option Price Heatmap")

        stock_range = np.linspace(S * 0.8, S * 1.2, 10)
        vol_range = np.linspace(0.1, 0.5, 10)

        heatmap_data = []

        for vol in vol_range:

            row = []

            for stock_price_test in stock_range:

                bs_temp = BlackScholes(
                    stock_price_test,
                    K,
                    T,
                    r,
                    vol
                )

                row.append(bs_temp.call_price())

            heatmap_data.append(row)

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[f"{v:.2f}" for v in vol_range],
            columns=[f"{s:.0f}" for s in stock_range]
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".1f",
            cmap="viridis",
            ax=ax
        )

        ax.set_xlabel("Stock Price")
        ax.set_ylabel("Volatility")
        ax.set_title("Call Option Price Heatmap")

        st.pyplot(fig)
        # Payoff Diagram
    with tab3:
     st.subheader("Option Payoff Diagram")

     stock_prices = np.linspace(S * 0.5, S * 1.5, 100)

     call_payoff = np.maximum(stock_prices - K, 0)
     put_payoff = np.maximum(K - stock_prices, 0)

     fig2, ax2 = plt.subplots(figsize=(10, 5))

     ax2.plot(
        stock_prices,
        call_payoff,
        label="Call Option Payoff"
     )

     ax2.plot(
        stock_prices,
        put_payoff,
        label="Put Option Payoff"
     )

     ax2.axvline(
        K,
        linestyle="--",
        label="Strike Price"
     )

     ax2.set_xlabel("Stock Price at Expiry")
     ax2.set_ylabel("Profit / Loss")

     ax2.set_title("Option Payoff at Expiry")

     ax2.legend()

     st.pyplot(fig2)
    # P&L Analysis
    
    if purchase_price > 0:

        st.subheader("P&L at Expiry")
        pnl_spots = np.linspace(
                S * 0.6,
                S * 1.4,
                200
            )

        pnl = [
                max(s - K, 0) - purchase_price
                for s in pnl_spots
            ]

        fig3, ax3 = plt.subplots(figsize=(10, 4))

        ax3.plot(
                pnl_spots,
                pnl
            )

        ax3.axhline(
                0,
                linestyle="--"
            )

        ax3.fill_between(
                pnl_spots,
                pnl,
                0,
                where=[p > 0 for p in pnl],
                alpha=0.2
            )

        ax3.fill_between(
                pnl_spots,
                pnl,
                0,
                where=[p < 0 for p in pnl],
                alpha=0.2
            )

        ax3.set_xlabel(
                "Spot Price at Expiry"
            )

        ax3.set_ylabel(
                "P&L (₹)"
            )

        ax3.set_title(
                "Option Profit & Loss at Expiry"
            )

        st.pyplot(fig3)

    with tab4:
     st.subheader("Option Greeks")

     col1, col2 = st.columns(2)

     with col1:

         st.metric(
            "Call Delta",
            f"{call_delta:.4f}"
         )

         st.metric(
            "Call Theta",
            f"{call_theta:.4f}"
         )

         st.metric(
            "Call Rho",
            f"{call_rho:.4f}"
         )

     with col2:

         st.metric(
            "Put Delta",
            f"{put_delta:.4f}"
         )

         st.metric(
            "Put Theta",
            f"{put_theta:.4f}"
         )

         st.metric(
            "Put Rho",
            f"{put_rho:.4f}"
         )

     st.metric(
        "Gamma",
        f"{gamma:.4f}"
     )

     st.metric(
        "Vega",
        f"{vega:.4f}"
     )
     st.subheader("Delta vs Spot Price")

     spots = np.linspace(S * 0.5, S * 1.5, 100)

     call_deltas = [
            BlackScholes(s, K, T, r, sigma).call_delta()
            for s in spots
        ]

     put_deltas = [
            BlackScholes(s, K, T, r, sigma).put_delta()
            for s in spots
        ]

     fig2, ax2 = plt.subplots(figsize=(10, 4))

     ax2.plot(
            spots,
            call_deltas,
            label="Call Delta"
        )

     ax2.plot(
            spots,
            put_deltas,
            label="Put Delta"
        )

     ax2.axvline(
            S,
            linestyle="--",
            label="Current Spot"
        )

     ax2.axvline(
            K,
            linestyle="--",
            label="Strike"
        )

     ax2.set_xlabel("Spot Price")
     ax2.set_ylabel("Delta")

     ax2.legend()

     ax2.set_title("Delta vs Spot Price")

     st.pyplot(fig2)
    # Stock Chart Tab
    with tab5:

        st.subheader(f"{ticker} Historical Price Chart")

        fig3, ax3 = plt.subplots(figsize=(12, 5))

        ax3.plot(
            stock_data.index,
            stock_data['Close']
        )

        ax3.set_xlabel("Date")
        ax3.set_ylabel("Closing Price")

        ax3.set_title(f"{ticker} - 6 Month Price History")

        st.pyplot(fig3)
        # Market Comparison Tab
    with tab6:
        st.subheader("Black-Scholes vs Market Price")

        market_price = st.number_input(
            "Enter Actual Market Option Price",
            value=10.0
        )

        bs_price = call_price

        diff = bs_price - market_price

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Black-Scholes Price",
            f"₹{bs_price:.2f}"
        )

        col2.metric(
            "Market Price",
            f"₹{market_price:.2f}"
        )

        col3.metric(
            "Difference",
            f"₹{diff:.2f}",
            delta=(
                "Overpriced"
                if diff > 0
                else "Underpriced"
            )
        )

        st.write(
            """
            Compare theoretical Black-Scholes price
            with actual NSE market option price.
            """
        )
   
else:
    st.error("Invalid ticker symbol or unable to fetch stock data.")

# Footer
st.markdown("---")

st.write(
    "Developed using Python, Streamlit, SciPy, and Yahoo Finance API"
)