import streamlit as st
import yfinance as yf
import pandas as pd
from urllib.error import URLError
from datetime import datetime

# Configure the page
st.set_page_config(page_title="Gold Intraday Signal", layout="centered")
st.title("📊 Gold Intraday Signal – Pivot + RSI Strategy")

# Custom RSI without pandas_ta
def calculate_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, float('inf'))
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        st.error(f"Error calculating RSI: {str(e)}")
        return None

@st.cache_data(ttl=60)
def load_intraday():
    try:
        df = yf.download("GC=F", interval="5m", period="1d", progress=False)
        if df.empty:
            raise ValueError("No data received from Yahoo Finance")
        return df.dropna()
    except URLError as e:
        st.error(f"Unable to fetch data. Please check your internet connection. Error: {e.reason}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_pivot_levels():
    try:
        hist = yf.download("GC=F", interval="1d", period="2d", progress=False)
        if hist.empty:
            raise ValueError("No historical data received")
        
        prev_day = hist.iloc[0]
        H, L, C = float(prev_day['High']), float(prev_day['Low']), float(prev_day['Close'])
        PP = (H + L + C) / 3
        R1 = 2 * PP - L
        S1 = 2 * PP - H
        return PP, R1, S1
    except Exception as e:
        st.error(f"Error calculating pivot levels: {str(e)}")
        return None, None, None

def get_signal(df, PP, R1, S1):
    try:
        if df is None or PP is None:
            raise ValueError("Missing required data")
            
        df['RSI'] = calculate_rsi(df['Close'])
        latest = df.iloc[-1]
        
        # Convert to float to avoid pandas Series comparison error
        price = float(latest['Close'])
        rsi = float(latest['RSI'])
        S1 = float(S1)
        R1 = float(R1)
        
        signal = "Hold"
        entry_price = None

        # Now using scalar float values for comparison
        if price <= S1 and rsi < 30:
            signal = "🟢 Entry: BUY"
            entry_price = price
        elif price >= R1 and rsi > 70:
            signal = "🔴 Entry: SELL"
            entry_price = price
        else:
            signal = "🟡 No Trade"

        return signal, price, rsi, entry_price
    except Exception as e:
        st.error(f"Error generating signal: {str(e)}")
        return "Error", None, None, None

# Main execution pipeline
try:
    # Load data
    df = load_intraday()
    if df is None:
        st.warning("Unable to load market data. Please try again later.")
        st.stop()

    # Get pivot levels
    PP, R1, S1 = get_pivot_levels()
    if None in (PP, R1, S1):
        st.warning("Unable to calculate pivot levels. Please try again later.")
        st.stop()

    # Generate signal
    signal, price, rsi, entry_price = get_signal(df, PP, R1, S1)
    
    # Display results
    st.markdown("### ⚙️ Strategy Summary")
    if price is not None:
        st.markdown(f"**Current Price:** ${price:.2f}")
    if rsi is not None:
        st.markdown(f"**RSI (14):** {rsi:.2f}")
    st.markdown(f"**Pivot Point:** {PP:.2f}")
    st.markdown(f"**R1 (Resistance):** {R1:.2f}")
    st.markdown(f"**S1 (Support):** {S1:.2f}")
    st.markdown("---")
    st.markdown(f"## 🚦 Signal: {signal}")
    
    if entry_price:
        st.success(f"📌 Entry Price: ${entry_price:.2f}")
    else:
        st.info("📌 No valid entry at the moment.")

    st.caption("Signal generated using Pivot Points and RSI (14). Refresh to update.")
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
