import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.error import URLError

# ---- CONFIG ----
st.set_page_config(page_title="Enhanced Gold Intraday Signal", layout="centered")
st.title("📊 Enhanced Gold Signal Generator")

# ---- PARAMETERS ----
RISK_REWARD_RATIO = 2.0
STOP_LOSS_PERCENT = 0.005
TAKE_PROFIT_PERCENT = STOP_LOSS_PERCENT * RISK_REWARD_RATIO

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

def to_scalar(val):
    """Convert pandas Series or numpy scalar to Python float."""
    if isinstance(val, pd.Series) or isinstance(val, np.ndarray):
        return float(val.iloc[0]) if hasattr(val, "iloc") else float(val[0])
    return float(val)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    avg_loss = avg_loss.replace(0, float('inf'))  # Avoid division by zero
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_pivot_levels():
    try:
        hist = yf.download("GC=F", interval="1d", period="2d", progress=False)
        if hist.empty:
            raise ValueError("No historical data received")
        prev_day = hist.iloc[0]
        H = to_scalar(prev_day['High'])
        L = to_scalar(prev_day['Low'])
        C = to_scalar(prev_day['Close'])
        PP = (H + L + C) / 3
        R1 = 2 * PP - L
        S1 = 2 * PP - H
        return PP, R1, S1
    except Exception as e:
        st.error(f"Error calculating pivot levels: {str(e)}")
        return None, None, None

def calculate_technical_indicators(df):
    try:
        df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
        df['EMA_short'] = df['Close'].ewm(span=EMA_SHORT, adjust=False).mean()
        df['EMA_medium'] = df['Close'].ewm(span=EMA_MEDIUM, adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=EMA_LONG, adjust=False).mean()
        exp1 = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=ATR_PERIOD).mean()
        df['Price_Change'] = df['Close'].diff()
        df['Trend_Strength'] = df['Price_Change'].rolling(window=10).sum()
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

def calculate_risk_levels(entry_price, trade_type='BUY'):
    if trade_type == 'BUY':
        stop_loss = entry_price * (1 - STOP_LOSS_PERCENT)
        take_profit = entry_price * (1 + TAKE_PROFIT_PERCENT)
    else:  # SELL
        stop_loss = entry_price * (1 + STOP_LOSS_PERCENT)
        take_profit = entry_price * (1 - TAKE_PROFIT_PERCENT)
    return stop_loss, take_profit

@st.cache_data(ttl=60)
def load_intraday():
    try:
        df = yf.download("GC=F", interval="5m", period="5d", progress=False)
        if df.empty:
            raise ValueError("No data received from Yahoo Finance")
        df = calculate_technical_indicators(df)
        if df is None:
            raise ValueError("Error in technical indicators calculation")
        return df.dropna()
    except URLError as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ---- MAIN EXECUTION ----
try:
    df = load_intraday()
    if df is None:
        st.warning("Unable to load market data. Please try again later.")
        st.stop()

    PP, R1, S1 = get_pivot_levels()
    if None in (PP, R1, S1):
        st.warning("Unable to calculate pivot levels. Please try again later.")
        st.stop()

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # Safely extract scalar values
    rsi_value = to_scalar(latest['RSI'])
    macd_hist = to_scalar(latest['MACD_Histogram'])
    atr_value = to_scalar(latest['ATR'])
    current_price = to_scalar(latest['Close'])
    previous_close = to_scalar(previous['Close'])
    price_delta = current_price - previous_close

    # --- Display Section ---
    st.markdown("### 📈 Current Market Status")
    st.metric(
        label="Gold Price", 
        value=f"${current_price:.2f}",
        delta=f"{price_delta:.2f}"
    )

    st.markdown("### 📊 Technical Indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RSI (14)", f"{rsi_value:.2f}")
        if rsi_value >= RSI_OVERBOUGHT:
            st.markdown("🔴 Overbought")
        elif rsi_value <= RSI_OVERSOLD:
            st.markdown("🟢 Oversold")
    with col2:
        st.metric("MACD Histogram", f"{macd_hist:.4f}")
        st.markdown("🟢 Bullish" if macd_hist > 0 else "🔴 Bearish")
    with col3:
        st.metric("ATR", f"{atr_value:.2f}")

    st.markdown("### 🎯 Pivot Levels")
    pivot_col1, pivot_col2, pivot_col3 = st.columns(3)
    with pivot_col1:
        st.metric("Resistance (R1)", f"${R1:.2f}")
    with pivot_col2:
        st.metric("Pivot Point (PP)", f"${PP:.2f}")
    with pivot_col3:
        st.metric("Support (S1)", f"${S1:.2f}")

    # --- Signal Logic ---
    signal = "🟡 HOLD"
    stop_loss = None
    take_profit = None
    trade_type = None

    if current_price <= S1 and rsi_value <= RSI_OVERSOLD and macd_hist > 0:
        signal = "🟢 BUY"
        trade_type = 'BUY'
        stop_loss, take_profit = calculate_risk_levels(current_price, trade_type)
    elif current_price >= R1 and rsi_value >= RSI_OVERBOUGHT and macd_hist < 0:
        signal = "🔴 SELL"
        trade_type = 'SELL'
        stop_loss, take_profit = calculate_risk_levels(current_price, trade_type)

    st.markdown("### 🚦 Trading Signal")
    st.markdown(f"## {signal}")

    if signal != "🟡 HOLD" and stop_loss is not None and take_profit is not None:
        st.markdown("### 💰 Trade Levels")
        levels_col1, levels_col2, levels_col3 = st.columns(3)
        with levels_col1:
            st.metric("Entry", f"${current_price:.2f}")
        with levels_col2:
            st.metric("Stop Loss", f"${stop_loss:.2f}")
        with levels_col3:
            st.metric("Take Profit", f"${take_profit:.2f}")

        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        st.markdown(f"**Risk/Reward Ratio:** 1:{RISK_REWARD_RATIO}")
        st.markdown(f"Potential Risk: ${risk:.2f}")
        st.markdown(f"Potential Reward: ${reward:.2f}")
    else:
        st.info("No active trade signal at this time.")

    st.markdown("---")
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
