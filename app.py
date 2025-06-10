import streamlit as st
st.set_page_config(page_title="Gold Intraday Signal", layout="centered")

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

# --- User must provide their Alpha Vantage API Key ---
API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"] if "ALPHAVANTAGE_API_KEY" in st.secrets else st.text_input("Enter your Alpha Vantage API Key:")

st.title("📊 Gold Intraday Signal – Multi-Timeframe (Alpha Vantage)")

# --- Constants ---
GOLD_SYMBOL = "XAUUSD"
TIMEFRAME_MAP = {
    "5 min": ("5min", "FX_INTRADAY"),
    "15 min": ("15min", "FX_INTRADAY"),
    "1 hour": ("60min", "FX_INTRADAY"),
}
RISK_REWARD_RATIO = 2.0
STOP_LOSS_PERCENT = 0.005
TAKE_PROFIT_PERCENT = STOP_LOSS_PERCENT * RISK_REWARD_RATIO
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

def to_scalar(val):
    if isinstance(val, pd.Series) or isinstance(val, np.ndarray):
        return float(val.iloc[0]) if hasattr(val, "iloc") else float(val[0])
    return float(val)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    avg_loss = avg_loss.replace(0, float('inf'))
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_pivot_levels(df):
    prev_day = df.iloc[-2]
    H = to_scalar(prev_day['high'])
    L = to_scalar(prev_day['low'])
    C = to_scalar(prev_day['close'])
    PP = (H + L + C) / 3
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    return PP, R1, S1

def calculate_risk_levels(entry_price, trade_type='BUY'):
    if trade_type == 'BUY':
        stop_loss = entry_price * (1 - STOP_LOSS_PERCENT)
        take_profit = entry_price * (1 + TAKE_PROFIT_PERCENT)
    else:
        stop_loss = entry_price * (1 + STOP_LOSS_PERCENT)
        take_profit = entry_price * (1 - TAKE_PROFIT_PERCENT)
    return stop_loss, take_profit

def fetch_alpha_vantage(symbol, interval, function, api_key, outputsize="compact"):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": "XAU",
        "to_symbol": "USD",
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize
    }
    response = requests.get(url, params=params)
    data = response.json()
    time_series_key = f"Time Series FX ({interval})"
    if time_series_key not in data:
        # Print all non-time-series keys for troubleshooting
        for key in data:
            st.error(f"{key}: {data[key]}")
        st.error(f"No '{time_series_key}' found in Alpha Vantage response. Response contains: {list(data.keys())}")
        return None
    timeseries = data[time_series_key]
    df = pd.DataFrame(timeseries).T
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close"
    })
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

# --- Streamlit UI ---
timeframe = st.selectbox("Select Timeframe", options=list(TIMEFRAME_MAP.keys()), index=2)
interval, function = TIMEFRAME_MAP[timeframe]

if not API_KEY:
    st.warning("Please enter your Alpha Vantage API Key to continue.")
    st.stop()

@st.cache_data(ttl=60, show_spinner=True)
def load_data():
    time.sleep(0.1)  # Add a small delay to avoid hitting rate limits
    df = fetch_alpha_vantage(GOLD_SYMBOL, interval, function, API_KEY, outputsize="full")
    if df is None or len(df) < 22:
        return None
    return df

df = load_data()
if df is None:
    st.error("Failed to load data. Please check your API key or try again later.")
    st.stop()

# --- Technicals & Signal ---
df['RSI'] = calculate_rsi(df['close'], RSI_PERIOD)
PP, R1, S1 = calculate_pivot_levels(df)
latest = df.iloc[-1]
previous = df.iloc[-2]
rsi_value = to_scalar(latest['RSI'])
current_price = to_scalar(latest['close'])
previous_close = to_scalar(previous['close'])
price_delta = current_price - previous_close

# --- Display ---
st.markdown(f"### ⏱️ {timeframe} Frame")
st.metric(
    label="Gold Price", 
    value=f"${current_price:.2f}",
    delta=f"{price_delta:.2f}"
)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("RSI (14)", f"{rsi_value:.2f}")
    if rsi_value >= RSI_OVERBOUGHT:
        st.markdown("🔴 Overbought")
    elif rsi_value <= RSI_OVERSOLD:
        st.markdown("🟢 Oversold")
with col2:
    st.metric("Pivot Point", f"${PP:.2f}")
with col3:
    st.metric("R1 / S1", f"R1: ${R1:.2f} / S1: ${S1:.2f}")

signal = "🟡 HOLD"
stop_loss = None
take_profit = None
trade_type = None

if current_price <= S1 and rsi_value <= RSI_OVERSOLD:
    signal = "🟢 BUY"
    trade_type = 'BUY'
    stop_loss, take_profit = calculate_risk_levels(current_price, trade_type)
elif current_price >= R1 and rsi_value >= RSI_OVERBOUGHT:
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

# --- Footer with timestamp and user info ---
NOW_UTC = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
st.markdown("---")
st.caption(f"Last updated: {NOW_UTC} UTC")
st.caption("Created by: GillyMwazala")
st.caption("Data: Alpha Vantage")

st.sidebar.markdown("### App Information")
st.sidebar.caption("Version: 1.0.0")
st.sidebar.caption(f"Last Updated: {NOW_UTC}")
st.sidebar.caption("Developer: GillyMwazala")
