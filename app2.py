import streamlit as st
st.set_page_config(page_title="Gold Intraday Signal", layout="centered")

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

# --- Require the Twelve Data API Key in secrets ---
if "TWELVE_DATA_API_KEY" not in st.secrets:
    st.error("Please add your Twelve Data API key to Streamlit secrets as 'TWELVE_DATA_API_KEY'.")
    st.stop()
API_KEY = st.secrets["TWELVE_DATA_API_KEY"]

st.title("📊 Gold Intraday Signal – Multi-Timeframe (Twelve Data)")

# --- Constants ---
GOLD_SYMBOL = "XAU/USD"
TIMEFRAME_MAP = {
    "5 min": "5min",
    "15 min": "15min",
    "1 hour": "1h",
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

def fetch_twelve_data(symbol, interval, api_key, outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
        "format": "JSON"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "values" not in data:
        for key in data:
            st.error(f"{key}: {data[key]}")
        st.error(f"No 'values' found in Twelve Data response. Response contains: {list(data.keys())}")
        return None
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={
        "datetime": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close"
    })
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df.sort_index()
    return df

# --- Streamlit UI ---
timeframe = st.selectbox("Select Timeframe", options=list(TIMEFRAME_MAP.keys()), index=2)
interval = TIMEFRAME_MAP[timeframe]

@st.cache_data(ttl=60, show_spinner=True)
def load_data():
    time.sleep(0.1)
    df = fetch_twelve_data(GOLD_SYMBOL, interval, API_KEY, outputsize=500)
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
prev_rsi = to_scalar(previous['RSI'])
current_price = to_scalar(latest['close'])
previous_close = to_scalar(previous['close'])
price_delta = current_price - previous_close

# --- Improved entry logic ---
signal = "🟡 HOLD"
stop_loss = None
take_profit = None
entry_price = None
trade_type = None

# BUY entry: RSI crosses below oversold and the current candle's low touches or goes below S1
if (prev_rsi > RSI_OVERSOLD and rsi_value <= RSI_OVERSOLD) and (latest['low'] <= S1):
    signal = "🟢 BUY"
    trade_type = 'BUY'
    entry_price = S1  # Simulate limit order at S1
    stop_loss, take_profit = calculate_risk_levels(entry_price, trade_type)
# SELL entry: RSI crosses above overbought and the current candle's high touches or goes above R1
elif (prev_rsi < RSI_OVERBOUGHT and rsi_value >= RSI_OVERBOUGHT) and (latest['high'] >= R1):
    signal = "🔴 SELL"
    trade_type = 'SELL'
    entry_price = R1  # Simulate limit order at R1
    stop_loss, take_profit = calculate_risk_levels(entry_price, trade_type)

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

st.markdown("### 🚦 Trading Signal")
st.markdown(f"## {signal}")

if signal != "🟡 HOLD" and stop_loss is not None and take_profit is not None and entry_price is not None:
    st.markdown("### 💰 Trade Levels")
    levels_col1, levels_col2, levels_col3 = st.columns(3)
    with levels_col1:
        st.metric("Entry", f"${entry_price:.2f}")
    with levels_col2:
        st.metric("Stop Loss", f"${stop_loss:.2f}")
    with levels_col3:
        st.metric("Take Profit", f"${take_profit:.2f}")

    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
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
st.caption("Data: Twelve Data")

st.sidebar.markdown("### App Information")
st.sidebar.caption("Version: 1.0.0")
st.sidebar.caption(f"Last Updated: {NOW_UTC}")
st.sidebar.caption("Developer: GillyMwazala")
