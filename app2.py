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

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_additional_pivots(df):
    prev_day = df.iloc[-2]
    H = to_scalar(prev_day['high'])
    L = to_scalar(prev_day['low'])
    C = to_scalar(prev_day['close'])
    PP = (H + L + C) / 3
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    R3 = H + 2 * (PP - L)
    S3 = L - 2 * (H - PP)
    return PP, R1, S1, R2, S2, R3, S3

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
df['ATR'] = calculate_atr(df)
df['SMA20'] = df['close'].rolling(window=20).mean()

PP, R1, S1, R2, S2, R3, S3 = calculate_additional_pivots(df)
latest = df.iloc[-1]
previous = df.iloc[-2]
rsi_value = to_scalar(latest['RSI'])
prev_rsi = to_scalar(previous['RSI'])
current_price = to_scalar(latest['close'])
previous_close = to_scalar(previous['close'])
price_delta = current_price - previous_close

entries = []

# --- Entry 1: RSI cross S1/R1, SMA20 confirmation ---
if (prev_rsi > RSI_OVERSOLD and rsi_value <= RSI_OVERSOLD) and (latest['low'] <= S1) and (latest['close'] > df['SMA20'].iloc[-1]):
    entries.append({'type': 'BUY', 'level': S1, 'desc': "RSI cross below 30 & price touched S1 & above SMA20"})
if (prev_rsi < RSI_OVERBOUGHT and rsi_value >= RSI_OVERBOUGHT) and (latest['high'] >= R1) and (latest['close'] < df['SMA20'].iloc[-1]):
    entries.append({'type': 'SELL', 'level': R1, 'desc': "RSI cross above 70 & price touched R1 & below SMA20"})

# --- Entry 2: ATR Buffer Breakout Confirmation ---
atr_buffer = df['ATR'].iloc[-1] * 0.2
if (rsi_value <= RSI_OVERSOLD) and (latest['low'] <= (S1 - atr_buffer)):
    entries.append({'type': 'BUY', 'level': S1 - atr_buffer, 'desc': "RSI oversold & price breaks S1 by 0.2 ATR"})
if (rsi_value >= RSI_OVERBOUGHT) and (latest['high'] >= (R1 + atr_buffer)):
    entries.append({'type': 'SELL', 'level': R1 + atr_buffer, 'desc': "RSI overbought & price breaks R1 by 0.2 ATR"})

# --- Entry 3: Multiple Levels (S2/S3/R2/R3) ---
if (prev_rsi > RSI_OVERSOLD and rsi_value <= RSI_OVERSOLD) and (latest['low'] <= S2):
    entries.append({'type': 'BUY', 'level': S2, 'desc': "RSI cross below 30 & price touched S2"})
if (prev_rsi > RSI_OVERSOLD and rsi_value <= RSI_OVERSOLD) and (latest['low'] <= S3):
    entries.append({'type': 'BUY', 'level': S3, 'desc': "RSI cross below 30 & price touched S3"})
if (prev_rsi < RSI_OVERBOUGHT and rsi_value >= RSI_OVERBOUGHT) and (latest['high'] >= R2):
    entries.append({'type': 'SELL', 'level': R2, 'desc': "RSI cross above 70 & price touched R2"})
if (prev_rsi < RSI_OVERBOUGHT and rsi_value >= RSI_OVERBOUGHT) and (latest['high'] >= R3):
    entries.append({'type': 'SELL', 'level': R3, 'desc': "RSI cross above 70 & price touched R3"})

# --- Entry 4: Strong Candle (body > 0.7 * ATR) ---
candle_body = abs(latest['close'] - latest['open'])
if (rsi_value <= RSI_OVERSOLD) and (candle_body > 0.7 * df['ATR'].iloc[-1]) and (latest['close'] > latest['open']):
    entries.append({'type': 'BUY', 'level': latest['open'], 'desc': "RSI oversold & strong bullish candle"})
if (rsi_value >= RSI_OVERBOUGHT) and (candle_body > 0.7 * df['ATR'].iloc[-1]) and (latest['close'] < latest['open']):
    entries.append({'type': 'SELL', 'level': latest['open'], 'desc': "RSI overbought & strong bearish candle"})

# --- Entry 5: Signal Stacking (loop through last 3 candles) ---
for i in range(-4, -1):
    prev = df.iloc[i - 1]
    curr = df.iloc[i]
    if (to_scalar(prev['RSI']) > RSI_OVERSOLD and to_scalar(curr['RSI']) <= RSI_OVERSOLD) and (curr['low'] <= S1):
        entries.append({'type': 'BUY', 'level': S1, 'desc': f"RSI cross below 30 in candle {df.index[i].strftime('%Y-%m-%d %H:%M')}"})
    if (to_scalar(prev['RSI']) < RSI_OVERBOUGHT and to_scalar(curr['RSI']) >= RSI_OVERBOUGHT) and (curr['high'] >= R1):
        entries.append({'type': 'SELL', 'level': R1, 'desc': f"RSI cross above 70 in candle {df.index[i].strftime('%Y-%m-%d %H:%M')}"})

# --- Multiple Timeframe Confirmation (15min + 1h) ---
if interval != "1h":
    @st.cache_data(ttl=120, show_spinner=False)
    def get_1h_rsi():
        df_1h = fetch_twelve_data(GOLD_SYMBOL, "1h", API_KEY, outputsize=500)
        if df_1h is not None:
            df_1h['RSI'] = calculate_rsi(df_1h['close'], RSI_PERIOD)
        return df_1h
    df_1h = get_1h_rsi()
    if df_1h is not None:
        rsi_1h = to_scalar(df_1h['RSI'].iloc[-1])
        if rsi_value <= RSI_OVERSOLD and rsi_1h <= RSI_OVERSOLD:
            entries.append({'type': 'BUY', 'level': latest['open'], 'desc': "RSI oversold on both 15min/5min and 1h"})
        if rsi_value >= RSI_OVERBOUGHT and rsi_1h >= RSI_OVERBOUGHT:
            entries.append({'type': 'SELL', 'level': latest['open'], 'desc': "RSI overbought on both 15min/5min and 1h"})

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

st.markdown("### 🚦 Trading Signals")

if entries:
    for i, entry in enumerate(entries):
        stop_loss, take_profit = calculate_risk_levels(entry['level'], entry['type'])
        st.markdown(f"#### {entry['type']} Signal #{i+1} ({entry['desc']})")
        levels_col1, levels_col2, levels_col3 = st.columns(3)
        with levels_col1:
            st.metric("Entry", f"${entry['level']:.2f}")
        with levels_col2:
            st.metric("Stop Loss", f"${stop_loss:.2f}")
        with levels_col3:
            st.metric("Take Profit", f"${take_profit:.2f}")
        risk = abs(entry['level'] - stop_loss)
        reward = abs(take_profit - entry['level'])
        st.markdown(f"**Risk/Reward Ratio:** 1:{RISK_REWARD_RATIO}")
        st.markdown(f"Potential Risk: ${risk:.2f}")
        st.markdown(f"Potential Reward: ${reward:.2f}")
        st.markdown("---")
else:
    st.info("No active trade signals at this time.")

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
