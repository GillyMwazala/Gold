import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ---- CONFIG ----
st.set_page_config(page_title="Gold Intraday Signal (Alpha Vantage)", layout="centered")
st.title("📊 Gold Signal Generator (Alpha Vantage)")

# ---- PARAMETERS ----
RISK_REWARD_RATIO = 2.0
STOP_LOSS_PERCENT = 0.005
TAKE_PROFIT_PERCENT = STOP_LOSS_PERCENT * RISK_REWARD_RATIO

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
ATR_PERIOD = 14

# ---- USER INPUTS ----
st.sidebar.header("Settings")
timeframe_map = {"1 Hour": "60min", "15 Minutes": "15min", "5 Minutes": "5min"}
selected_tf = st.sidebar.radio("Timeframe", list(timeframe_map.keys()), index=2)
selected_tf_alpha = timeframe_map[selected_tf]

api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
symbol = st.sidebar.text_input("Symbol (default XAUUSD)", value="XAUUSD")

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

def calculate_technical_indicators(df):
    try:
        df['RSI'] = calculate_rsi(df['close'], RSI_PERIOD)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=ATR_PERIOD).mean()
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

@st.cache_data(ttl=60)
def load_intraday(symbol, interval, api_key):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": symbol[:3],  # e.g. 'XAU'
        "to_symbol": symbol[3:],    # e.g. 'USD'
        "interval": interval,
        "apikey": api_key,
        "outputsize": "full"
    }
    r = requests.get(base_url, params=params)
    if r.status_code != 200:
        st.error("Could not retrieve data from Alpha Vantage")
        return None
    data = r.json()
    key = f"Time Series FX ({interval})"
    if key not in data:
        st.error(f"No data returned for {symbol} {interval}. Error: {data.get('Note') or data.get('Error Message') or data}")
        return None
    df = pd.DataFrame(data[key]).T
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close"
    })
    df.index = pd.to_datetime(df.index)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    df = df.sort_index()
    return df

def calculate_pivot_levels(df):
    prev = df.iloc[-2]  # previous completed period
    H = to_scalar(prev['high'])
    L = to_scalar(prev['low'])
    C = to_scalar(prev['close'])
    PP = (H + L + C) / 3
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    return PP, R1, S1

def calculate_risk_levels(entry_price, trade_type='BUY'):
    if trade_type == 'BUY':
        stop_loss = entry_price * (1 - STOP_LOSS_PERCENT)
        take_profit = entry_price * (1 + TAKE_PROFIT_PERCENT)
    else:  # SELL
        stop_loss = entry_price * (1 + STOP_LOSS_PERCENT)
        take_profit = entry_price * (1 - TAKE_PROFIT_PERCENT)
    return stop_loss, take_profit

# ---- MAIN ----
if api_key:
    df = load_intraday(symbol, selected_tf_alpha, api_key)
    if df is not None and len(df) > RSI_PERIOD + 2:
        df = calculate_technical_indicators(df)
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        PP, R1, S1 = calculate_pivot_levels(df)

        # Trading signal
        rsi = to_scalar(latest['RSI'])
        price = to_scalar(latest['close'])
        atr = to_scalar(latest['ATR'])
        signal = "🟡 HOLD"
        stop_loss = None
        take_profit = None

        if price <= S1 and rsi < RSI_OVERSOLD:
            signal = "🟢 BUY"
            stop_loss, take_profit = calculate_risk_levels(price, 'BUY')
        elif price >= R1 and rsi > RSI_OVERBOUGHT:
            signal = "🔴 SELL"
            stop_loss, take_profit = calculate_risk_levels(price, 'SELL')

        # --- Display ---
        st.markdown(f"### ⏰ Timeframe: {selected_tf} ({selected_tf_alpha})")
        st.metric("Gold Price", f"${price:.2f}")
        st.metric("RSI", f"{rsi:.2f}")
        st.metric("ATR",
