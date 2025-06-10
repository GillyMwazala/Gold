import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Gold Intraday Signal", layout="centered")
st.title("📊 Gold Intraday Signal – Pivot + RSI Strategy")

# Custom RSI without pandas_ta
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(ttl=60)
def load_intraday():
    df = yf.download("GC=F", interval="5m", period="1d", progress=False)
    return df.dropna()

def get_pivot_levels():
    hist = yf.download("GC=F", interval="1d", period="2d", progress=False)
    prev_day = hist.iloc[0]
    H, L, C = prev_day['High'], prev_day['Low'], prev_day['Close']
    PP = (H + L + C) / 3
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    return PP, R1, S1

def get_signal(df, PP, R1, S1):
    df['RSI'] = calculate_rsi(df['Close'])
    latest = df.iloc[-1]
    price = float(latest['Close'])
    rsi = float(latest['RSI'])
    signal = "Hold"
    entry_price = None

    if price <= S1 and rsi < 30:
        signal = "🟢 Entry: BUY"
        entry_price = price
    elif price >= R1 and rsi > 70:
        signal = "🔴 Entry: SELL"
        entry_price = price
    else:
        signal = "🟡 No Trade"

    return signal, price, rsi, entry_price

# Pipeline
df = load_intraday()
PP, R1, S1 = get_pivot_levels()
signal, price, rsi, entry_price = get_signal(df, PP, R1, S1)

# Display
st.markdown("### ⚙️ Strategy Summary")
st.markdown(f"**Current Price:** ${price:.2f}")
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
