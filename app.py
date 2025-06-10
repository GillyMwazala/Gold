import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

st.set_page_config(page_title="Gold Intraday Trader", layout="wide")
st.title("📈 Gold (XAU/USD) Intraday Strategy – Pivot + RSI")

# 1. Load 5-min intraday data
@st.cache_data(ttl=60)
def load_intraday():
    df = yf.download("GC=F", interval="5m", period="1d", progress=False)
    return df.dropna()

# 2. Load previous day for pivot levels
def get_pivot_levels():
    hist = yf.download("GC=F", interval="1d", period="2d", progress=False)
    prev_day = hist.iloc[0]
    H, L, C = prev_day['High'], prev_day['Low'], prev_day['Close']
    PP = (H + L + C) / 3
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    R2 = PP + (R1 - S1)
    S2 = PP - (R1 - S1)
    return PP, R1, S1, R2, S2

# 3. Add indicators
def add_indicators(df):
    df['RSI'] = ta.rsi(df['Close'], length=14)
    return df

# 4. Generate signal
def get_signal(df, PP, R1, S1):
    latest = df.iloc[-1]
    price = latest['Close']
    rsi = latest['RSI']
    signal = "Hold"
    if price <= S1 and rsi < 30:
        signal = "🟢 Buy"
    elif price >= R1 and rsi > 70:
        signal = "🔴 Sell"
    return signal, price, rsi

# Run pipeline
df = load_intraday()
PP, R1, S1, R2, S2 = get_pivot_levels()
df = add_indicators(df)
signal, price, rsi = get_signal(df, PP, R1, S1)

# Display metrics
st.subheader(f"💰 Current Price: ${price:.2f}")
st.markdown(f"**📉 RSI (14):** {rsi:.2f}")
st.markdown(f"**📊 Pivot Point:** {PP:.2f} | **R1:** {R1:.2f} | **S1:** {S1:.2f}")
st.markdown(f"### 🚦 Trading Signal: {signal}")

# Chart
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name="Gold"
)])

# Add pivot levels
fig.add_hline(y=PP, line=dict(color="gray", dash="dot"), annotation_text="Pivot", annotation_position="top left")
fig.add_hline(y=R1, line=dict(color="red", dash="dot"), annotation_text="R1", annotation_position="top left")
fig.add_hline(y=S1, line=dict(color="green", dash="dot"), annotation_text="S1", annotation_position="bottom left")

fig.update_layout(title="Gold Intraday Chart (5m)", height=500, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.caption("Built with 💛 using Streamlit, Yahoo Finance, and Plotly")
