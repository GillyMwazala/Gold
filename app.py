import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from urllib.error import URLError

# Configure the page
st.set_page_config(page_title="Enhanced Gold Intraday Signal", layout="centered")
st.title("📊 Enhanced Gold Signal Generator")

# Constants for risk management and technical analysis
RISK_REWARD_RATIO = 2.0
STOP_LOSS_PERCENT = 0.005
TAKE_PROFIT_PERCENT = STOP_LOSS_PERCENT * RISK_REWARD_RATIO

# Technical Analysis Parameters
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

def calculate_rsi(series, period=14):
    """Calculate RSI (Relative Strength Index)"""
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
        st.error(f"Error in RSI calculation: {str(e)}")
        return pd.Series([np.nan] * len(series))

def get_pivot_levels():
    """Calculate pivot points based on previous day's data"""
    try:
        hist = yf.download("GC=F", interval="1d", period="2d", progress=False)
        if hist.empty:
            raise ValueError("No historical data received")
        
        prev_day = hist.iloc[0]
        H = float(prev_day['High'])
        L = float(prev_day['Low'])
        C = float(prev_day['Close'])
        
        PP = (H + L + C) / 3
        R1 = 2 * PP - L
        S1 = 2 * PP - H
        
        return PP, R1, S1
        
    except Exception as e:
        st.error(f"Error calculating pivot levels: {str(e)}")
        return None, None, None

def calculate_technical_indicators(df):
    """Calculate multiple technical indicators"""
    try:
        # RSI
        df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
        
        # EMAs
        df['EMA_short'] = df['Close'].ewm(span=EMA_SHORT, adjust=False).mean()
        df['EMA_medium'] = df['Close'].ewm(span=EMA_MEDIUM, adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=EMA_LONG, adjust=False).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=ATR_PERIOD).mean()
        
        # Trend Strength
        df['Price_Change'] = df['Close'].diff()
        df['Trend_Strength'] = df['Price_Change'].rolling(window=10).sum()
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

def calculate_risk_levels(entry_price, trade_type='BUY'):
    """Calculate Stop Loss and Take Profit levels"""
    try:
        if trade_type == 'BUY':
            stop_loss = entry_price * (1 - STOP_LOSS_PERCENT)
            take_profit = entry_price * (1 + TAKE_PROFIT_PERCENT)
        else:  # SELL
            stop_loss = entry_price * (1 + STOP_LOSS_PERCENT)
            take_profit = entry_price * (1 - TAKE_PROFIT_PERCENT)
        return stop_loss, take_profit
    except Exception as e:
        st.error(f"Error calculating risk levels: {str(e)}")
        return None, None

@st.cache_data(ttl=60)
def load_intraday():
    """Load and prepare intraday data"""
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

# Main execution pipeline
try:
    # Load market data
    df = load_intraday()
    if df is None:
        st.warning("Unable to load market data. Please try again later.")
        st.stop()

    # Calculate pivot levels
    PP, R1, S1 = get_pivot_levels()
    if None in (PP, R1, S1):
        st.warning("Unable to calculate pivot levels. Please try again later.")
        st.stop()

    # Get latest data
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    # Display current market price
    st.markdown("### 📈 Current Market Status")
    price_delta = latest['Close'] - previous['Close']
    st.metric(
        label="Gold Price", 
        value=f"${latest['Close']:.2f}",
        delta=f"{price_delta:.2f}"
    )

    # Display technical indicators
    st.markdown("### 📊 Technical Indicators")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rsi_value = float(latest['RSI'])
        st.metric("RSI (14)", f"{rsi_value:.2f}")
        if rsi_value >= RSI_OVERBOUGHT:
            st.markdown("🔴 Overbought")
        elif rsi_value <= RSI_OVERSOLD:
            st.markdown("🟢 Oversold")
            
    with col2:
        macd_hist = float(latest['MACD_Histogram'])
        st.metric("MACD Histogram", f"{macd_hist:.4f}")
        if macd_hist > 0:
            st.markdown("🟢 Bullish")
        else:
            st.markdown("🔴 Bearish")
            
    with col3:
        atr_value = float(latest['ATR'])
        st.metric("ATR", f"{atr_value:.2f}")
        
    # Display pivot levels
    st.markdown("### 🎯 Pivot Levels")
    pivot_col1, pivot_col2, pivot_col3 = st.columns(3)
    
    with pivot_col1:
        st.metric("Resistance (R1)", f"${R1:.2f}")
    with pivot_col2:
        st.metric("Pivot Point (PP)", f"${PP:.2f}")
    with pivot_col3:
        st.metric("Support (S1)", f"${S1:.2f}")

    # Generate trading signal
    current_price = float(latest['Close'])
    rsi = float(latest['RSI'])
    
    signal = "🟡 HOLD"
    stop_loss = None
    take_profit = None
    
    if current_price <= S1 and rsi <= RSI_OVERSOLD and macd_hist > 0:
        signal = "🟢 BUY"
        stop_loss, take_profit = calculate_risk_levels(current_price, 'BUY')
    elif current_price >= R1 and rsi >= RSI_OVERBOUGHT and macd_hist < 0:
        signal = "🔴 SELL"
        stop_loss, take_profit = calculate_risk_levels(current_price, 'SELL')

    # Display signal and levels
    st.markdown("### 🚦 Trading Signal")
    st.markdown(f"## {signal}")
    
    if signal != "🟡 HOLD":
        st.markdown("### 💰 Trade Levels")
        levels_col1, levels_col2, levels_col3 = st.columns(3)
        
        with levels_col1:
            st.metric("Entry", f"${current_price:.2f}")
        with levels_col2:
            st.metric("Stop Loss", f"${stop_loss:.2f}")
        with levels_col3:
            st.metric("Take Profit", f"${take_profit:.2f}")

        # Display Risk/Reward
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        st.markdown(f"Risk/Reward Ratio: 1:{RISK_REWARD_RATIO}")
        st.markdown(f"Potential Risk: ${risk:.2f}")
        st.markdown(f"Potential Reward: ${reward:.2f}")

    # Display timestamp
    st.markdown("---")
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
