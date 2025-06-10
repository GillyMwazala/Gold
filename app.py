import streamlit as st
import yfinance as yf
import pandas as pd
from urllib.error import URLError
from datetime import datetime

# Configure the page
st.set_page_config(page_title="Gold Intraday Signal", layout="centered")
st.title("📊 Gold Intraday Signal – Pivot + RSI Strategy")

# Constants for risk management
RISK_REWARD_RATIO = 2.0  # 1:2 risk-reward ratio
STOP_LOSS_PERCENT = 0.005  # 0.5% stop loss
TAKE_PROFIT_PERCENT = STOP_LOSS_PERCENT * RISK_REWARD_RATIO  # 1% take profit

def calculate_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
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

def calculate_risk_levels(entry_price, trade_type='BUY'):
    """Calculate Stop Loss and Take Profit levels based on trade type"""
    if trade_type == 'BUY':
        stop_loss = entry_price * (1 - STOP_LOSS_PERCENT)
        take_profit = entry_price * (1 + TAKE_PROFIT_PERCENT)
    else:  # SELL
        stop_loss = entry_price * (1 + STOP_LOSS_PERCENT)
        take_profit = entry_price * (1 - TAKE_PROFIT_PERCENT)
    return stop_loss, take_profit

def get_signal(df, PP, R1, S1):
    try:
        if df is None or PP is None:
            raise ValueError("Missing required data")
            
        df['RSI'] = calculate_rsi(df['Close'])
        latest = df.iloc[-1]
        
        price = float(latest['Close'])
        rsi = float(latest['RSI'])
        S1 = float(S1)
        R1 = float(R1)
        
        signal = "Hold"
        entry_price = None
        stop_loss = None
        take_profit = None
        trade_type = None

        # Generate signals with risk management levels
        if price <= S1 and rsi < 30:
            trade_type = 'BUY'
            signal = "🟢 Entry: BUY"
            entry_price = price
            stop_loss, take_profit = calculate_risk_levels(entry_price, trade_type)
        elif price >= R1 and rsi > 70:
            trade_type = 'SELL'
            signal = "🔴 Entry: SELL"
            entry_price = price
            stop_loss, take_profit = calculate_risk_levels(entry_price, trade_type)
        else:
            signal = "🟡 No Trade"

        return signal, price, rsi, entry_price, stop_loss, take_profit, trade_type
    except Exception as e:
        st.error(f"Error generating signal: {str(e)}")
        return "Error", None, None, None, None, None, None

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

    # Generate signal with risk levels
    signal, price, rsi, entry_price, stop_loss, take_profit, trade_type = get_signal(df, PP, R1, S1)
    
    # Display Strategy Summary
    st.markdown("### ⚙️ Strategy Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        if price is not None:
            st.markdown(f"**Current Price:** ${price:.2f}")
        if rsi is not None:
            st.markdown(f"**RSI (14):** {rsi:.2f}")
        st.markdown(f"**Pivot Point:** ${PP:.2f}")

    with col2:
        st.markdown(f"**R1 (Resistance):** ${R1:.2f}")
        st.markdown(f"**S1 (Support):** ${S1:.2f}")
        
    st.markdown("---")
    st.markdown(f"## 🚦 Signal: {signal}")
    
    # Display trade details if there's an entry
    if entry_price and stop_loss and take_profit:
        st.success(f"📌 Entry Price: ${entry_price:.2f}")
        
        # Calculate potential profit/loss
        if trade_type == 'BUY':
            risk_amount = entry_price - stop_loss
            reward_amount = take_profit - entry_price
        else:  # SELL
            risk_amount = stop_loss - entry_price
            reward_amount = entry_price - take_profit
            
        # Display risk management levels
        risk_mgmt_col1, risk_mgmt_col2 = st.columns(2)
        
        with risk_mgmt_col1:
            st.error(f"🛑 Stop Loss: ${stop_loss:.2f}")
            st.write(f"Risk: ${risk_amount:.2f}")
            
        with risk_mgmt_col2:
            st.success(f"🎯 Take Profit: ${take_profit:.2f}")
            st.write(f"Reward: ${reward_amount:.2f}")
            
        # Display Risk:Reward ratio
        st.info(f"Risk:Reward Ratio = 1:{RISK_REWARD_RATIO}")
        
    else:
        st.info("📌 No valid entry at the moment.")

    # Display timestamp and caption
    st.markdown("---")
    st.caption("Signal generated using Pivot Points and RSI (14). Refresh to update.")
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
