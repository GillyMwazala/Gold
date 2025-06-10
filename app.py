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

def calculate_technical_indicators(df):
    """Calculate multiple technical indicators for better signal generation"""
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
        
        # ATR (Average True Range)
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

@st.cache_data(ttl=60)
def load_intraday():
    try:
        # Get more historical data for better analysis
        df = yf.download("GC=F", interval="5m", period="5d", progress=False)
        if df.empty:
            raise ValueError("No data received from Yahoo Finance")
        df = calculate_technical_indicators(df)
        return df.dropna()
    except URLError as e:
        st.error(f"Unable to fetch data. Please check your internet connection. Error: {e.reason}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def analyze_market_context(df):
    """Analyze overall market context"""
    latest = df.iloc[-1]
    context = {
        'trend': 'neutral',
        'strength': 'weak',
        'volatility': 'normal'
    }
    
    # Trend Analysis
    if latest['EMA_short'] > latest['EMA_medium'] > latest['EMA_long']:
        context['trend'] = 'strong_uptrend'
    elif latest['EMA_short'] < latest['EMA_medium'] < latest['EMA_long']:
        context['trend'] = 'strong_downtrend'
    elif latest['EMA_short'] > latest['EMA_long']:
        context['trend'] = 'uptrend'
    elif latest['EMA_short'] < latest['EMA_long']:
        context['trend'] = 'downtrend'
    
    # Trend Strength
    strength_abs = abs(latest['Trend_Strength'])
    if strength_abs > df['Trend_Strength'].std() * 2:
        context['strength'] = 'strong'
    elif strength_abs > df['Trend_Strength'].std():
        context['strength'] = 'moderate'
    
    # Volatility
    if latest['ATR'] > df['ATR'].mean() * 1.5:
        context['volatility'] = 'high'
    elif latest['ATR'] < df['ATR'].mean() * 0.5:
        context['volatility'] = 'low'
    
    return context

def calculate_confidence_score(price, rsi, macd_hist, context, pivot_distance):
    """Calculate a confidence score for the signal"""
    confidence = 50  # Base confidence
    
    # RSI contribution
    if rsi < 20 or rsi > 80:
        confidence += 15
    elif rsi < 30 or rsi > 70:
        confidence += 10
    
    # MACD contribution
    if abs(macd_hist) > df['MACD_Histogram'].std() * 2:
        confidence += 10
    
    # Trend alignment
    if (context['trend'].endswith('uptrend') and price > pivot_distance) or \
       (context['trend'].endswith('downtrend') and price < pivot_distance):
        confidence += 15
    
    # Strength and volatility adjustments
    if context['strength'] == 'strong':
        confidence += 10
    if context['volatility'] == 'high':
        confidence -= 10
    elif context['volatility'] == 'low':
        confidence += 5
        
    return min(confidence, 100)  # Cap at 100%

def get_enhanced_signal(df, PP, R1, S1):
    try:
        if df is None or PP is None:
            raise ValueError("Missing required data")
            
        latest = df.iloc[-1]
        price = float(latest['Close'])
        rsi = float(latest['RSI'])
        macd_hist = float(latest['MACD_Histogram'])
        
        # Convert pivot levels to float
        S1 = float(S1)
        R1 = float(R1)
        PP = float(PP)
        
        # Analyze market context
        context = analyze_market_context(df)
        
        signal = "Hold"
        entry_price = None
        stop_loss = None
        take_profit = None
        trade_type = None
        confidence = 0

        # Enhanced signal generation with multiple confirmations
        if price <= S1 and rsi < RSI_OVERSOLD:
            pivot_distance = abs(price - S1)
            if latest['MACD_Histogram'] > 0 and latest['EMA_short'] > latest['EMA_medium']:
                trade_type = 'BUY'
                signal = "🟢 Strong Buy Signal"
                entry_price = price
                confidence = calculate_confidence_score(price, rsi, macd_hist, context, pivot_distance)
            elif latest['MACD_Histogram'] > 0:
                trade_type = 'BUY'
                signal = "🟡 Moderate Buy Signal"
                entry_price = price
                confidence = calculate_confidence_score(price, rsi, macd_hist, context, pivot_distance) * 0.8
                
        elif price >= R1 and rsi > RSI_OVERBOUGHT:
            pivot_distance = abs(price - R1)
            if latest['MACD_Histogram'] < 0 and latest['EMA_short'] < latest['EMA_medium']:
                trade_type = 'SELL'
                signal = "🔴 Strong Sell Signal"
                entry_price = price
                confidence = calculate_confidence_score(price, rsi, macd_hist, context, pivot_distance)
            elif latest['MACD_Histogram'] < 0:
                trade_type = 'SELL'
                signal = "🟡 Moderate Sell Signal"
                entry_price = price
                confidence = calculate_confidence_score(price, rsi, macd_hist, context, pivot_distance) * 0.8

        # Calculate risk levels if we have a trade
        if entry_price:
            stop_loss, take_profit = calculate_risk_levels(entry_price, trade_type)
            # Adjust based on ATR
            atr = float(latest['ATR'])
            if context['volatility'] == 'high':
                stop_loss = adjust_risk_levels_for_volatility(entry_price, stop_loss, atr, trade_type)

        return {
            'signal': signal,
            'price': price,
            'rsi': rsi,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trade_type': trade_type,
            'confidence': confidence,
            'context': context,
            'macd_histogram': macd_hist,
            'atr': float(latest['ATR'])
        }
    except Exception as e:
        st.error(f"Error generating signal: {str(e)}")
        return None

def adjust_risk_levels_for_volatility(entry_price, stop_loss, atr, trade_type):
    """Adjust stop loss based on ATR for high volatility periods"""
    atr_multiplier = 1.5
    if trade_type == 'BUY':
        new_stop = entry_price - (atr * atr_multiplier)
        return max(new_stop, stop_loss)  # Use the wider stop loss
    else:
        new_stop = entry_price + (atr * atr_multiplier)
        return min(new_stop, stop_loss)  # Use the wider stop loss

# Main execution pipeline
try:
    df = load_intraday()
    if df is None:
        st.warning("Unable to load market data. Please try again later.")
        st.stop()

    PP, R1, S1 = get_pivot_levels()
    if None in (PP, R1, S1):
        st.warning("Unable to calculate pivot levels. Please try again later.")
        st.stop()

    # Generate enhanced signal
    signal_data = get_enhanced_signal(df, PP, R1, S1)
    
    if signal_data:
        # Display market context
        st.markdown("### 🌍 Market Context")
        context = signal_data['context']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Trend:** {context['trend'].replace('_', ' ').title()}")
        with col2:
            st.write(f"**Strength:** {context['strength'].title()}")
        with col3:
            st.write(f"**Volatility:** {context['volatility'].title()}")

        # Display technical indicators
        st.markdown("### 📊 Technical Indicators")
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        
        with tech_col1:
            st.write(f"**RSI (14):** {signal_data['rsi']:.2f}")
            rsi_color = 'red' if signal_data['rsi'] > 70 else 'green' if signal_data['rsi'] < 30 else 'orange'
            st.markdown(f"<span style='color:{rsi_color}'>{'Overbought' if signal_data['rsi'] > 70 else 'Oversold' if signal_data['rsi'] < 30 else 'Neutral'}</span>", unsafe_allow_html=True)

        with tech_col2:
            st.write(f"**MACD Histogram:** {signal_data['macd_histogram']:.4f}")
            macd_color = 'green' if signal_data['macd_histogram'] > 0 else 'red'
            st.markdown(f"<span style='color:{macd_color}'>{'Bullish' if signal_data['macd_histogram'] > 0 else 'Bearish'}</span>", unsafe_allow_html=True)

        with tech_col3:
            st.write(f"**ATR:** {signal_data['atr']:.2f}")
            
        # Display signal and confidence
        st.markdown("### 🎯 Trading Signal")
        signal_col1, signal_col2 = st.columns(2)
        
        with signal_col1:
            st.markdown(f"## {signal_data['signal']}")
            
        with signal_col2:
            confidence = signal_data['confidence']
            confidence_color = 'green' if confidence > 70 else 'orange' if confidence > 50 else 'red'
            st.markdown(f"**Confidence Score:** <span style='color:{confidence_color}'>{confidence:.1f}%</span>", unsafe_allow_html=True)

        # Display trade details if there's an entry
        if signal_data['entry_price']:
            st.markdown("### 💹 Trade Details")
            trade_col1, trade_col2, trade_col3 = st.columns(3)
            
            with trade_col1:
                st.success(f"📌 Entry: ${signal_data['entry_price']:.2f}")
                
            with trade_col2:
                st.error(f"🛑 Stop Loss: ${signal_data['stop_loss']:.2f}")
                
            with trade_col3:
                st.info(f"🎯 Take Profit: ${signal_data['take_profit']:.2f}")

            # Calculate potential profit/loss
            if signal_data['trade_type'] == 'BUY':
                risk = signal_data['entry_price'] - signal_data['stop_loss']
                reward = signal_data['take_profit'] - signal_data['entry_price']
            else:
                risk = signal_data['stop_loss'] - signal_data['entry_price']
                reward = signal_data['entry_price'] - signal_data['take_profit']

            st.markdown("### 📊 Risk Management")
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                st.write(f"**Risk Amount:** ${risk:.2f}")
                st.write(f"**Reward Amount:** ${reward:.2f}")
                
            with risk_col2:
                st.write(f"**Risk:Reward Ratio:** 1:{RISK_REWARD_RATIO}")
                st.write(f"**Position Size Suggestion:** Based on 1% account risk")

    # Display timestamp
    st.markdown("---")
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
