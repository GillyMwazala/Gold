import streamlit as st
st.set_page_config(page_title="Gold Intraday Signal – Backtest Improved Entries", layout="centered")

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

st.title("📊 Gold Intraday Signal – Backtest with Confirmed & Scaled Entries")

# --- Constants and Parameters ---
GOLD_SYMBOL = "XAU/USD"
TIMEFRAME_MAP = {
    "5 min": "5min",
    "15 min": "15min",
    "1 hour": "1h",
}
RISK_REWARD_RATIO = st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 2.0, 0.1)
STOP_LOSS_ATR = st.sidebar.slider("Stop Loss (x ATR)", 0.5, 5.0, 2.0, 0.1)
SCALE_LEVELS = st.sidebar.slider("Max Scaling Entries", 1, 4, 2, 1)
SCALE_ATR_BUFFER = st.sidebar.slider("ATR scale-in buffer (xATR)", 0.05, 1.0, 0.2, 0.05)
RSI_PERIOD = st.sidebar.slider("RSI Period", 2, 30, 14, 1)
RSI_OVERSOLD = st.sidebar.slider("RSI Oversold", 10, 40, 30, 1)
RSI_OVERBOUGHT = st.sidebar.slider("RSI Overbought", 60, 90, 70, 1)
TREND_SMA = st.sidebar.slider("Trend SMA Period", 10, 100, 20, 2)

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

def fetch_twelve_data(symbol, interval, api_key, outputsize=1500):
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
    df = fetch_twelve_data(GOLD_SYMBOL, interval, API_KEY, outputsize=1500)
    if df is None or len(df) < 60:
        return None
    return df

df = load_data()
if df is None:
    st.error("Failed to load data. Please check your API key or try again later.")
    st.stop()

# --- Technicals ---
df['RSI'] = calculate_rsi(df['close'], RSI_PERIOD)
df['ATR'] = calculate_atr(df)
df['SMA'] = df['close'].rolling(window=TREND_SMA).mean()
df['prev_close'] = df['close'].shift(1)
df['prev_high'] = df['high'].shift(1)
df['prev_low'] = df['low'].shift(1)

# --- Backtest with improved confirmations & scaling into winners only ---
def backtest_confirmed_scaled(df):
    trades = []
    open_trade = None
    equity_curve = []
    timestamp_curve = []
    equity = 0

    for i in range(TREND_SMA + 2, len(df)-1):  # ensure enough lookback for SMA
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        curr_time = df.index[i]
        atr = to_scalar(row['ATR'])

        # --- ENTRY LOGIC WITH CONFIRMATION ---
        # Buy confirmation: RSI crosses below oversold, price closes above prev high, AND above SMA (trend filter)
        buy_confirm = (
            prev_row['RSI'] > RSI_OVERSOLD and row['RSI'] <= RSI_OVERSOLD and
            row['close'] > prev_row['high'] and
            row['close'] > row['SMA']
        )
        # Sell confirmation: RSI crosses above overbought, price closes below prev low, AND below SMA
        sell_confirm = (
            prev_row['RSI'] < RSI_OVERBOUGHT and row['RSI'] >= RSI_OVERBOUGHT and
            row['close'] < prev_row['low'] and
            row['close'] < row['SMA']
        )

        # --- SCALING LOGIC: Only add to winning trades ---
        # For scaling, wait for price to move in favor by scale_atr_buffer*ATR before next add
        if buy_confirm and open_trade is None:
            entry = row['close']
            stop = entry - STOP_LOSS_ATR * atr
            target = entry + STOP_LOSS_ATR * atr * RISK_REWARD_RATIO
            open_trade = {
                'side': "BUY",
                'lots': [entry],
                'entry_times': [curr_time],
                'stops': [stop],
                'targets': [target],
                'scale_count': 1,
                'last_scale_price': entry
            }
        elif open_trade and open_trade['side'] == "BUY" and open_trade['scale_count'] < SCALE_LEVELS:
            # Only scale in if price in profit by at least SCALE_ATR_BUFFER*ATR from last scale
            if row['close'] > open_trade['last_scale_price'] + SCALE_ATR_BUFFER * atr:
                entry = row['close']
                stop = entry - STOP_LOSS_ATR * atr
                target = entry + STOP_LOSS_ATR * atr * RISK_REWARD_RATIO
                open_trade['lots'].append(entry)
                open_trade['entry_times'].append(curr_time)
                open_trade['stops'].append(stop)
                open_trade['targets'].append(target)
                open_trade['scale_count'] += 1
                open_trade['last_scale_price'] = entry

        if sell_confirm and open_trade is None:
            entry = row['close']
            stop = entry + STOP_LOSS_ATR * atr
            target = entry - STOP_LOSS_ATR * atr * RISK_REWARD_RATIO
            open_trade = {
                'side': "SELL",
                'lots': [entry],
                'entry_times': [curr_time],
                'stops': [stop],
                'targets': [target],
                'scale_count': 1,
                'last_scale_price': entry
            }
        elif open_trade and open_trade['side'] == "SELL" and open_trade['scale_count'] < SCALE_LEVELS:
            # Only scale in if price in profit by at least SCALE_ATR_BUFFER*ATR from last scale
            if row['close'] < open_trade['last_scale_price'] - SCALE_ATR_BUFFER * atr:
                entry = row['close']
                stop = entry + STOP_LOSS_ATR * atr
                target = entry - STOP_LOSS_ATR * atr * RISK_REWARD_RATIO
                open_trade['lots'].append(entry)
                open_trade['entry_times'].append(curr_time)
                open_trade['stops'].append(stop)
                open_trade['targets'].append(target)
                open_trade['scale_count'] += 1
                open_trade['last_scale_price'] = entry

        # --- EXIT LOGIC for open trade: check each lot separately (partial exits) ---
        if open_trade:
            exit_indices = []
            trade_results = []
            for idx, entry in enumerate(open_trade['lots']):
                stop = open_trade['stops'][idx]
                target = open_trade['targets'][idx]
                entry_time = open_trade['entry_times'][idx]
                exit_price = None
                exit_time = None
                result = None
                # For BUY
                if open_trade['side'] == "BUY":
                    # Stop loss
                    if row['low'] <= stop:
                        exit_price = stop
                        exit_time = curr_time
                        result = "Stopped"
                    # Target
                    elif row['high'] >= target:
                        exit_price = target
                        exit_time = curr_time
                        result = "Target"
                # For SELL
                else:
                    if row['high'] >= stop:
                        exit_price = stop
                        exit_time = curr_time
                        result = "Stopped"
                    elif row['low'] <= target:
                        exit_price = target
                        exit_time = curr_time
                        result = "Target"
                if exit_price is not None:
                    pnl = (exit_price - entry) if open_trade['side'] == "BUY" else (entry - exit_price)
                    trade_results.append({
                        'side': open_trade['side'],
                        'entry': entry,
                        'entry_time': entry_time,
                        'exit': exit_price,
                        'exit_time': exit_time,
                        'result': result,
                        'pnl': pnl,
                        'scale': idx+1,
                    })
                    exit_indices.append(idx)

            # Remove exited lots (do in reverse to avoid index errors)
            for idx in sorted(exit_indices, reverse=True):
                del open_trade['lots'][idx]
                del open_trade['entry_times'][idx]
                del open_trade['stops'][idx]
                del open_trade['targets'][idx]
                open_trade['scale_count'] -= 1

            # If any trades closed, record them
            if trade_results:
                for tr in trade_results:
                    trades.append(tr)
                    equity += tr['pnl']
            # If all lots are closed, reset open_trade
            if open_trade['scale_count'] == 0:
                open_trade = None

        equity_curve.append(equity)
        timestamp_curve.append(curr_time)

    # At the end, close any open trades at last price
    if open_trade:
        row = df.iloc[-1]
        for idx, entry in enumerate(open_trade['lots']):
            exit_price = row['close']
            pnl = (exit_price - entry) if open_trade['side'] == "BUY" else (entry - exit_price)
            trades.append({
                'side': open_trade['side'],
                'entry': entry,
                'entry_time': open_trade['entry_times'][idx],
                'exit': exit_price,
                'exit_time': df.index[-1],
                'result': "ManualClose",
                'pnl': pnl,
                'scale': idx+1,
            })
            equity += pnl
        open_trade = None
        equity_curve.append(equity)
        timestamp_curve.append(df.index[-1])

    return trades, equity_curve, timestamp_curve

# --- Display current status ---
current_price = to_scalar(df.iloc[-1]['close'])
previous_close = to_scalar(df.iloc[-2]['close'])
price_delta = current_price - previous_close
rsi_value = to_scalar(df.iloc[-1]['RSI'])

st.markdown(f"### ⏱️ {timeframe} Frame")
st.metric("Gold Price", f"${current_price:.2f}", delta=f"{price_delta:.2f}")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("RSI", f"{rsi_value:.2f}")
with col2:
    st.metric("ATR", f"{df.iloc[-1]['ATR']:.2f}")
with col3:
    st.metric(f"SMA{TREND_SMA}", f"{df.iloc[-1]['SMA']:.2f}")

run_backtest = st.button("Run Backtest")

if run_backtest:
    st.info("Running backtest, please wait...")
    trades, equity_curve, timestamp_curve = backtest_confirmed_scaled(df)
    df_trades = pd.DataFrame(trades)

    # --- Safe summary stats block ---
    if not df_trades.empty and 'pnl' in df_trades.columns:
        total_pnl = df_trades['pnl'].sum()
        win_trades = df_trades[df_trades['pnl'] > 0]
        loss_trades = df_trades[df_trades['pnl'] <= 0]
        win_rate = len(win_trades) / len(df_trades) * 100 if len(df_trades) else 0
        avg_win = win_trades['pnl'].mean() if not win_trades.empty else 0
        avg_loss = loss_trades['pnl'].mean() if not loss_trades.empty else 0
        profit_factor = win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum()) if not loss_trades.empty and abs(loss_trades['pnl'].sum()) > 0 else np.inf
    else:
        total_pnl = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    max_drawdown = min(0, min(equity_curve) - max(equity_curve[:equity_curve.index(min(equity_curve))]) if equity_curve else 0)

    st.success(f"Backtest completed. {len(df_trades)} exits (partial lots) recorded.")

    # Display stats
    st.markdown("#### Backtest Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Trades (lots closed)", len(df_trades))
    c2.metric("Win Rate (%)", f"{win_rate:.1f}")
    c3.metric("Net PnL ($)", f"{total_pnl:.2f}")
    c1.metric("Profit Factor", f"{profit_factor:.2f}")
    c2.metric("Avg Win", f"{avg_win:.2f}")
    c3.metric("Avg Loss", f"{avg_loss:.2f}")
    c1.metric("Max Drawdown", f"{max_drawdown:.2f}")

    # Show sample trades
    st.markdown("#### Trade Log (most recent)")
    if not df_trades.empty:
        st.dataframe(df_trades.sort_values('exit_time', ascending=False).head(20), use_container_width=True)
    else:
        st.info("No trades were generated by the current strategy/parameters.")

    # Show equity curve
    st.markdown("#### Equity Curve")
    if len(equity_curve) > 0:
        eq_df = pd.DataFrame({'equity': equity_curve}, index=pd.to_datetime(timestamp_curve))
        st.line_chart(eq_df)
    else:
        st.info("No equity curve available.")

    # Show open/close signals on the price chart
    import plotly.graph_objects as go
    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ))
    # Add buy/sell markers
    if not df_trades.empty:
        buy_trades = df_trades[df_trades['side'] == 'BUY']
        sell_trades = df_trades[df_trades['side'] == 'SELL']
        price_fig.add_trace(go.Scatter(
            x=buy_trades['entry_time'],
            y=buy_trades['entry'],
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy Entries'
        ))
        price_fig.add_trace(go.Scatter(
            x=sell_trades['entry_time'],
            y=sell_trades['entry'],
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell Entries'
        ))
        price_fig.add_trace(go.Scatter(
            x=buy_trades['exit_time'],
            y=buy_trades['exit'],
            mode='markers',
            marker=dict(color='lime', symbol='star', size=8),
            name='Buy Exits'
        ))
        price_fig.add_trace(go.Scatter(
            x=sell_trades['exit_time'],
            y=sell_trades['exit'],
            mode='markers',
            marker=dict(color='orange', symbol='star', size=8),
            name='Sell Exits'
        ))
    st.plotly_chart(price_fig, use_container_width=True)

# --- Footer ---
NOW_UTC = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
st.markdown("---")
st.caption(f"Last updated: {NOW_UTC} UTC")
st.caption("Created by: GillyMwazala")
st.caption("Data: Twelve Data")

st.sidebar.markdown("### App Information")
st.sidebar.caption("Version: 2.2.0")
st.sidebar.caption(f"Last Updated: {NOW_UTC}")
st.sidebar.caption("Developer: GillyMwazala")
