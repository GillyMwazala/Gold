import streamlit as st
st.set_page_config(page_title="Gold Intraday Signal – Backtest & Scaled Entries", layout="centered")

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

# Require the Twelve Data API Key in secrets
if "TWELVE_DATA_API_KEY" not in st.secrets:
    st.error("Please add your Twelve Data API key to Streamlit secrets as 'TWELVE_DATA_API_KEY'.")
    st.stop()
API_KEY = st.secrets["TWELVE_DATA_API_KEY"]

st.title("📊 Gold Intraday Signal – Backtest & Scaled Entries")

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
RSI_PERIOD = st.sidebar.slider("RSI Period", 2, 30, 14, 1)
RSI_OVERSOLD = st.sidebar.slider("RSI Oversold", 10, 40, 30, 1)
RSI_OVERBOUGHT = st.sidebar.slider("RSI Overbought", 60, 90, 70, 1)
SCALE_LEVELS = st.sidebar.slider("Max Scaling Entries", 1, 5, 3, 1)
SCALE_ATR_BUFFER = st.sidebar.slider("ATR scale-in buffer (xATR)", 0.1, 1.0, 0.2, 0.05)

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

def calculate_risk_levels(entry_price, trade_type='BUY', stop_loss_pct=STOP_LOSS_PERCENT, rr=RISK_REWARD_RATIO):
    if trade_type == 'BUY':
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + stop_loss_pct * rr)
    else:
        stop_loss = entry_price * (1 + stop_loss_pct)
        take_profit = entry_price * (1 - stop_loss_pct * rr)
    return stop_loss, take_profit

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
df['SMA20'] = df['close'].rolling(window=20).mean()

# --- Backtest function with scaling ---
def backtest_scaled_entries(df, show_signals=False):
    PP, R1, S1, R2, S2, R3, S3 = calculate_additional_pivots(df)
    scale_buffer = SCALE_ATR_BUFFER
    max_scales = SCALE_LEVELS

    # Each trade is a dict with keys:
    # side, lots, entries, stops, targets, entry_times, exit_time, exit_price, result, total_pnl, scale_count
    trades = []
    open_trade = None
    equity_curve = []
    timestamp_curve = []
    equity = 0

    for i in range(22, len(df)-1):  # start after enough lookback for indicators
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        curr_time = df.index[i]

        # --- ENTRY LOGIC ---
        # Buy scale-in: RSI cross below oversold AND price touches new scale level near S1/S2/S3
        scale_levels = []
        base = S1
        atr = to_scalar(row['ATR'])
        for n in range(max_scales):
            scale_levels.append(base - n * atr * scale_buffer)

        triggered_scales = []
        if prev_row['RSI'] > RSI_OVERSOLD and row['RSI'] <= RSI_OVERSOLD:
            for lvl in scale_levels:
                if row['low'] <= lvl:
                    triggered_scales.append(lvl)

        # Sell scale-in: RSI cross above overbought AND price touches new scale level near R1/R2/R3
        scale_levels_sell = []
        base_sell = R1
        for n in range(max_scales):
            scale_levels_sell.append(base_sell + n * atr * scale_buffer)

        triggered_scales_sell = []
        if prev_row['RSI'] < RSI_OVERBOUGHT and row['RSI'] >= RSI_OVERBOUGHT:
            for lvl in scale_levels_sell:
                if row['high'] >= lvl:
                    triggered_scales_sell.append(lvl)

        # --- OPEN/ADD TO BUY TRADE ---
        if triggered_scales and (open_trade is None or open_trade['side'] == "BUY"):
            if open_trade is None:
                open_trade = {
                    'side': "BUY",
                    'lots': [],
                    'entry_times': [],
                    'stops': [],
                    'targets': [],
                    'scale_levels': [],
                    'scale_count': 0
                }
            for lvl in triggered_scales:
                open_trade['lots'].append(lvl)
                open_trade['entry_times'].append(curr_time)
                stop, target = calculate_risk_levels(lvl, 'BUY')
                open_trade['stops'].append(stop)
                open_trade['targets'].append(target)
                open_trade['scale_levels'].append(lvl)
                open_trade['scale_count'] += 1

        # --- OPEN/ADD TO SELL TRADE ---
        if triggered_scales_sell and (open_trade is None or open_trade['side'] == "SELL"):
            if open_trade is None:
                open_trade = {
                    'side': "SELL",
                    'lots': [],
                    'entry_times': [],
                    'stops': [],
                    'targets': [],
                    'scale_levels': [],
                    'scale_count': 0
                }
            for lvl in triggered_scales_sell:
                open_trade['lots'].append(lvl)
                open_trade['entry_times'].append(curr_time)
                stop, target = calculate_risk_levels(lvl, 'SELL')
                open_trade['stops'].append(stop)
                open_trade['targets'].append(target)
                open_trade['scale_levels'].append(lvl)
                open_trade['scale_count'] += 1

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
                else:
                    # Stop loss
                    if row['high'] >= stop:
                        exit_price = stop
                        exit_time = curr_time
                        result = "Stopped"
                    # Target
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
                del open_trade['scale_levels'][idx]
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

# --- Actual run ---
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
    st.metric("SMA20", f"{df.iloc[-1]['SMA20']:.2f}")

run_backtest = st.button("Run Backtest")

if run_backtest:
    st.info("Running backtest, please wait...")
    trades, equity_curve, timestamp_curve = backtest_scaled_entries(df)
    df_trades = pd.DataFrame(trades)
    st.success(f"Backtest completed. {len(df_trades)} exits (partial lots) recorded.")

    # Summary stats
    total_pnl = df_trades['pnl'].sum() if not df_trades.empty else 0
    win_trades = df_trades[df_trades['pnl'] > 0]
    loss_trades = df_trades[df_trades['pnl'] <= 0]
    win_rate = len(win_trades) / len(df_trades) * 100 if len(df_trades) else 0
    avg_win = win_trades['pnl'].mean() if not win_trades.empty else 0
    avg_loss = loss_trades['pnl'].mean() if not loss_trades.empty else 0
    profit_factor = win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum()) if not loss_trades.empty else np.inf
    max_drawdown = min(0, min(equity_curve) - max(equity_curve[:equity_curve.index(min(equity_curve))]) if equity_curve else 0)

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

    # Show equity curve
    st.markdown("#### Equity Curve")
    eq_df = pd.DataFrame({'equity': equity_curve}, index=pd.to_datetime(timestamp_curve))
    st.line_chart(eq_df)

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
st.sidebar.caption("Version: 2.0.0")
st.sidebar.caption(f"Last Updated: {NOW_UTC}")
st.sidebar.caption("Developer: GillyMwazala")
