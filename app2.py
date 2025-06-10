import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
API_KEY = "YOUR_TWELVE_DATA_API_KEY"
SYMBOL = "XAU/USD"
INTERVAL = "1h"  # or "4h"
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
REWARD_TO_RISK = 2  # 2:1 reward:risk

def fetch_data(symbol, interval, api_key, limit=500):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": limit,
        "apikey": api_key,
        "format": "JSON"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise Exception("Failed to fetch data: " + str(data))
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def signal_generator(df):
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_slow"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["signal"] = ""

    for i in range(1, len(df)):
        # --- Detect EMA crossovers ---
        prev_fast = df.loc[i-1, "ema_fast"]
        prev_slow = df.loc[i-1, "ema_slow"]
        curr_fast = df.loc[i,   "ema_fast"]
        curr_slow = df.loc[i,   "ema_slow"]
        curr_rsi = df.loc[i,    "rsi"]

        # Golden Cross (buy)
        if (prev_fast < prev_slow) and (curr_fast > curr_slow):
            if 50 < curr_rsi < 70:
                df.loc[i, "signal"] = "buy"
        # Death Cross (sell)
        elif (prev_fast > prev_slow) and (curr_fast < curr_slow):
            if 30 < curr_rsi < 50:
                df.loc[i, "signal"] = "sell"
    return df

def find_swing(df, i, direction="low"):
    window = 10  # lookback window for swing highs/lows
    if direction == "low":
        return df["low"].iloc[max(0, i-window):i+1].min()
    elif direction == "high":
        return df["high"].iloc[max(0, i-window):i+1].max()

def backtest(df):
    trades = []
    position = None
    entry_price = sl = tp = None
    for i in range(1, len(df)):
        row = df.iloc[i]
        if position is None:
            if row.signal == "buy":
                swing_low = find_swing(df, i, "low")
                sl = swing_low
                entry_price = row.close
                tp = entry_price + (entry_price - sl) * REWARD_TO_RISK
                position = "long"
                entry_idx = i
                trades.append({"type": "buy", "entry": entry_price, "sl": sl, "tp": tp, "entry_idx": i})
            elif row.signal == "sell":
                swing_high = find_swing(df, i, "high")
                sl = swing_high
                entry_price = row.close
                tp = entry_price - (sl - entry_price) * REWARD_TO_RISK
                position = "short"
                entry_idx = i
                trades.append({"type": "sell", "entry": entry_price, "sl": sl, "tp": tp, "entry_idx": i})
        else:
            # manage open position
            open_trade = trades[-1]
            if open_trade["type"] == "buy":
                # Stop-loss or take-profit hit
                if row.low <= open_trade["sl"]:
                    open_trade["exit"] = open_trade["sl"]
                    open_trade["exit_idx"] = i
                    open_trade["result"] = "stop"
                    position = None
                elif row.high >= open_trade["tp"]:
                    open_trade["exit"] = open_trade["tp"]
                    open_trade["exit_idx"] = i
                    open_trade["result"] = "tp"
                    position = None
                # Optional exit: RSI > 70
                elif row.rsi >= 70:
                    open_trade["exit"] = row.close
                    open_trade["exit_idx"] = i
                    open_trade["result"] = "rsi_exit"
                    position = None
            elif open_trade["type"] == "sell":
                if row.high >= open_trade["sl"]:
                    open_trade["exit"] = open_trade["sl"]
                    open_trade["exit_idx"] = i
                    open_trade["result"] = "stop"
                    position = None
                elif row.low <= open_trade["tp"]:
                    open_trade["exit"] = open_trade["tp"]
                    open_trade["exit_idx"] = i
                    open_trade["result"] = "tp"
                    position = None
                # Optional exit: RSI < 30
                elif row.rsi <= 30:
                    open_trade["exit"] = row.close
                    open_trade["exit_idx"] = i
                    open_trade["result"] = "rsi_exit"
                    position = None
    # Calculate P/L
    for trade in trades:
        if "exit" in trade:
            if trade["type"] == "buy":
                trade["pnl"] = trade["exit"] - trade["entry"]
            else:
                trade["pnl"] = trade["entry"] - trade["exit"]
        else:
            trade["pnl"] = 0
    return trades

def main():
    df = fetch_data(SYMBOL, INTERVAL, API_KEY)
    df = signal_generator(df)
    trades = backtest(df)
    print("Trade Log:")
    for t in trades:
        print(t)
    total_pnl = sum(t["pnl"] for t in trades)
    wins = sum(1 for t in trades if t.get("result") == "tp")
    losses = sum(1 for t in trades if t.get("result") == "stop")
    print(f"Total PnL: {total_pnl:.2f}, Wins: {wins}, Losses: {losses}, Total Trades: {len(trades)}")

if __name__ == "__main__":
    main()
