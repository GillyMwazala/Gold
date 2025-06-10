import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
API_KEY = "your_api_key_here"  # Replace with your actual API key
SYMBOL = "XAUUSD"
INTERVAL = "1h"
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
REWARD_TO_RISK = 2
USER = "GillyMwazala"

def fetch_data(symbol: str, interval: str, api_key: str, limit: int = 500) -> pd.DataFrame:
    """
    Fetch historical data from Twelve Data API
    """
    base_url = "https://api.twelvedata.com/time_series"
    
    params: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": str(limit),  # Convert to string as expected by the API
        "format": "JSON",
        "timezone": "UTC"
    }
    
    try:
        logger.info(f"Fetching data for {symbol} with {interval} interval")
        response = requests.get(base_url, params=params, timeout=30)
        
        # Log the actual URL being called (without API key)
        debug_params = params.copy()
        debug_params['apikey'] = 'HIDDEN'
        logger.info(f"API URL (without key): {base_url}?{requests.compat.urlencode(debug_params)}")
        
        if response.status_code != 200:
            logger.error(f"API returned status code {response.status_code}")
            logger.error(f"Response content: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}")
            
        data = response.json()
        
        if 'status' in data and data['status'] == 'error':
            logger.error(f"API returned error: {data.get('message', 'Unknown error')}")
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
            
        if 'values' not in data:
            logger.error(f"Unexpected API response format: {data}")
            raise Exception("API response missing 'values' key")
            
        # Create DataFrame
        df = pd.DataFrame(data['values'])
        
        # Convert types
        df['datetime'] = pd.to_datetime(df['datetime'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        logger.info(f"Successfully fetched {len(df)} rows of data")
        return df
        
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        raise Exception("API request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise Exception(f"API request failed: {str(e)}")
    except ValueError as e:
        logger.error(f"Failed to parse API response: {str(e)}")
        raise Exception(f"Failed to parse API response: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    # Calculate EMAs
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals"""
    df['signal'] = ''
    
    for i in range(1, len(df)):
        # Golden Cross (buy)
        if (df['ema_fast'].iloc[i-1] < df['ema_slow'].iloc[i-1] and 
            df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and
            50 < df['rsi'].iloc[i] < 70):
            df.loc[i, 'signal'] = 'buy'
            
        # Death Cross (sell)
        elif (df['ema_fast'].iloc[i-1] > df['ema_slow'].iloc[i-1] and 
              df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and
              30 < df['rsi'].iloc[i] < 50):
            df.loc[i, 'signal'] = 'sell'
            
    return df

def backtest(df: pd.DataFrame) -> list:
    """Run backtest on the signals"""
    trades = []
    position = None
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # New position entry
        if position is None and row['signal'] in ['buy', 'sell']:
            entry_price = row['close']
            position = row['signal']
            sl = row['low'] - (row['high'] - row['low']) if position == 'buy' else row['high'] + (row['high'] - row['low'])
            tp = entry_price + (entry_price - sl) * REWARD_TO_RISK if position == 'buy' else entry_price - (sl - entry_price) * REWARD_TO_RISK
            
            trades.append({
                'entry_date': row['datetime'],
                'type': position,
                'entry': entry_price,
                'sl': sl,
                'tp': tp
            })
        
        # Position management
        elif position is not None:
            current_trade = trades[-1]
            
            # Check for exit conditions
            if position == 'buy':
                if row['low'] <= current_trade['sl']:
                    current_trade.update({
                        'exit_date': row['datetime'],
                        'exit': current_trade['sl'],
                        'result': 'stop',
                        'pnl': current_trade['sl'] - current_trade['entry']
                    })
                    position = None
                elif row['high'] >= current_trade['tp']:
                    current_trade.update({
                        'exit_date': row['datetime'],
                        'exit': current_trade['tp'],
                        'result': 'target',
                        'pnl': current_trade['tp'] - current_trade['entry']
                    })
                    position = None
                elif row['rsi'] >= 70:
                    current_trade.update({
                        'exit_date': row['datetime'],
                        'exit': row['close'],
                        'result': 'rsi_exit',
                        'pnl': row['close'] - current_trade['entry']
                    })
                    position = None
            else:  # position == 'sell'
                if row['high'] >= current_trade['sl']:
                    current_trade.update({
                        'exit_date': row['datetime'],
                        'exit': current_trade['sl'],
                        'result': 'stop',
                        'pnl': current_trade['entry'] - current_trade['sl']
                    })
                    position = None
                elif row['low'] <= current_trade['tp']:
                    current_trade.update({
                        'exit_date': row['datetime'],
                        'exit': current_trade['tp'],
                        'result': 'target',
                        'pnl': current_trade['entry'] - current_trade['tp']
                    })
                    position = None
                elif row['rsi'] <= 30:
                    current_trade.update({
                        'exit_date': row['datetime'],
                        'exit': row['close'],
                        'result': 'rsi_exit',
                        'pnl': current_trade['entry'] - row['close']
                    })
                    position = None
                    
    return trades

def main():
    try:
        logger.info(f"Starting strategy backtest for user: {USER}")
        logger.info(f"Fetching data for {SYMBOL} at {INTERVAL} interval")
        
        # Fetch data
        df = fetch_data(SYMBOL, INTERVAL, API_KEY)
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Generate signals
        df = generate_signals(df)
        
        # Run backtest
        trades = backtest(df)
        
        # Calculate statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        
        # Print results
        print("\nBacktest Results:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {(winning_trades/total_trades*100):.2f}% if total_trades > 0 else 0}%")
        print(f"Total PnL: ${total_pnl:.2f}")
        
        # Print detailed trade log
        print("\nDetailed Trade Log:")
        for trade in trades:
            print(f"\nEntry Date: {trade['entry_date']}")
            print(f"Type: {trade['type']}")
            print(f"Entry: ${trade['entry']:.2f}")
            print(f"Exit: ${trade['exit']:.2f}")
            print(f"PnL: ${trade['pnl']:.2f}")
            print(f"Result: {trade['result']}")
            
    except Exception as e:
        logger.error(f"Strategy execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
