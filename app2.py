import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
API_KEY = "your_api_key_here"  # Replace with your actual API key
SYMBOL = "XAUUSD"  # Changed from XAU/USD to XAUUSD as per Twelve Data format
INTERVAL = "1h"
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
REWARD_TO_RISK = 2

def fetch_data(symbol, interval, api_key, limit=500):
    """
    Fetch data from Twelve Data API with improved error handling
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": limit,
        "apikey": api_key
    }
    
    try:
        logger.info(f"Fetching data for {symbol} at {interval} interval")
        r = requests.get(url, params=params)
        r.raise_for_status()  # Raise an exception for bad status codes
        
        data = r.json()
        logger.info(f"Response received: {data.keys() if isinstance(data, dict) else 'Invalid JSON'}")
        
        # Check for API error messages
        if isinstance(data, dict) and data.get('status') == 'error':
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
            
        if "values" not in data:
            raise Exception(f"Unexpected API response format: {data}")
            
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.reset_index(drop=True)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        raise Exception(f"Failed to connect to Twelve Data API: {e}")
    except ValueError as e:
        logger.error(f"Data processing error: {e}")
        raise Exception(f"Failed to process API data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def test_api_connection():
    """
    Test API connection and symbol validity
    """
    test_url = "https://api.twelvedata.com/symbol_search"
    params = {
        "symbol": SYMBOL,
        "apikey": API_KEY
    }
    
    try:
        r = requests.get(test_url, params=params)
        r.raise_for_status()
        data = r.json()
        logger.info(f"API Connection test response: {data}")
        return data
    except Exception as e:
        logger.error(f"API Connection test failed: {e}")
        return None

# ... [rest of the code remains the same] ...

def main():
    try:
        logger.info("Testing API connection...")
        test_result = test_api_connection()
        if test_result:
            logger.info("API connection successful")
        
        logger.info("Fetching market data...")
        df = fetch_data(SYMBOL, INTERVAL, API_KEY)
        
        logger.info("Generating signals...")
        df = signal_generator(df)
        
        logger.info("Running backtest...")
        trades = backtest(df)
        
        print("\nTrade Log:")
        for t in trades:
            print(t)
            
        total_pnl = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t.get("result") == "tp")
        losses = sum(1 for t in trades if t.get("result") == "stop")
        
        print(f"\nResults:")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Total Trades: {len(trades)}")
        
        if len(trades) > 0:
            win_rate = (wins / len(trades)) * 100
            print(f"Win Rate: {win_rate:.2f}%")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
