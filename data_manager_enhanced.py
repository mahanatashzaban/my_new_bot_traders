#!/usr/bin/env python3
"""
Enhanced Data Manager with Better Data Fetching
"""

import ccxt
import pandas as pd
import time
from datetime import datetime

class EnhancedDataManager:
    def __init__(self):
        self.exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        self.symbol = 'BTC/USDT'
        self.timeframe = '5m'
        
    def fetch_historical_data(self, limit=500):
        """Fetch historical data with better error handling"""
        try:
            print(f"📊 Fetching {limit} {self.timeframe} candles for {self.symbol}...")
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, 
                self.timeframe, 
                limit=limit
            )
            
            if not ohlcv:
                print("❌ No data returned from exchange")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Successfully fetched {len(df)} {self.timeframe} candles")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def get_current_price(self):
        """Get current BTC price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            print(f"❌ Error getting current price: {e}")
            return None

# Update the imports in your bots to use this enhanced data manager
