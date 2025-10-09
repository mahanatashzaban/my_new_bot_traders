#!/usr/bin/env python3
"""
Data Manager for Crypto Trading Bot
Handles data fetching from exchanges without requiring API keys for historical data
"""

import ccxt
import pandas as pd
import time
from config import EXCHANGE_CONFIG, TRADING_CONFIG

class DataManager:
    def __init__(self):
        self.exchange = self._initialize_exchange()
        self.symbol = TRADING_CONFIG['symbol']
        self.timeframe = TRADING_CONFIG['timeframe']
    
    def _initialize_exchange(self):
        """Initialize exchange connection for public data"""
        exchange_id = EXCHANGE_CONFIG['exchange_id']
        
        try:
            exchange_class = getattr(ccxt, exchange_id)
        except AttributeError:
            print(f"❌ Exchange {exchange_id} not found. Using binance instead.")
            exchange_class = getattr(ccxt, 'binance')
        
        # For public data fetching, minimal config needed
        exchange_config = {
            'enableRateLimit': True,  # Important to avoid rate limits
            'timeout': 30000,  # 30 second timeout
        }
        
        # Only add credentials if provided (for future live trading)
        api_key = EXCHANGE_CONFIG.get('api_key', '')
        secret = EXCHANGE_CONFIG.get('secret', '')
        
        if api_key and secret:
            exchange_config.update({
                'apiKey': api_key,
                'secret': secret,
            })
            print(f"✅ Initialized {exchange_id} with API keys")
        else:
            print(f"✅ Initialized {exchange_id} for public data access")
        
        exchange = exchange_class(exchange_config)
        
        # Test connection
        try:
            exchange.load_markets()
            print(f"✅ Successfully connected to {exchange_id}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load markets: {e}")
            print("Trying to continue with basic data access...")
        
        return exchange
    
    def fetch_historical_data(self, limit=1000):
        """Fetch historical OHLCV data - workaround for Binance limit"""
        try:
            print(f"📊 Fetching {limit} candles of {self.timeframe} data for {self.symbol}...")
            
            # For Binance, we need to fetch multiple times to get more data
            all_ohlcv = []
            remaining = limit
            
            while remaining > 0:
                chunk_size = min(1000, remaining)  # Binance max is 1000 per call
                
                # If we already have data, get older data
                if all_ohlcv:
                    since = all_ohlcv[0][0] - (chunk_size * 60 * 1000)  # Go back in time
                    ohlcv = self.exchange.fetch_ohlcv(
                        self.symbol, 
                        self.timeframe, 
                        since=since,
                        limit=chunk_size
                    )
                else:
                    # First call - get most recent data
                    ohlcv = self.exchange.fetch_ohlcv(
                        self.symbol, 
                        self.timeframe, 
                        limit=chunk_size
                    )
                
                if not ohlcv:
                    break
                    
                # Add to beginning to maintain chronological order
                all_ohlcv = ohlcv + all_ohlcv
                remaining -= len(ohlcv)
                
                print(f"   Fetched {len(ohlcv)} candles, {remaining} remaining...")
                
                if remaining > 0:
                    time.sleep(0.2)  # Rate limiting
            
            if not all_ohlcv:
                print("❌ No data returned from exchange")
                return None
            
            # Trim to exact limit if we got more
            all_ohlcv = all_ohlcv[:limit]
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"✅ Successfully fetched {len(df)} candles")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return df
            
        except ccxt.NetworkError as e:
            print(f"❌ Network error fetching data: {e}")
            return None
        except ccxt.ExchangeError as e:
            print(f"❌ Exchange error fetching data: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error fetching data: {e}")
            return None
    
    def fetch_realtime_data(self):
        """Fetch latest candle for real-time trading"""
        return self.fetch_historical_data(limit=1)
    
    def get_current_price(self):
        """Get current BTC price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            print(f"❌ Error fetching current price: {e}")
            return None
    
    def get_exchange_info(self):
        """Get exchange information"""
        try:
            markets = self.exchange.load_markets()
            symbol_info = markets.get(self.symbol)
            
            if symbol_info:
                print(f"\n📋 Exchange Info for {self.symbol}:")
                print(f"   Base: {symbol_info['base']}")
                print(f"   Quote: {symbol_info['quote']}")
                print(f"   Active: {symbol_info['active']}")
                if 'limits' in symbol_info:
                    print(f"   Amount Min: {symbol_info['limits']['amount'].get('min', 'N/A')}")
                    print(f"   Price Min: {symbol_info['limits']['price'].get('min', 'N/A')}")
            else:
                print(f"❌ No info found for {self.symbol}")
                
        except Exception as e:
            print(f"❌ Error getting exchange info: {e}")
    
    def test_connection(self):
        """Test exchange connection"""
        print("🔧 Testing exchange connection...")
        
        # Test 1: Fetch ticker
        price = self.get_current_price()
        if price:
            print(f"✅ Current price: ${price:.2f}")
        else:
            print("❌ Failed to get current price")
            return False
        
        # Test 2: Fetch small amount of historical data
        test_data = self.fetch_historical_data(limit=10)
        if test_data is not None and len(test_data) > 0:
            print(f"✅ Historical data fetch successful")
            return True
        else:
            print("❌ Historical data fetch failed")
            return False

# Test function
def test_data_manager():
    """Test the data manager"""
    print("🧪 Testing Data Manager...")
    print("=" * 50)
    
    dm = DataManager()
    
    # Test connection
    if dm.test_connection():
        print("\n✅ Data Manager is working correctly!")
        
        # Get exchange info
        dm.get_exchange_info()
        
        # Fetch sample data
        print(f"\n📊 Fetching sample data...")
        data = dm.fetch_historical_data(limit=100)
        if data is not None:
            print(f"Sample data:")
            print(data.tail())
            return True
    else:
        print("\n❌ Data Manager has issues!")
        return False

if __name__ == "__main__":
    test_data_manager()
