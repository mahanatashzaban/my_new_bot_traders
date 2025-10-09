#!/usr/bin/env python3
"""
Data Manager with Simple Config
"""

import ccxt
import pandas as pd
import time
from simple_config import EXCHANGE_CONFIG, TRADING_CONFIG  # Use simple config

class DataManager:
    def __init__(self):
        self.exchange = self._initialize_exchange()
        self.symbol = TRADING_CONFIG['symbol']
        self.timeframe = TRADING_CONFIG['timeframe']  # Now 5m

    def _initialize_exchange(self):
        exchange_id = EXCHANGE_CONFIG['exchange_id']
        
        try:
            exchange_class = getattr(ccxt, exchange_id)
        except AttributeError:
            print(f"❌ Exchange {exchange_id} not found. Using binance.")
            exchange_class = getattr(ccxt, 'binance')

        exchange_config = {
            'enableRateLimit': True,
            'timeout': 30000,
        }

        # Only add credentials if provided
        api_key = EXCHANGE_CONFIG.get('api_key', '')
        secret = EXCHANGE_CONFIG.get('secret', '')

        if api_key and secret:
            exchange_config.update({
                'apiKey': api_key,
                'secret': secret,
            })

        exchange = exchange_class(exchange_config)

        # Test connection
        try:
            exchange.load_markets()
            print(f"✅ Connected to {exchange_id} for {self.timeframe} data")
        except Exception as e:
            print(f"⚠️  Warning: {e}")

        return exchange

    def fetch_historical_data(self, limit=1000):
        try:
            print(f"📊 Fetching {limit} {self.timeframe} candles for {self.symbol}...")
            
            all_ohlcv = []
            remaining = limit

            while remaining > 0:
                chunk_size = min(1000, remaining)

                if all_ohlcv:
                    since = all_ohlcv[0][0] - (chunk_size * 5 * 60 * 1000)  # 5 minutes in ms
                    ohlcv = self.exchange.fetch_ohlcv(
                        self.symbol,
                        self.timeframe,
                        since=since,
                        limit=chunk_size
                    )
                else:
                    ohlcv = self.exchange.fetch_ohlcv(
                        self.symbol,
                        self.timeframe,
                        limit=chunk_size
                    )

                if not ohlcv:
                    break

                all_ohlcv = ohlcv + all_ohlcv
                remaining -= len(ohlcv)

                print(f"   Fetched {len(ohlcv)} candles, {remaining} remaining...")

                if remaining > 0:
                    time.sleep(0.2)

            all_ohlcv = all_ohlcv[:limit]

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"✅ Successfully fetched {len(df)} {self.timeframe} candles")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

            return df

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None

    def get_current_price(self):
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            print(f"❌ Error fetching price: {e}")
            return None

# Test the data manager
if __name__ == "__main__":
    print("🧪 Testing 5-minute Data Manager...")
    dm = DataManager()
    data = dm.fetch_historical_data(limit=10)
    if data is not None:
        print("✅ 5-minute data working!")
        print(data.tail())
