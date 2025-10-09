import os
from dotenv import load_dotenv

load_dotenv()

# Exchange Configuration
EXCHANGE_CONFIG = {
    'exchange_id': 'binance',  # More reliable for data
    'api_key': '',  # Empty for public data
    'secret': '',   # Empty for public data
}

# Trading Parameters
TRADING_CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframe': '5m',
    'initial_balance': 1000,  # USDT
    'risk_per_trade': 0.02,  # 2% risk per trade
    'max_open_trades': 3,
}

# Feature Engineering
FEATURE_CONFIG = {
    'indicators': {
        'rsi_period': 14,
        'bb_period': 20,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'atr_period': 14,
    }
}
