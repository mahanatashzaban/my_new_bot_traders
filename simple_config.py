#!/usr/bin/env python3
"""
Simple Configuration - No dotenv dependency
"""

# Exchange Configuration
EXCHANGE_CONFIG = {
    'exchange_id': 'binance',
    'api_key': '',  # Empty for public data
    'secret': '',   # Empty for public data
}

# Trading Parameters - 5 MINUTE TIMEFRAME WITH LEVERAGE
TRADING_CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframe': '5m',
    'initial_balance': 1000,
    'risk_per_trade': 0.02,
    'max_open_trades': 3,
    'leverage': 10,
    'tp_percent': 0.008,  # 0.8% take profit for scalping
    'sl_percent': 0.004,  # 0.4% stop loss for scalping
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

print("✅ Simple config loaded - 5m timeframe with leverage set")
