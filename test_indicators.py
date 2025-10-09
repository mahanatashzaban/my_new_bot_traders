#!/usr/bin/env python3
"""
Test which ta library indicators work
"""

import ta
from ta import trend, momentum, volatility, volume
import pandas as pd
import numpy as np

# Create test data
test_data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109], 
    'low': [95, 96, 97, 98, 99],
    'close': [102, 103, 104, 105, 106],
    'volume': [1000, 2000, 3000, 4000, 5000]
})

print("Testing ta library availability...")
print("ta module contents:", [x for x in dir(ta) if not x.startswith('_')])
print("\ntrend module contents:", [x for x in dir(trend) if not x.startswith('_')][:10])
print("\nTesting individual indicators...")

# Test basic indicators
indicators_to_test = [
    ('SMA', lambda: trend.SMAIndicator(test_data['close'], window=14)),
    ('EMA', lambda: trend.EMAIndicator(test_data['close'], window=14)),
    ('WMA', lambda: trend.WMAIndicator(test_data['close'], window=14)),
    ('RSI', lambda: momentum.RSIIndicator(test_data['close'], window=14)),
    ('MACD', lambda: trend.MACD(test_data['close'])),
]

for name, indicator_func in indicators_to_test:
    try:
        indicator = indicator_func()
        methods = [x for x in dir(indicator) if not x.startswith('_') and not x.startswith('df')]
        result_methods = [m for m in methods if 'indicator' in m or 'macd' in m or 'rsi' in m]
        print(f"✅ {name}: Available methods: {result_methods}")
        
        # Try to get actual values
        for method in result_methods:
            try:
                result = getattr(indicator, method)()
                print(f"   {method}(): {result}")
            except:
                print(f"   {method}(): Failed")
                
    except Exception as e:
        print(f"❌ {name}: Failed to create - {e}")

print("\nTesting direct functions...")
# Test direct functions
direct_functions = [
    ('sma', lambda: trend.sma_indicator(test_data['close'], window=14)),
    ('ema', lambda: trend.ema_indicator(test_data['close'], window=14)),
    ('wma', lambda: trend.wma_indicator(test_data['close'], window=14)),
    ('rsi', lambda: momentum.rsi(test_data['close'], window=14)),
]

for name, func in direct_functions:
    try:
        result = func()
        print(f"✅ {name}: Works - {result}")
    except Exception as e:
        print(f"❌ {name}: Failed - {e}")

print("\n✅ Indicator test completed")
