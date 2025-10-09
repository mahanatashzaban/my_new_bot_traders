#!/usr/bin/env python3
"""
Quick test for bidirectional trading
"""

from data_manager_simple import DataManager
from simple_indicators import SimpleMLIndicatorEngine

def quick_test():
    print("🚀 QUICK BIDIRECTIONAL TEST")
    print("=" * 50)
    
    dm = DataManager()
    indicator_engine = SimpleMLIndicatorEngine()
    
    # Get current data
    data = dm.fetch_historical_data(limit=300)
    
    if data is not None:
        print(f"✅ Data loaded: {len(data)} candles")
        print(f"Current price: ${data['close'].iloc[-1]:.2f}")
        
        # Calculate indicators
        data_with_indicators = indicator_engine.calculate_all_indicators(data.copy())
        print(f"✅ Indicators calculated: {len(indicator_engine.get_feature_columns())} features")
        
        # Show recent price action
        recent = data.tail(10)
        print("\n📈 Recent price action:")
        for i, (idx, row) in enumerate(recent.iterrows()):
            change = (row['close'] - row['open']) / row['open'] * 100
            print(f"  {idx.strftime('%H:%M')}: ${row['close']:.2f} ({change:+.2f}%)")
    
    return data is not None

if __name__ == "__main__":
    quick_test()
