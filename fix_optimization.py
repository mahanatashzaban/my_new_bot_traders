#!/usr/bin/env python3
"""
Fix and Debug Strategy Optimization
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from indicators import IndicatorEngine

def debug_strategy():
    print("🔍 DEBUGGING STRATEGY PERFORMANCE")
    print("=" * 50)
    
    # Load data
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2000)
    if data is None:
        print("❌ Failed to fetch data")
        return
    
    print("Calculating indicators...")
    engine = IndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    print(f"\n📊 DATA ANALYSIS:")
    print(f"Total candles: {len(data)}")
    print(f"Trendline touches: {data['trendline_touch'].sum()}")
    print(f"Average rejection strength: {data['rejection_strength'].mean():.3f}")
    print(f"Max rejection strength: {data['rejection_strength'].max():.3f}")
    
    # Analyze what happens after trendline touches
    print(f"\n🔍 ANALYZING TRENDLINE TOUCHES:")
    
    for i in range(50, len(data) - 10):
        if data['trendline_touch'].iloc[i] == 1:
            current_price = data['close'].iloc[i]
            future_5min = data['close'].iloc[i+5]  # 5 minutes later
            future_10min = data['close'].iloc[i+10]  # 10 minutes later
            
            price_move_5min = (future_5min - current_price) / current_price
            price_move_10min = (future_10min - current_price) / current_price
            
            direction = "SHORT" if data['trendline_slope'].iloc[i] < 0 else "LONG"
            rejection = data['rejection_strength'].iloc[i]
            
            print(f"  {data.index[i]} - {direction} - Rejection: {rejection:.2f}")
            print(f"    5min move: {price_move_5min*100:+.2f}%")
            print(f"    10min move: {price_move_10min*100:+.2f}%")
            
            # Only show first 10 for analysis
            if i > 60:
                break
    
    # Test simple strategy: Enter on strong rejection, exit after 5 minutes
    print(f"\n🧪 TESTING SIMPLE STRATEGY:")
    
    balance = 1000
    trades = []
    
    for i in range(50, len(data) - 5):
        if (data['trendline_touch'].iloc[i] == 1 and 
            data['rejection_strength'].iloc[i] > 1.0):
            
            entry_price = data['close'].iloc[i]
            exit_price = data['close'].iloc[i+5]  # Exit after 5 minutes
            
            if data['trendline_slope'].iloc[i] < 0:  # SHORT
                pnl_percent = (entry_price - exit_price) / entry_price
            else:  # LONG
                pnl_percent = (exit_price - entry_price) / entry_price
            
            pnl_amount = balance * 0.02 * pnl_percent  # 2% position size
            balance += pnl_amount
            
            trades.append({
                'time': data.index[i],
                'type': 'SHORT' if data['trendline_slope'].iloc[i] < 0 else 'LONG',
                'rejection': data['rejection_strength'].iloc[i],
                'pnl_percent': pnl_percent,
                'balance': balance
            })
    
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        print(f"Trades executed: {len(trades_df)}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total return: {total_return:+.2f}%")
        print(f"Final balance: ${balance:.2f}")
        
        # Show first few trades
        print(f"\nFirst 5 trades:")
        for i, trade in trades_df.head().iterrows():
            print(f"  {trade['time']} - {trade['type']} - Rej: {trade['rejection']:.2f} - PnL: {trade['pnl_percent']*100:+.2f}%")
    else:
        print("No trades executed - strategy too restrictive")

def find_best_parameters():
    """Find parameters that actually work"""
    print(f"\n🎯 FINDING WORKING PARAMETERS")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2000)
    engine = IndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    # Test different rejection thresholds
    print("Testing rejection thresholds:")
    
    for rejection_thresh in [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
        trades = 0
        successful = 0
        
        for i in range(50, len(data) - 5):
            if (data['trendline_touch'].iloc[i] == 1 and 
                data['rejection_strength'].iloc[i] > rejection_thresh):
                
                trades += 1
                entry_price = data['close'].iloc[i]
                exit_price = data['close'].iloc[i+5]
                
                if data['trendline_slope'].iloc[i] < 0:  # SHORT
                    move = (entry_price - exit_price) / entry_price
                else:  # LONG
                    move = (exit_price - entry_price) / entry_price
                
                if move > 0.001:  # 0.1% profit threshold
                    successful += 1
        
        if trades > 0:
            win_rate = successful / trades
            print(f"  Rejection > {rejection_thresh}: {trades} trades, {win_rate:.3f} win rate")

def main():
    debug_strategy()
    find_best_parameters()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"1. Check if trendline detection is working correctly")
    print(f"2. Adjust rejection strength thresholds")
    print(f"3. Consider shorter timeframes for exits (1-3 minutes)")
    print(f"4. Add additional filters (volume, RSI, etc.)")

if __name__ == "__main__":
    main()
