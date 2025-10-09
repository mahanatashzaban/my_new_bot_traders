#!/usr/bin/env python3
"""
Test Improved Strategy with Better Parameters
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from improved_indicators import ImprovedIndicatorEngine

def test_improved_strategy():
    print("🎯 TESTING IMPROVED STRATEGY")
    print("=" * 50)
    
    # Load data
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2000)
    
    if data is None:
        print("❌ Failed to fetch data")
        return
    
    # Use improved indicators
    engine = ImprovedIndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    # Test strategy with improved parameters
    balance = 1000
    trades = []
    
    print("\n🔍 Running improved backtest...")
    
    for i in range(100, len(data) - 3):  # Shorter hold time (3 minutes)
        # STRICTER ENTRY CONDITIONS
        is_quality_touch = data['trendline_touch'].iloc[i] == 1
        has_strong_rejection = data['rejection_strength'].iloc[i] > 1.5
        good_volume = data['volume_ma_ratio'].iloc[i] > 1.2
        not_overbought = data['rsi_14'].iloc[i] < 70  # Additional filter
        
        if is_quality_touch and has_strong_rejection and good_volume and not_overbought:
            entry_price = data['close'].iloc[i]
            exit_price = data['close'].iloc[i+3]  # Exit after 3 minutes
            
            if data['trendline_slope'].iloc[i] < 0:  # SHORT
                pnl_percent = (entry_price - exit_price) / entry_price
                signal_type = "SHORT"
            else:  # LONG
                pnl_percent = (exit_price - entry_price) / entry_price
                signal_type = "LONG"
            
            pnl_amount = balance * 0.02 * pnl_percent  # 2% position size
            balance += pnl_amount
            
            trades.append({
                'time': data.index[i],
                'type': signal_type,
                'rejection': data['rejection_strength'].iloc[i],
                'pnl_percent': pnl_percent,
                'balance': balance
            })
            
            print(f"🎯 {signal_type} at ${entry_price:.2f} - Rej: {data['rejection_strength'].iloc[i]:.2f} - PnL: {pnl_percent*100:+.2f}%")
    
    # Analyze results
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        print(f"\n📊 IMPROVED STRATEGY RESULTS:")
        print(f"Trades executed: {len(trades_df)}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total return: {total_return:+.2f}%")
        print(f"Final balance: ${balance:.2f}")
        
        if len(trades_df) > 0:
            avg_trade_return = trades_df['pnl_percent'].mean() * 100
            best_trade = trades_df['pnl_percent'].max() * 100
            worst_trade = trades_df['pnl_percent'].min() * 100
            
            print(f"Average trade: {avg_trade_return:+.2f}%")
            print(f"Best trade: {best_trade:+.2f}%")
            print(f"Worst trade: {worst_trade:+.2f}%")
        
        # Show all trades
        print(f"\n📋 ALL TRADES:")
        for i, trade in trades_df.iterrows():
            print(f"  {trade['time']} - {trade['type']} - Rej: {trade['rejection']:.2f} - PnL: {trade['pnl_percent']*100:+.2f}%")
    
    else:
        print("No quality trades found - strategy is very selective (this is good!)")

if __name__ == "__main__":
    test_improved_strategy()
