#!/usr/bin/env python3
"""
Simple Backtest for Trendline Strategy - CLEAN VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_manager import DataManager
from indicators import IndicatorEngine
import joblib

class SimpleBacktester:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades = []
        
    def run_simple_backtest(self, df):
        """Simple backtest without ML - just test the basic strategy"""
        print("🔍 Running Simple Backtest...")
        print("=" * 50)
        
        balance = self.initial_balance
        position = None
        trades = []
        
        for i in range(50, len(df)):
            current_data = df.iloc[i]
            current_price = current_data['close']
            current_time = df.index[i]
            
            # EXIT LOGIC
            if position:
                pnl_percent = self._calculate_pnl(position, current_price)
                
                # Exit conditions
                exit_trade = False
                exit_reason = ""
                
                # Stop loss (2%)
                if position['type'] == 'LONG' and pnl_percent <= -0.02:
                    exit_trade = True
                    exit_reason = "STOP LOSS"
                elif position['type'] == 'SHORT' and pnl_percent <= -0.02:
                    exit_trade = True
                    exit_reason = "STOP LOSS"
                
                # Take profit (3%)
                elif position['type'] == 'LONG' and pnl_percent >= 0.03:
                    exit_trade = True
                    exit_reason = "TAKE PROFIT"
                elif position['type'] == 'SHORT' and pnl_percent >= 0.03:
                    exit_trade = True
                    exit_reason = "TAKE PROFIT"
                
                # Time exit (30 minutes)
                elif (current_time - position['entry_time']).total_seconds() > 1800:
                    exit_trade = True
                    exit_reason = "TIME EXIT"
                
                if exit_trade:
                    pnl_amount = balance * pnl_percent
                    balance += pnl_amount
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_percent': pnl_percent,
                        'pnl_amount': pnl_amount,
                        'balance': balance,
                        'reason': exit_reason
                    })
                    
                    print(f"🔴 EXIT {position['type']}: {exit_reason} | PnL: {pnl_percent*100:+.2f}%")
                    position = None
            
            # ENTRY LOGIC
            if not position:
                # Simple trendline strategy rules
                is_touch = current_data['trendline_touch'] == 1
                has_rejection = current_data['rejection_strength'] > 0.8
                
                if is_touch and has_rejection:
                    # Determine direction
                    if current_data['trendline_slope'] < 0:  # Resistance - SHORT
                        signal = "SHORT"
                        print(f"🎯 SHORT Signal at ${current_price:.2f}")
                    else:  # Support - LONG
                        signal = "LONG" 
                        print(f"🎯 LONG Signal at ${current_price:.2f}")
                    
                    # Enter trade with 2% risk
                    position_size = (balance * 0.02) / current_price
                    position = {
                        'type': signal,
                        'entry_price': current_price,
                        'size': position_size,
                        'entry_time': current_time
                    }
        
        # Store results
        self.balance = balance
        self.trades = trades
        self.position = position
        
        return self._analyze_results()
    
    def _calculate_pnl(self, position, current_price):
        """Calculate P&L percentage"""
        if position['type'] == 'LONG':
            return (current_price - position['entry_price']) / position['entry_price']
        else:
            return (position['entry_price'] - current_price) / position['entry_price']
    
    def _analyze_results(self):
        """Analyze backtest results"""
        if not self.trades:
            print("❌ No trades executed")
            return False
        
        trades_df = pd.DataFrame(self.trades)
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl_percent'].mean() * 100
        avg_loss = losing_trades['pnl_percent'].mean() * 100
        
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS")
        print("="*60)
        print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        print(f"💰 Final Balance: ${self.balance:.2f}")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"🎯 Total Trades: {len(trades_df)}")
        print(f"✅ Winning Trades: {len(winning_trades)}")
        print(f"❌ Losing Trades: {len(losing_trades)}")
        print(f"📊 Win Rate: {win_rate:.1f}%")
        print(f"🔥 Average Win: {avg_win:+.2f}%")
        print(f"💧 Average Loss: {avg_loss:+.2f}%")
        
        # Check if strategy is profitable
        is_profitable = total_return > 0
        has_good_win_rate = win_rate > 45
        
        print(f"\n🔍 STRATEGY STATUS: {'✅ PROFITABLE' if is_profitable else '❌ NOT PROFITABLE'}")
        
        return is_profitable

def main():
    print("🎯 SIMPLE TRENDLINE STRATEGY BACKTEST")
    print("=" * 50)
    
    # Load data
    dm = DataManager()
    print("Fetching data...")
    data = dm.fetch_historical_data(limit=1000)  # Smaller dataset for quick testing
    
    if data is None:
        print("❌ Failed to fetch data")
        return
    
    print(f"📊 Loaded {len(data)} candles")
    
    # Calculate indicators ONCE
    print("Calculating indicators...")
    engine = IndicatorEngine()
    data_with_indicators = engine.calculate_all_indicators(data)
    
    # Run backtest
    backtester = SimpleBacktester(initial_balance=1000)
    success = backtester.run_simple_backtest(data_with_indicators)
    
    if success:
        print("\n🎉 Strategy shows promise! Consider optimizing parameters.")
    else:
        print("\n⚠️  Strategy needs improvement. Try different parameters.")

if __name__ == "__main__":
    main()
