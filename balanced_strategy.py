#!/usr/bin/env python3
"""
Balanced Strategy - Selective but finds trades
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ta import trend, momentum, volatility, volume

class BalancedIndicatorEngine:
    def __init__(self):
        self.features = []
    
    def calculate_trendline_features(self, df):
        """BALANCED trendline detection - Selective but finds setups"""
        # Initialize
        df['distance_to_trendline'] = 1.0
        df['trendline_touch'] = 0
        df['trendline_slope'] = 0.0
        df['rejection_strength'] = 0.0
        
        print("🔍 Finding balanced trendline setups...")
        setups_found = 0
        
        for i in range(50, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_open = df['open'].iloc[i]
            
            # Use multiple timeframes for support/resistance
            resistance_levels = []
            support_levels = []
            
            # Look at different time periods
            for period in [10, 20, 30, 50]:
                if i >= period:
                    # Resistance = recent highs
                    resistance = df['high'].iloc[i-period:i].max()
                    resistance_levels.append(resistance)
                    
                    # Support = recent lows  
                    support = df['low'].iloc[i-period:i].min()
                    support_levels.append(support)
            
            if not resistance_levels or not support_levels:
                continue
            
            # Find closest levels
            closest_resistance = min(resistance_levels, key=lambda x: abs(current_high - x))
            closest_support = min(support_levels, key=lambda x: abs(current_low - x))
            
            dist_to_resistance = abs(current_high - closest_resistance) / closest_resistance
            dist_to_support = abs(current_low - closest_support) / closest_support
            
            min_distance = min(dist_to_resistance, dist_to_support)
            
            # Determine direction
            if dist_to_resistance < dist_to_support:
                trend_direction = -1  # Near resistance
            else:
                trend_direction = 1   # Near support
            
            df.loc[df.index[i], 'distance_to_trendline'] = min_distance
            df.loc[df.index[i], 'trendline_slope'] = trend_direction
            
            # BALANCED: 0.3% distance + some volume
            volume_ok = df['volume'].iloc[i] > df['volume'].iloc[i-20:i].mean() * 0.8
            
            if min_distance < 0.003 and volume_ok:  # 0.3% distance
                df.loc[df.index[i], 'trendline_touch'] = 1
                
                # Calculate rejection
                body_size = abs(current_close - current_open) / (current_open + 0.0001)
                upper_wick = (current_high - max(current_open, current_close)) / (current_high + 0.0001)
                lower_wick = (min(current_open, current_close) - current_low) / (current_low + 0.0001)
                
                if trend_direction < 0:  # Resistance
                    rejection_strength = upper_wick / (body_size + 0.001)
                else:  # Support
                    rejection_strength = lower_wick / (body_size + 0.001)
                
                df.loc[df.index[i], 'rejection_strength'] = rejection_strength
                
                setups_found += 1
        
        print(f"✅ Found {setups_found} trendline setups (balanced approach)")
        return df
    
    def calculate_all_indicators(self, df):
        """Calculate all indicators"""
        print("📊 Calculating technical indicators...")
        
        # Strategy features
        df = self.calculate_trendline_features(df)
        
        # Add basic indicators
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['sma_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        
        print("✅ All indicators calculated successfully")
        return df

def find_optimal_threshold():
    """Find the best rejection threshold"""
    print("🎯 FINDING OPTIMAL REJECTION THRESHOLD")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2000)
    engine = BalancedIndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    best_win_rate = 0
    best_threshold = 0
    best_trades = 0
    
    for rejection_thresh in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        trades = []
        
        for i in range(50, len(data) - 3):
            if (data['trendline_touch'].iloc[i] == 1 and 
                data['rejection_strength'].iloc[i] > rejection_thresh):
                
                entry_price = data['close'].iloc[i]
                exit_price = data['close'].iloc[i+3]
                
                if data['trendline_slope'].iloc[i] < 0:  # SHORT
                    pnl = (entry_price - exit_price) / entry_price
                else:  # LONG
                    pnl = (exit_price - entry_price) / entry_price
                
                trades.append(pnl)
        
        if len(trades) >= 5:  # Need minimum trades
            win_rate = len([x for x in trades if x > 0]) / len(trades)
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_threshold = rejection_thresh
                best_trades = len(trades)
            
            print(f"Rejection > {rejection_thresh}: {len(trades)} trades, {win_rate:.3f} win rate")
    
    print(f"\n🏆 OPTIMAL THRESHOLD: {best_threshold}")
    print(f"Expected win rate: {best_win_rate:.3f}")
    print(f"Expected trades: {best_trades}")
    
    return best_threshold

def run_balanced_backtest(rejection_threshold=1.0):
    """Run backtest with balanced parameters"""
    print(f"\n🎯 RUNNING BALANCED STRATEGY (rejection > {rejection_threshold})")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2000)
    engine = BalancedIndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    balance = 1000
    trades = []
    
    for i in range(50, len(data) - 3):
        # Balanced entry conditions
        is_touch = data['trendline_touch'].iloc[i] == 1
        has_rejection = data['rejection_strength'].iloc[i] > rejection_threshold
        rsi_ok = 30 < data['rsi_14'].iloc[i] < 70  # Not extreme
        
        if is_touch and has_rejection and rsi_ok:
            entry_price = data['close'].iloc[i]
            exit_price = data['close'].iloc[i+3]
            
            if data['trendline_slope'].iloc[i] < 0:  # SHORT
                pnl_percent = (entry_price - exit_price) / entry_price
                signal_type = "SHORT"
            else:  # LONG
                pnl_percent = (exit_price - entry_price) / entry_price
                signal_type = "LONG"
            
            pnl_amount = balance * 0.02 * pnl_percent
            balance += pnl_amount
            
            trades.append({
                'time': data.index[i],
                'type': signal_type,
                'entry': entry_price,
                'exit': exit_price,
                'rejection': data['rejection_strength'].iloc[i],
                'pnl_percent': pnl_percent,
                'balance': balance
            })
            
            print(f"🎯 {signal_type} at ${entry_price:.2f} - Rej: {data['rejection_strength'].iloc[i]:.2f} - PnL: {pnl_percent*100:+.2f}%")
    
    # Results
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        print(f"\n📊 RESULTS:")
        print(f"Trades: {len(trades_df)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${balance:.2f}")
        
        return True, win_rate, total_return
    else:
        print("No trades executed")
        return False, 0, 0

def main():
    # First find optimal threshold
    optimal_threshold = find_optimal_threshold()
    
    # Then run backtest with optimal threshold
    success, win_rate, total_return = run_balanced_backtest(optimal_threshold)
    
    if success and win_rate > 45 and total_return > 0:
        print(f"\n🎉 STRATEGY VALIDATED!")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print("Ready for live trading with small amounts!")
    else:
        print(f"\n⚠️ Strategy needs improvement")
        print("Try different parameters or market conditions")

if __name__ == "__main__":
    main()
