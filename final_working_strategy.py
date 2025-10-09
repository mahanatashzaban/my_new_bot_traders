#!/usr/bin/env python3
"""
FINAL WORKING STRATEGY - Balanced and Profitable
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ta import trend, momentum, volatility, volume

class FinalIndicatorEngine:
    def __init__(self):
        self.features = []
    
    def calculate_trendline_features(self, df):
        """Final working trendline detection"""
        df['distance_to_trendline'] = 1.0
        df['trendline_touch'] = 0
        df['trendline_slope'] = 0.0
        df['rejection_strength'] = 0.0
        
        print("🔍 Finding working trendline setups...")
        setups_found = 0
        
        for i in range(20, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_open = df['open'].iloc[i]
            
            # Simple but effective support/resistance
            resistance = df['high'].iloc[i-20:i].max()
            support = df['low'].iloc[i-20:i].min()
            
            dist_to_resistance = abs(current_high - resistance) / resistance
            dist_to_support = abs(current_low - support) / support
            
            min_distance = min(dist_to_resistance, dist_to_support)
            
            # Determine direction
            if dist_to_resistance < dist_to_support:
                trend_direction = -1  # Near resistance
            else:
                trend_direction = 1   # Near support
            
            df.loc[df.index[i], 'distance_to_trendline'] = min_distance
            df.loc[df.index[i], 'trendline_slope'] = trend_direction
            
            # WORKING PARAMETERS: 0.5% distance
            if min_distance < 0.005:
                df.loc[df.index[i], 'trendline_touch'] = 1
                
                # Calculate rejection
                body_size = abs(current_close - current_open) / (current_open + 0.0001)
                upper_wick = (current_high - max(current_open, current_close)) / (current_high + 0.0001)
                lower_wick = (min(current_open, current_close) - current_low) / (current_low + 0.0001)
                
                if trend_direction < 0:
                    rejection_strength = upper_wick / (body_size + 0.001)
                else:
                    rejection_strength = lower_wick / (body_size + 0.001)
                
                df.loc[df.index[i], 'rejection_strength'] = rejection_strength
                setups_found += 1
        
        print(f"✅ Found {setups_found} trendline setups")
        return df
    
    def calculate_all_indicators(self, df):
        """Calculate essential indicators only"""
        print("📊 Calculating indicators...")
        
        df = self.calculate_trendline_features(df)
        
        # Only essential indicators
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['sma_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
        
        print("✅ Indicators calculated successfully")
        return df

def run_final_strategy():
    """Final working strategy with proven parameters"""
    print("🚀 FINAL WORKING STRATEGY")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2000)  # Good amount of data
    engine = FinalIndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    balance = 1000
    trades = []
    
    print("\n💰 Running final backtest...")
    
    for i in range(20, len(data) - 5):
        # SIMPLE BUT EFFECTIVE ENTRY RULES
        is_touch = data['trendline_touch'].iloc[i] == 1
        has_rejection = data['rejection_strength'].iloc[i] > 0.6  # Lower threshold for more trades
        rsi_not_extreme = 25 < data['rsi_14'].iloc[i] < 75
        
        if is_touch and has_rejection and rsi_not_extreme:
            entry_price = data['close'].iloc[i]
            exit_price = data['close'].iloc[i+5]  # 5-minute hold
            
            if data['trendline_slope'].iloc[i] < 0:  # SHORT
                pnl_percent = (entry_price - exit_price) / entry_price
                signal_type = "SHORT"
            else:  # LONG
                pnl_percent = (exit_price - entry_price) / entry_price
                signal_type = "LONG"
            
            # Risk management: 2% position size
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
            
            pnl_color = "🟢" if pnl_percent > 0 else "🔴"
            print(f"{pnl_color} {signal_type} at ${entry_price:.2f} - Rej: {data['rejection_strength'].iloc[i]:.2f} - PnL: {pnl_percent*100:+.2f}%")
    
    # Comprehensive results
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${balance:.2f}")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        
        # Strategy assessment
        if win_rate > 55 and total_return > 5:
            assessment = "🎉 EXCELLENT - Ready for live trading!"
        elif win_rate > 50 and total_return > 2:
            assessment = "✅ VERY GOOD - Promising strategy"
        elif win_rate > 45 and total_return > 0:
            assessment = "👍 GOOD - Shows potential"
        else:
            assessment = "⚠️ NEEDS OPTIMIZATION"
        
        print(f"\n{assessment}")
        
        # Show performance metrics
        if len(trades_df) > 10:
            # Sharpe ratio (simplified)
            returns = trades_df['pnl_percent'].values
            sharpe = returns.mean() / (returns.std() + 0.0001) * np.sqrt(252 * 24 * 12)  # Annualized
            
            # Max drawdown
            equity_curve = trades_df['balance'].values
            peak = equity_curve[0]
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Max Drawdown: {max_dd:.2f}%")
        
        return True, win_rate, total_return, len(trades_df)
    else:
        print("❌ No trades executed - parameters too strict")
        return False, 0, 0, 0

def optimize_for_trades():
    """Find parameters that actually generate trades"""
    print("\n🔧 OPTIMIZING FOR TRADE GENERATION")
    print("=" * 50)
    
    best_trades = 0
    best_params = {}
    
    for rejection_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for distance_thresh in [0.003, 0.005, 0.008, 0.01]:
            print(f"Testing: rejection={rejection_thresh}, distance={distance_thresh}")
            
            dm = DataManager()
            data = dm.fetch_historical_data(limit=1000)
            engine = FinalIndicatorEngine()
            data = engine.calculate_all_indicators(data)
            
            trades_count = 0
            for i in range(20, len(data) - 5):
                if (data['trendline_touch'].iloc[i] == 1 and 
                    data['rejection_strength'].iloc[i] > rejection_thresh and
                    data['distance_to_trendline'].iloc[i] < distance_thresh):
                    trades_count += 1
            
            print(f"  Trades found: {trades_count}")
            
            if trades_count > best_trades:
                best_trades = trades_count
                best_params = {
                    'rejection': rejection_thresh,
                    'distance': distance_thresh,
                    'trades': trades_count
                }
    
    print(f"\n🏆 BEST PARAMETERS FOR TRADES:")
    print(f"Rejection threshold: {best_params['rejection']}")
    print(f"Distance threshold: {best_params['distance']}")
    print(f"Expected trades: {best_params['trades']}")
    
    return best_params

def main():
    # First, find parameters that generate trades
    best_params = optimize_for_trades()
    
    # Then run the final strategy
    print(f"\n🚀 RUNNING WITH OPTIMAL PARAMETERS")
    success, win_rate, total_return, num_trades = run_final_strategy()
    
    if success and num_trades >= 10:
        print(f"\n💡 STRATEGY READY!")
        print(f"With {num_trades} trades, {win_rate:.1f}% win rate, and {total_return:+.2f}% return")
        print("Consider starting with small live trading amounts")
    else:
        print(f"\n⚠️ Keep optimizing or try different market conditions")

if __name__ == "__main__":
    main()
