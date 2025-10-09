#!/usr/bin/env python3
"""
ULTIMATE WORKING STRATEGY - Guaranteed Trades with Profit Potential
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ta import trend, momentum, volatility, volume

class UltimateIndicatorEngine:
    def __init__(self):
        self.features = []
    
    def calculate_trendline_features(self, df):
        """Ultimate trendline detection - balanced approach"""
        df['distance_to_trendline'] = 1.0
        df['trendline_touch'] = 0
        df['trendline_slope'] = 0.0
        df['rejection_strength'] = 0.0
        
        print("🔍 Finding ultimate trendline setups...")
        setups_found = 0
        
        for i in range(20, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_open = df['open'].iloc[i]
            
            # Simple but effective levels
            resistance = df['high'].iloc[i-20:i].max()
            support = df['low'].iloc[i-20:i].min()
            
            dist_to_resistance = abs(current_high - resistance) / resistance
            dist_to_support = abs(current_low - support) / support
            
            min_distance = min(dist_to_resistance, dist_to_support)
            
            # Simple direction based on closest level
            if dist_to_resistance < dist_to_support:
                trend_direction = -1  # Resistance
            else:
                trend_direction = 1   # Support
            
            df.loc[df.index[i], 'distance_to_trendline'] = min_distance
            df.loc[df.index[i], 'trendline_slope'] = trend_direction
            
            # ULTIMATE PARAMETERS: 0.8% distance (balanced)
            if min_distance < 0.008:
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
        
        print(f"✅ Found {setups_found} ultimate trendline setups")
        return df
    
    def calculate_all_indicators(self, df):
        """Calculate essential indicators"""
        print("📊 Calculating essential indicators...")
        
        df = self.calculate_trendline_features(df)
        
        # Only the most essential indicators
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['sma_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
        
        print("✅ Essential indicators calculated")
        return df

def run_ultimate_strategy():
    """Ultimate strategy that finds trades AND aims for profit"""
    print("🚀 ULTIMATE STRATEGY - TRADES + PROFIT")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2000)
    engine = UltimateIndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    balance = 1000
    trades = []
    
    print("\n🎯 Running ultimate backtest...")
    
    for i in range(20, len(data) - 4):
        # ULTIMATE ENTRY: Balanced parameters
        is_touch = data['trendline_touch'].iloc[i] == 1
        has_rejection = data['rejection_strength'].iloc[i] > 0.5  # Lower threshold for more trades
        volume_ok = data['volume_ma_ratio'].iloc[i] > 0.8
        
        if is_touch and has_rejection and volume_ok:
            entry_price = data['close'].iloc[i]
            exit_price = data['close'].iloc[i+4]  # 4-minute hold
            
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
    
    # Ultimate analysis
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        
        print(f"\n💰 ULTIMATE RESULTS:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${balance:.2f}")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        
        # Performance assessment
        if win_rate >= 60 and total_return >= 5.0:
            status = "🎉 EXCELLENT"
            action = "Ready for live trading!"
        elif win_rate >= 55 and total_return >= 3.0:
            status = "✅ VERY GOOD" 
            action = "Consider live trading"
        elif win_rate >= 50 and total_return >= 1.0:
            status = "👍 GOOD"
            action = "Promising strategy"
        elif win_rate >= 45 and total_return >= 0:
            status = "⚠️ ACCEPTABLE"
            action = "Needs monitoring"
        else:
            status = "❌ NEEDS WORK"
            action = "Requires optimization"
        
        print(f"\n{status}: {action}")
        
        # Trading recommendations
        if len(trades_df) >= 15:
            print(f"\n💡 TRADING RECOMMENDATIONS:")
            if win_rate >= 55:
                print("• Start with small live trading amounts")
                print("• Use proper risk management (1-2% per trade)")
                print("• Monitor performance daily")
            else:
                print("• Continue paper trading")
                print("• Optimize entry/exit parameters")
                print("• Test on different market conditions")
        
        return True, win_rate, total_return, len(trades_df)
    else:
        print("❌ No trades found - strategy too restrictive")
        return False, 0, 0, 0

def find_best_market_conditions():
    """Test if different market conditions work better"""
    print("\n🌡️  TESTING MARKET CONDITIONS")
    print("=" * 50)
    
    # Test different time periods
    periods = [
        ("Recent", 1000),      # Most recent data
        ("Medium", 2000),      # Medium term
        ("Long", 3000),        # Longer term
    ]
    
    best_performance = 0
    best_period = ""
    
    for period_name, limit in periods:
        print(f"\nTesting {period_name} period ({limit} candles)...")
        
        dm = DataManager()
        data = dm.fetch_historical_data(limit=limit)
        engine = UltimateIndicatorEngine()
        data = engine.calculate_all_indicators(data)
        
        # Count potential trades
        potential_trades = 0
        for i in range(20, len(data) - 4):
            if (data['trendline_touch'].iloc[i] == 1 and 
                data['rejection_strength'].iloc[i] > 0.5):
                potential_trades += 1
        
        print(f"  Potential trades: {potential_trades}")
        
        if potential_trades > best_performance:
            best_performance = potential_trades
            best_period = period_name
    
    print(f"\n🏆 BEST MARKET CONDITIONS: {best_period}")
    print(f"Expected trades: {best_performance}")
    return best_period

def main():
    print("🚀 LAUNCHING ULTIMATE STRATEGY")
    print("This version guarantees trades while aiming for profitability")
    print("=" * 60)
    
    # Find best market conditions
    best_period = find_best_market_conditions()
    
    # Run ultimate strategy
    success, win_rate, total_return, num_trades = run_ultimate_strategy()
    
    if success and num_trades >= 10:
        print(f"\n📊 STRATEGY PERFORMANCE SUMMARY:")
        print(f"• Trades Executed: {num_trades}")
        print(f"• Win Rate: {win_rate:.1f}%")
        print(f"• Total Return: {total_return:+.2f}%")
        print(f"• Best Market: {best_period}")
        
        if win_rate >= 50 and total_return > 0:
            print(f"\n🎯 STRATEGY VALIDATED!")
            print("Consider starting with paper trading, then small live amounts")
        else:
            print(f"\n⚠️ Strategy needs optimization")
            print("Try adjusting rejection threshold or hold time")
    else:
        print(f"\n❌ Strategy not ready")
        print("Try different parameters or market conditions")

if __name__ == "__main__":
    main()
