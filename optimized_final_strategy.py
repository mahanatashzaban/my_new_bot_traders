#!/usr/bin/env python3
"""
Optimized Final Strategy with Better Risk Management
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ta import trend, momentum, volatility, volume

class OptimizedIndicatorEngine:
    def __init__(self):
        self.features = []
    
    def calculate_trendline_features(self, df):
        """Optimized trendline detection"""
        df['distance_to_trendline'] = 1.0
        df['trendline_touch'] = 0
        df['trendline_slope'] = 0.0
        df['rejection_strength'] = 0.0
        
        print("🔍 Finding optimized trendline setups...")
        setups_found = 0
        
        for i in range(50, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_open = df['open'].iloc[i]
            
            # Use swing points for better levels
            resistance_levels = []
            support_levels = []
            
            for period in [20, 30, 50]:
                if i >= period:
                    # Use rolling high/low but with some filtering
                    recent_high = df['high'].iloc[i-period:i].max()
                    recent_low = df['low'].iloc[i-period:i].min()
                    
                    # Only add if it's a significant level (not too close to current price)
                    if abs(current_high - recent_high) / recent_high > 0.001:
                        resistance_levels.append(recent_high)
                    if abs(current_low - recent_low) / recent_low > 0.001:
                        support_levels.append(recent_low)
            
            if not resistance_levels or not support_levels:
                continue
            
            closest_resistance = min(resistance_levels, key=lambda x: abs(current_high - x))
            closest_support = min(support_levels, key=lambda x: abs(current_low - x))
            
            dist_to_resistance = abs(current_high - closest_resistance) / closest_resistance
            dist_to_support = abs(current_low - closest_support) / closest_support
            
            min_distance = min(dist_to_resistance, dist_to_support)
            
            if dist_to_resistance < dist_to_support:
                trend_direction = -1
            else:
                trend_direction = 1
            
            df.loc[df.index[i], 'distance_to_trendline'] = min_distance
            df.loc[df.index[i], 'trendline_slope'] = trend_direction
            
            # Optimized: 0.2% distance + volume
            volume_ok = df['volume'].iloc[i] > df['volume'].iloc[i-20:i].mean() * 0.9
            
            if min_distance < 0.002 and volume_ok:
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
        
        print(f"✅ Found {setups_found} optimized trendline setups")
        return df
    
    def calculate_all_indicators(self, df):
        """Calculate all indicators"""
        print("📊 Calculating technical indicators...")
        
        df = self.calculate_trendline_features(df)
        
        # Comprehensive indicators
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['sma_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['ema_12'] = trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['macd'] = trend.MACD(df['close']).macd()
        df['bb_upper'] = volatility.BollingerBands(df['close']).bollinger_hband()
        df['bb_lower'] = volatility.BollingerBands(df['close']).bollinger_lband()
        df['atr'] = volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        print("✅ All indicators calculated successfully")
        return df

def run_optimized_strategy():
    """Run optimized strategy with better risk management"""
    print("🎯 OPTIMIZED STRATEGY WITH RISK MANAGEMENT")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=3000)  # More data
    engine = OptimizedIndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    balance = 1000
    trades = []
    
    print("\n🔍 Running optimized backtest...")
    
    for i in range(100, len(data) - 5):
        # OPTIMIZED ENTRY CONDITIONS
        is_touch = data['trendline_touch'].iloc[i] == 1
        has_rejection = data['rejection_strength'].iloc[i] > 0.8  # Higher threshold
        good_volume = data['volume_ma_ratio'].iloc[i] > 1.0
        rsi_ok = 35 < data['rsi_14'].iloc[i] < 65  # Avoid extremes
        trend_aligned = True
        
        # Additional trend alignment check
        if data['trendline_slope'].iloc[i] < 0:  # SHORT
            trend_aligned = data['close'].iloc[i] < data['ema_12'].iloc[i]  # Below EMA
        else:  # LONG
            trend_aligned = data['close'].iloc[i] > data['ema_12'].iloc[i]  # Above EMA
        
        if is_touch and has_rejection and good_volume and rsi_ok and trend_aligned:
            entry_price = data['close'].iloc[i]
            
            # DYNAMIC STOP LOSS AND TAKE PROFIT
            atr = data['atr'].iloc[i]
            if data['trendline_slope'].iloc[i] < 0:  # SHORT
                stop_loss = entry_price * 1.015  # 1.5% stop loss
                take_profit = entry_price * 0.985  # 1.5% take profit
                signal_type = "SHORT"
            else:  # LONG
                stop_loss = entry_price * 0.985  # 1.5% stop loss
                take_profit = entry_price * 1.015  # 1.5% take profit
                signal_type = "LONG"
            
            # Simulate trade with proper exit conditions
            exit_price = None
            exit_reason = ""
            pnl_percent = 0
            
            for j in range(1, 11):  # Check next 10 minutes
                if i + j >= len(data):
                    break
                
                current_price = data['close'].iloc[i+j]
                
                # Check stop loss
                if (signal_type == "SHORT" and current_price >= stop_loss) or \
                   (signal_type == "LONG" and current_price <= stop_loss):
                    exit_price = current_price
                    exit_reason = "STOP LOSS"
                    break
                
                # Check take profit
                if (signal_type == "SHORT" and current_price <= take_profit) or \
                   (signal_type == "LONG" and current_price >= take_profit):
                    exit_price = current_price
                    exit_reason = "TAKE PROFIT"
                    break
            
            # If no exit in 10 minutes, exit at current price
            if exit_price is None and i + 10 < len(data):
                exit_price = data['close'].iloc[i+10]
                exit_reason = "TIME EXIT"
            elif exit_price is None:
                exit_price = data['close'].iloc[-1]
                exit_reason = "END OF DATA"
            
            # Calculate PnL
            if signal_type == "SHORT":
                pnl_percent = (entry_price - exit_price) / entry_price
            else:
                pnl_percent = (exit_price - entry_price) / entry_price
            
            pnl_amount = balance * 0.02 * pnl_percent  # 2% position size
            balance += pnl_amount
            
            trades.append({
                'time': data.index[i],
                'type': signal_type,
                'entry': entry_price,
                'exit': exit_price,
                'rejection': data['rejection_strength'].iloc[i],
                'pnl_percent': pnl_percent,
                'balance': balance,
                'reason': exit_reason
            })
            
            pnl_color = "🟢" if pnl_percent > 0 else "🔴"
            print(f"{pnl_color} {signal_type} at ${entry_price:.2f} - Rej: {data['rejection_strength'].iloc[i]:.2f} - PnL: {pnl_percent*100:+.2f}% - {exit_reason}")
    
    # Detailed analysis
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl_percent'].sum() / losing_trades['pnl_percent'].sum()) if losing_trades['pnl_percent'].sum() != 0 else float('inf')
        
        print(f"\n📊 OPTIMIZED RESULTS:")
        print(f"Trades: {len(trades_df)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${balance:.2f}")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Exit reason analysis
        exit_reasons = trades_df['reason'].value_counts()
        print(f"\n📋 Exit Reasons:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count} trades")
        
        # Trade type analysis
        long_trades = trades_df[trades_df['type'] == 'LONG']
        short_trades = trades_df[trades_df['type'] == 'SHORT']
        
        if len(long_trades) > 0:
            long_win_rate = len(long_trades[long_trades['pnl_percent'] > 0]) / len(long_trades) * 100
            print(f"LONG Trades: {len(long_trades)} | Win Rate: {long_win_rate:.1f}%")
        
        if len(short_trades) > 0:
            short_win_rate = len(short_trades[short_trades['pnl_percent'] > 0]) / len(short_trades) * 100
            print(f"SHORT Trades: {len(short_trades)} | Win Rate: {short_win_rate:.1f}%")
        
        # Validation
        if win_rate > 50 and total_return > 2.0:
            print(f"\n🎉 EXCELLENT! Strategy is ready for live trading!")
        elif win_rate > 45 and total_return > 0:
            print(f"\n✅ GOOD! Strategy shows promise.")
        else:
            print(f"\n⚠️ NEEDS IMPROVEMENT. Consider different parameters.")
        
        return True, win_rate, total_return
    else:
        print("No trades executed with current parameters")
        return False, 0, 0

def main():
    success, win_rate, total_return = run_optimized_strategy()
    
    if success:
        print(f"\n💡 NEXT STEPS:")
        print(f"1. If results are good, consider live trading with small amounts")
        print(f"2. Continue monitoring and optimizing")
        print(f"3. Add more filters if needed (volatility, time of day, etc.)")

if __name__ == "__main__":
    main()
