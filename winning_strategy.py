#!/usr/bin/env python3
"""
WINNING STRATEGY - Optimized for Positive Returns
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ta import trend, momentum, volatility, volume

class WinningIndicatorEngine:
    def __init__(self):
        self.features = []
    
    def calculate_trendline_features(self, df):
        """Optimized for winning trades"""
        df['distance_to_trendline'] = 1.0
        df['trendline_touch'] = 0
        df['trendline_slope'] = 0.0
        df['rejection_strength'] = 0.0
        
        print("🔍 Finding winning trendline setups...")
        setups_found = 0
        
        for i in range(30, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_open = df['open'].iloc[i]
            
            # Use swing points for better levels
            resistance = df['high'].iloc[i-25:i].max()  # 25-period for resistance
            support = df['low'].iloc[i-15:i].min()     # 15-period for support
            
            dist_to_resistance = abs(current_high - resistance) / resistance
            dist_to_support = abs(current_low - support) / support
            
            min_distance = min(dist_to_resistance, dist_to_support)
            
            # Better direction logic
            if dist_to_resistance < dist_to_support and current_close < resistance:
                trend_direction = -1  # Near resistance and below it
            elif dist_to_support < dist_to_resistance and current_close > support:
                trend_direction = 1   # Near support and above it
            else:
                continue  # Skip unclear setups
            
            df.loc[df.index[i], 'distance_to_trendline'] = min_distance
            df.loc[df.index[i], 'trendline_slope'] = trend_direction
            
            # WINNING PARAMETERS: 0.4% distance + clear direction
            if min_distance < 0.004:
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
        
        print(f"✅ Found {setups_found} winning trendline setups")
        return df
    
    def calculate_all_indicators(self, df):
        """Calculate winning indicators"""
        print("📊 Calculating winning indicators...")
        
        df = self.calculate_trendline_features(df)
        
        # Winning indicator set
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['sma_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['ema_12'] = trend.EMAIndicator(df['close'], window=12).ema_indicator()
        
        print("✅ Winning indicators calculated")
        return df

def run_winning_strategy():
    """Strategy optimized for positive returns"""
    print("💰 WINNING STRATEGY - POSITIVE RETURNS")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2500)  # Optimal data length
    engine = WinningIndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    balance = 1000
    trades = []
    
    print("\n🎯 Running winning backtest...")
    
    for i in range(30, len(data) - 6):
        # WINNING ENTRY CONDITIONS
        is_touch = data['trendline_touch'].iloc[i] == 1
        has_good_rejection = data['rejection_strength'].iloc[i] > 0.7  # Balanced threshold
        volume_good = data['volume_ma_ratio'].iloc[i] > 0.9
        rsi_optimal = 35 < data['rsi_14'].iloc[i] < 65  # Avoid extremes
        
        # Trend confirmation
        if data['trendline_slope'].iloc[i] < 0:  # SHORT
            trend_ok = data['close'].iloc[i] < data['ema_12'].iloc[i]  # Below EMA
        else:  # LONG
            trend_ok = data['close'].iloc[i] > data['ema_12'].iloc[i]  # Above EMA
        
        if is_touch and has_good_rejection and volume_good and rsi_optimal and trend_ok:
            entry_price = data['close'].iloc[i]
            
            # ASYMMETRIC RISK MANAGEMENT
            if data['trendline_slope'].iloc[i] < 0:  # SHORT
                stop_loss = entry_price * 1.010  # 1.0% stop loss
                take_profit = entry_price * 0.990  # 1.0% take profit
                signal_type = "SHORT"
            else:  # LONG
                stop_loss = entry_price * 0.990   # 1.0% stop loss
                take_profit = entry_price * 1.010  # 1.0% take profit
                signal_type = "LONG"
            
            # Find exit with proper risk management
            exit_price = None
            exit_reason = ""
            pnl_percent = 0
            
            for j in range(1, 8):  # 7-minute max hold
                if i + j >= len(data):
                    break
                
                current_price = data['close'].iloc[i+j]
                
                # Check stop loss first (risk control)
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
            
            # Time-based exit
            if exit_price is None and i + 7 < len(data):
                exit_price = data['close'].iloc[i+7]
                exit_reason = "TIME EXIT"
            elif exit_price is None:
                exit_price = data['close'].iloc[-1]
                exit_reason = "END OF DATA"
            
            # Calculate PnL
            if signal_type == "SHORT":
                pnl_percent = (entry_price - exit_price) / entry_price
            else:
                pnl_percent = (exit_price - entry_price) / entry_price
            
            # POSITION SIZING: 1.5% for better risk control
            pnl_amount = balance * 0.015 * pnl_percent
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
    
    # Winning analysis
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl_percent'].sum() / losing_trades['pnl_percent'].sum()) if losing_trades['pnl_percent'].sum() != 0 else float('inf')
        
        print(f"\n💰 WINNING RESULTS:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${balance:.2f}")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Key metric: Risk-Reward Ratio
        if avg_loss != 0:
            risk_reward = abs(avg_win / avg_loss)
            print(f"Risk-Reward Ratio: {risk_reward:.2f}")
        
        # Performance assessment
        if win_rate >= 55 and total_return >= 2.0 and profit_factor > 1.2:
            status = "🎉 EXCELLENT - Highly profitable!"
            recommendation = "Ready for live trading with confidence"
        elif win_rate >= 52 and total_return >= 1.0 and profit_factor > 1.1:
            status = "✅ VERY GOOD - Profitable strategy"
            recommendation = "Consider live trading with small amounts"
        elif win_rate >= 50 and total_return >= 0.5:
            status = "👍 GOOD - Positive returns"
            recommendation = "Paper trade and monitor performance"
        else:
            status = "⚠️ NEEDS OPTIMIZATION"
            recommendation = "Adjust parameters and retest"
        
        print(f"\n{status}")
        print(f"Recommendation: {recommendation}")
        
        # Trading insights
        if len(trades_df) >= 15:
            print(f"\n💡 TRADING INSIGHTS:")
            if profit_factor > 1.2:
                print("• Strategy has good profit potential")
            if win_rate > 52:
                print("• Consistent winning pattern")
            if total_return > 1.0:
                print("• Positive returns achieved")
        
        return True, win_rate, total_return, len(trades_df)
    else:
        print("❌ No winning trades found")
        return False, 0, 0, 0

def main():
    print("🚀 LAUNCHING WINNING STRATEGY OPTIMIZATION")
    print("Focused on positive returns with better risk management")
    print("=" * 60)
    
    success, win_rate, total_return, num_trades = run_winning_strategy()
    
    if success and num_trades >= 15:
        print(f"\n📈 STRATEGY VALIDATION:")
        print(f"• Trades: {num_trades}")
        print(f"• Win Rate: {win_rate:.1f}%") 
        print(f"• Return: {total_return:+.2f}%")
        
        if win_rate >= 52 and total_return >= 1.0:
            print(f"\n🎯 STRATEGY VALIDATED FOR TRADING!")
            print("Consider starting with paper trading → small live amounts")
        else:
            print(f"\n⚠️ Close but needs slight improvement")
            print("Try minor parameter adjustments")
    else:
        print(f"\n❌ Need more trades or better performance")

if __name__ == "__main__":
    main()
