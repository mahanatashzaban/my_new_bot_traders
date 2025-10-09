#!/usr/bin/env python3
"""
RISK-OPTIMIZED STRATEGY - Better Risk-Reward Ratio
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ta import trend, momentum, volatility, volume

class RiskOptimizedEngine:
    def __init__(self):
        self.features = []
    
    def calculate_trendline_features(self, df):
        """Optimized for better risk-reward"""
        df['distance_to_trendline'] = 1.0
        df['trendline_touch'] = 0
        df['trendline_slope'] = 0.0
        df['rejection_strength'] = 0.0
        
        print("🔍 Finding risk-optimized setups...")
        setups_found = 0
        
        for i in range(25, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_open = df['open'].iloc[i]
            
            # Better level detection
            resistance_20 = df['high'].iloc[i-20:i].max()
            resistance_10 = df['high'].iloc[i-10:i].max()
            support_20 = df['low'].iloc[i-20:i].min()
            support_10 = df['low'].iloc[i-10:i].min()
            
            # Use the most relevant level
            resistance = max(resistance_10, resistance_20)
            support = min(support_10, support_20)
            
            dist_to_resistance = abs(current_high - resistance) / resistance
            dist_to_support = abs(current_low - support) / support
            
            min_distance = min(dist_to_resistance, dist_to_support)
            
            # Improved direction logic
            if dist_to_resistance < dist_to_support and current_close < resistance:
                trend_direction = -1  # Near resistance and below it
            elif dist_to_support < dist_to_resistance and current_close > support:
                trend_direction = 1   # Near support and above it
            else:
                trend_direction = 0   # Skip unclear
            
            df.loc[df.index[i], 'distance_to_trendline'] = min_distance
            df.loc[df.index[i], 'trendline_slope'] = trend_direction
            
            # RISK-OPTIMIZED: 0.5% distance + clear direction
            if min_distance < 0.005 and trend_direction != 0:
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
        
        print(f"✅ Found {setups_found} risk-optimized setups")
        return df
    
    def calculate_all_indicators(self, df):
        """Calculate risk management indicators"""
        print("📊 Calculating risk indicators...")
        
        df = self.calculate_trendline_features(df)
        
        # Risk management indicators
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['atr'] = volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['sma_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
        
        print("✅ Risk indicators calculated")
        return df

def run_risk_optimized_strategy():
    """Strategy with better risk-reward management"""
    print("💰 RISK-OPTIMIZED STRATEGY")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2000)
    engine = RiskOptimizedEngine()
    data = engine.calculate_all_indicators(data)
    
    balance = 1000
    trades = []
    
    print("\n🎯 Running risk-optimized backtest...")
    
    for i in range(25, len(data) - 5):
        # RISK-OPTIMIZED ENTRY
        is_touch = data['trendline_touch'].iloc[i] == 1
        has_good_rejection = data['rejection_strength'].iloc[i] > 0.8  # Higher quality
        volume_ok = data['volume_ma_ratio'].iloc[i] > 0.9
        rsi_good = 40 < data['rsi_14'].iloc[i] < 60  # Avoid extremes
        
        if is_touch and has_good_rejection and volume_ok and rsi_good:
            entry_price = data['close'].iloc[i]
            atr = data['atr'].iloc[i]
            
            # IMPROVED RISK MANAGEMENT
            if data['trendline_slope'].iloc[i] < 0:  # SHORT
                stop_loss = entry_price * 1.008   # 0.8% stop loss
                take_profit = entry_price * 0.992  # 0.8% take profit
                signal_type = "SHORT"
            else:  # LONG
                stop_loss = entry_price * 0.992   # 0.8% stop loss
                take_profit = entry_price * 1.008  # 0.8% take profit
                signal_type = "LONG"
            
            # Find exit with risk management
            exit_price = None
            exit_reason = ""
            
            for j in range(1, 6):  # 5-minute max hold
                if i + j >= len(data):
                    break
                
                current_price = data['close'].iloc[i+j]
                
                # Stop loss check
                if (signal_type == "SHORT" and current_price >= stop_loss) or \
                   (signal_type == "LONG" and current_price <= stop_loss):
                    exit_price = current_price
                    exit_reason = "STOP LOSS"
                    break
                
                # Take profit check
                if (signal_type == "SHORT" and current_price <= take_profit) or \
                   (signal_type == "LONG" and current_price >= take_profit):
                    exit_price = current_price
                    exit_reason = "TAKE PROFIT"
                    break
            
            # Time exit
            if exit_price is None and i + 5 < len(data):
                exit_price = data['close'].iloc[i+5]
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
    
    # Risk-optimized analysis
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        
        # Key metric: Risk-Reward Ratio
        if avg_loss != 0:
            risk_reward = abs(avg_win / avg_loss)
        else:
            risk_reward = float('inf')
        
        # Profit factor
        if losing_trades['pnl_percent'].sum() != 0:
            profit_factor = abs(winning_trades['pnl_percent'].sum() / losing_trades['pnl_percent'].sum())
        else:
            profit_factor = float('inf')
        
        print(f"\n💰 RISK-OPTIMIZED RESULTS:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${balance:.2f}")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        print(f"Risk-Reward Ratio: {risk_reward:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Exit analysis
        exit_reasons = trades_df['reason'].value_counts()
        print(f"\n📋 EXIT ANALYSIS:")
        for reason, count in exit_reasons.items():
            reason_trades = trades_df[trades_df['reason'] == reason]
            reason_pnl = reason_trades['pnl_percent'].mean() * 100
            print(f"  {reason}: {count} trades, avg: {reason_pnl:+.2f}%")
        
        # STRATEGY ASSESSMENT
        if win_rate >= 55 and total_return >= 2.0 and risk_reward >= 1.2:
            assessment = "🎉 EXCELLENT - Highly profitable!"
            action = "Ready for live trading"
        elif win_rate >= 52 and total_return >= 1.0 and risk_reward >= 1.1:
            assessment = "✅ VERY GOOD - Profitable strategy"
            action = "Consider live trading"
        elif win_rate >= 50 and total_return >= 0.5:
            assessment = "👍 GOOD - Positive returns"
            action = "Paper trade and monitor"
        elif win_rate >= 48 and total_return >= 0:
            assessment = "⚠️ ACCEPTABLE - Break-even"
            action = "Needs optimization"
        else:
            assessment = "❌ NEEDS WORK - Not profitable"
            action = "Requires significant changes"
        
        print(f"\n{assessment}")
        print(f"Action: {action}")
        
        # Trading recommendations
        if len(trades_df) >= 10:
            print(f"\n💡 TRADING INSIGHTS:")
            if risk_reward < 1.0:
                print("• Improve risk-reward ratio (tighten stops)")
            if win_rate < 50:
                print("• Improve entry timing (better filters)")
            if total_return < 0:
                print("• Focus on reducing average loss size")
        
        return True, win_rate, total_return, risk_reward
    else:
        print("❌ No risk-optimized trades found")
        return False, 0, 0, 0

def main():
    print("🚀 LAUNCHING RISK-OPTIMIZED STRATEGY")
    print("Focused on improving risk-reward ratio for profitability")
    print("=" * 60)
    
    success, win_rate, total_return, risk_reward = run_risk_optimized_strategy()
    
    if success:
        print(f"\n📈 FINAL STRATEGY PERFORMANCE:")
        print(f"• Win Rate: {win_rate:.1f}%")
        print(f"• Total Return: {total_return:+.2f}%")
        print(f"• Risk-Reward: {risk_reward:.2f}")
        
        if win_rate >= 52 and total_return >= 1.0:
            print(f"\n🎯 STRATEGY VALIDATED!")
            print("Ready for paper trading → small live amounts")
        elif win_rate >= 50 and total_return >= 0:
            print(f"\n⚠️ Strategy shows promise")
            print("Continue optimizing risk management")
        else:
            print(f"\n💡 Keep testing different parameters")
    else:
        print(f"\n🔧 Try adjusting entry criteria")

if __name__ == "__main__":
    main()
