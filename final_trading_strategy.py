#!/usr/bin/env python3
"""
FINAL TRADING STRATEGY - Executes Trades with Profit Goal
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ta import trend, momentum, volatility, volume

class TradingIndicatorEngine:
    def __init__(self):
        self.features = []
    
    def calculate_trendline_features(self, df):
        """Trading-focused trendline detection"""
        df['distance_to_trendline'] = 1.0
        df['trendline_touch'] = 0
        df['trendline_slope'] = 0.0
        df['rejection_strength'] = 0.0
        
        print("🔍 Finding trading setups...")
        setups_found = 0
        
        for i in range(20, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_open = df['open'].iloc[i]
            
            # Simple support/resistance
            resistance = df['high'].iloc[i-20:i].max()
            support = df['low'].iloc[i-20:i].min()
            
            dist_to_resistance = abs(current_high - resistance) / resistance
            dist_to_support = abs(current_low - support) / support
            
            min_distance = min(dist_to_resistance, dist_to_support)
            
            # Simple direction
            if dist_to_resistance < dist_to_support:
                trend_direction = -1  # Resistance
            else:
                trend_direction = 1   # Support
            
            df.loc[df.index[i], 'distance_to_trendline'] = min_distance
            df.loc[df.index[i], 'trendline_slope'] = trend_direction
            
            # TRADING PARAMETERS: 0.6% distance
            if min_distance < 0.006:
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
        
        print(f"✅ Found {setups_found} trading setups")
        return df
    
    def calculate_all_indicators(self, df):
        """Calculate trading indicators"""
        print("📊 Calculating trading indicators...")
        
        df = self.calculate_trendline_features(df)
        
        # Trading indicators only
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        print("✅ Trading indicators calculated")
        return df

def run_trading_strategy():
    """Final trading strategy that executes trades"""
    print("💰 FINAL TRADING STRATEGY")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=1500)  # Good data length
    engine = TradingIndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    balance = 1000
    trades = []
    
    print("\n🎯 Executing trades...")
    
    for i in range(20, len(data) - 4):
        # TRADING ENTRY: Simple but effective
        is_touch = data['trendline_touch'].iloc[i] == 1
        has_rejection = data['rejection_strength'].iloc[i] > 0.6  # Good threshold
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
            
            # Trading: 2% position size
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
    
    # Trading results
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        
        print(f"\n💰 TRADING RESULTS:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${balance:.2f}")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        
        # Trading decision
        if win_rate >= 55 and total_return >= 3.0:
            decision = "🎉 EXCELLENT - Trade with confidence!"
        elif win_rate >= 52 and total_return >= 1.5:
            decision = "✅ VERY GOOD - Ready for trading"
        elif win_rate >= 50 and total_return >= 0.5:
            decision = "👍 GOOD - Can start trading"
        elif win_rate >= 48 and total_return >= 0:
            decision = "⚠️ ACCEPTABLE - Paper trade first"
        else:
            decision = "❌ NEEDS WORK - Not ready"
        
        print(f"\n{decision}")
        
        # Trading recommendations
        if len(trades_df) >= 10:
            print(f"\n💡 TRADING RECOMMENDATIONS:")
            if total_return > 0:
                print("• Strategy shows profit potential")
                print("• Consider starting with small amounts")
                print("• Monitor performance closely")
            else:
                print("• Strategy needs optimization")
                print("• Try different timeframes or markets")
                print("• Consider adding filters")
        
        return True, win_rate, total_return, len(trades_df)
    else:
        print("❌ No trades executed")
        return False, 0, 0, 0

def test_different_markets():
    """Test if strategy works on different conditions"""
    print("\n🌍 TESTING DIFFERENT MARKET CONDITIONS")
    print("=" * 50)
    
    test_cases = [
        ("Normal", 1500),
        ("Short-term", 800),
        ("Recent", 500),
    ]
    
    best_case = None
    best_trades = 0
    
    for case_name, data_limit in test_cases:
        print(f"\nTesting {case_name} market...")
        
        dm = DataManager()
        data = dm.fetch_historical_data(limit=data_limit)
        engine = TradingIndicatorEngine()
        data = engine.calculate_all_indicators(data)
        
        # Count potential trades
        potential = 0
        for i in range(20, len(data) - 4):
            if (data['trendline_touch'].iloc[i] == 1 and 
                data['rejection_strength'].iloc[i] > 0.6):
                potential += 1
        
        print(f"  Potential trades: {potential}")
        
        if potential > best_trades:
            best_trades = potential
            best_case = case_name
    
    print(f"\n🏆 BEST MARKET: {best_case}")
    return best_case

def main():
    print("🚀 FINAL TRADING STRATEGY LAUNCH")
    print("This version executes trades and evaluates profitability")
    print("=" * 60)
    
    # Test different markets
    best_market = test_different_markets()
    
    # Run trading strategy
    success, win_rate, total_return, num_trades = run_trading_strategy()
    
    if success and num_trades >= 8:
        print(f"\n📊 FINAL ASSESSMENT:")
        print(f"• Trades Executed: {num_trades}")
        print(f"• Win Rate: {win_rate:.1f}%")
        print(f"• Total Return: {total_return:+.2f}%")
        print(f"• Best Market: {best_market}")
        
        if win_rate >= 50 and total_return > 0:
            print(f"\n🎯 STRATEGY IS PROFITABLE!")
            print("Consider starting with paper trading → small live amounts")
        elif win_rate >= 48:
            print(f"\n⚠️ Strategy is close to profitable")
            print("Minor optimizations needed")
        else:
            print(f"\n❌ Strategy needs work")
            print("Try different parameters or markets")
    else:
        print(f"\n💡 SUGGESTIONS:")
        print("• Try different rejection thresholds (0.4-0.8)")
        print("• Test different hold times (2-6 minutes)")
        print("• Consider different market hours")

if __name__ == "__main__":
    main()
