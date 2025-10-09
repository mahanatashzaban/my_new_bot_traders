#!/usr/bin/env python3
"""
PRODUCTION STRATEGY - Ready for Live Trading
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ta import trend, momentum, volatility, volume

class ProductionEngine:
    def __init__(self):
        self.features = []
    
    def calculate_trendline_features(self, df):
        """Production-ready trendline detection"""
        df['distance_to_trendline'] = 1.0
        df['trendline_touch'] = 0
        df['trendline_slope'] = 0.0
        df['rejection_strength'] = 0.0
        
        print("🔍 Finding production setups...")
        setups_found = 0
        
        for i in range(20, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_open = df['open'].iloc[i]
            
            # Production levels - balanced approach
            resistance = df['high'].iloc[i-15:i].max()  # 15-period resistance
            support = df['low'].iloc[i-15:i].min()     # 15-period support
            
            dist_to_resistance = abs(current_high - resistance) / resistance
            dist_to_support = abs(current_low - support) / support
            
            min_distance = min(dist_to_resistance, dist_to_support)
            
            # Production direction
            if dist_to_resistance < dist_to_support:
                trend_direction = -1  # Resistance
            else:
                trend_direction = 1   # Support
            
            df.loc[df.index[i], 'distance_to_trendline'] = min_distance
            df.loc[df.index[i], 'trendline_slope'] = trend_direction
            
            # PRODUCTION PARAMETERS: 0.7% distance
            if min_distance < 0.007:
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
        
        print(f"✅ Found {setups_found} production setups")
        return df
    
    def calculate_all_indicators(self, df):
        """Calculate production indicators"""
        print("📊 Calculating production indicators...")
        
        df = self.calculate_trendline_features(df)
        
        # Production indicators
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['sma_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
        
        print("✅ Production indicators calculated")
        return df

def run_production_strategy():
    """Production strategy ready for trading"""
    print("💰 PRODUCTION TRADING STRATEGY")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=1500)
    engine = ProductionEngine()
    data = engine.calculate_all_indicators(data)
    
    balance = 1000
    trades = []
    
    print("\n🎯 Executing production trades...")
    
    for i in range(20, len(data) - 3):
        # PRODUCTION ENTRY: Balanced for trade frequency
        is_touch = data['trendline_touch'].iloc[i] == 1
        has_rejection = data['rejection_strength'].iloc[i] > 0.7  # Good quality
        volume_ok = data['volume_ma_ratio'].iloc[i] > 0.8
        
        if is_touch and has_rejection and volume_ok:
            entry_price = data['close'].iloc[i]
            exit_price = data['close'].iloc[i+3]  # 3-minute hold
            
            if data['trendline_slope'].iloc[i] < 0:  # SHORT
                pnl_percent = (entry_price - exit_price) / entry_price
                signal_type = "SHORT"
            else:  # LONG
                pnl_percent = (exit_price - entry_price) / entry_price
                signal_type = "LONG"
            
            # PRODUCTION RISK: 2% position size
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
    
    # Production analysis
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        
        print(f"\n💰 PRODUCTION RESULTS:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${balance:.2f}")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        
        # PRODUCTION DECISION MATRIX
        if len(trades_df) >= 10:
            if win_rate >= 55 and total_return >= 2.0:
                status = "🎉 PRODUCTION READY"
                recommendation = "Deploy to live trading"
            elif win_rate >= 52 and total_return >= 1.0:
                status = "✅ NEAR PRODUCTION"
                recommendation = "Paper trade then go live"
            elif win_rate >= 50 and total_return >= 0.5:
                status = "👍 PROMISING"
                recommendation = "Continue optimization"
            else:
                status = "⚠️ DEVELOPMENT"
                recommendation = "Needs more work"
        else:
            status = "📊 INSUFFICIENT DATA"
            recommendation = f"Need more trades (current: {len(trades_df)})"
        
        print(f"\n{status}")
        print(f"Recommendation: {recommendation}")
        
        # Trading insights
        if len(trades_df) >= 8:
            print(f"\n💡 TRADING INSIGHTS:")
            print(f"• Trade Frequency: {len(trades_df)} trades")
            print(f"• Performance: {win_rate:.1f}% win rate, {total_return:+.2f}% return")
            
            if total_return > 0:
                print("• Strategy shows profit potential")
            else:
                print("• Focus on improving win rate or risk management")
        
        return True, win_rate, total_return, len(trades_df)
    else:
        print("❌ No production trades executed")
        return False, 0, 0, 0

def run_live_trading_simulation():
    """Simulate live trading conditions"""
    print("\n🔄 LIVE TRADING SIMULATION")
    print("=" * 50)
    
    # Test multiple periods to simulate live trading
    periods = [800, 1200, 1500]  # Different data lengths
    all_results = []
    
    for period in periods:
        print(f"\nTesting with {period} candles...")
        
        dm = DataManager()
        data = dm.fetch_historical_data(limit=period)
        engine = ProductionEngine()
        data = engine.calculate_all_indicators(data)
        
        balance = 1000
        trades_count = 0
        
        for i in range(20, len(data) - 3):
            if (data['trendline_touch'].iloc[i] == 1 and 
                data['rejection_strength'].iloc[i] > 0.7 and
                data['volume_ma_ratio'].iloc[i] > 0.8):
                trades_count += 1
        
        all_results.append({
            'period': period,
            'trades': trades_count,
            'trades_per_day': trades_count / (period / 1440)  # Approx trades per day
        })
        
        print(f"  Trades found: {trades_count}")
        print(f"  Estimated daily trades: {trades_count / (period / 1440):.1f}")
    
    # Analyze simulation results
    avg_daily_trades = np.mean([r['trades_per_day'] for r in all_results])
    print(f"\n📈 LIVE TRADING ESTIMATES:")
    print(f"Average Daily Trades: {avg_daily_trades:.1f}")
    
    if avg_daily_trades >= 5:
        print("• Good trade frequency for live trading")
    elif avg_daily_trades >= 3:
        print("• Reasonable trade frequency")
    else:
        print("• Low trade frequency - consider higher timeframe")
    
    return avg_daily_trades

def main():
    print("🚀 PRODUCTION STRATEGY LAUNCH")
    print("Final assessment for live trading readiness")
    print("=" * 60)
    
    # Run production strategy
    success, win_rate, total_return, num_trades = run_production_strategy()
    
    # Run live trading simulation
    daily_trades = run_live_trading_simulation()
    
    # FINAL ASSESSMENT
    print(f"\n🎯 FINAL PRODUCTION ASSESSMENT:")
    print(f"• Historical Performance: {win_rate:.1f}% win rate, {total_return:+.2f}% return")
    print(f"• Trade Frequency: {num_trades} trades in test, {daily_trades:.1f} daily estimated")
    print(f"• Strategy: Trendline Rejection with Risk Management")
    
    # PRODUCTION DECISION
    if success and num_trades >= 10 and win_rate >= 52 and total_return >= 1.0:
        print(f"\n🎉 PRODUCTION APPROVED!")
        print("Strategy meets production criteria")
        print("\n📝 LIVE TRADING PLAN:")
        print("1. Start with paper trading for 1 week")
        print("2. If consistent, trade with small amounts ($100-500)")
        print("3. Use 1-2% position sizing")
        print("4. Monitor performance daily")
        print("5. Adjust parameters if needed")
    elif success and num_trades >= 8:
        print(f"\n⚠️ NEAR PRODUCTION")
        print("Strategy shows promise but needs more testing")
        print("Continue paper trading and optimization")
    else:
        print(f"\n🔧 DEVELOPMENT PHASE")
        print("Strategy needs more development before live trading")
        print("Focus on improving trade frequency and consistency")

if __name__ == "__main__":
    main()
