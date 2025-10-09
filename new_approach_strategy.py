#!/usr/bin/env python3
"""
COMPLETELY NEW APPROACH - Fixing Fundamental Issues
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ta import trend, momentum, volatility, volume
import matplotlib.pyplot as plt

class NewApproachEngine:
    def __init__(self):
        self.features = []
    
    def find_real_support_resistance(self, df, window=20):
        """Find REAL support/resistance using price clustering"""
        print("🎯 Finding REAL support/resistance levels...")
        
        df['support_level'] = 0.0
        df['resistance_level'] = 0.0
        df['near_support'] = 0
        df['near_resistance'] = 0
        
        for i in range(window, len(df)):
            # Look for price clusters (areas where price spent time)
            recent_lows = df['low'].iloc[i-window:i]
            recent_highs = df['high'].iloc[i-window:i]
            
            # Find significant support (price bounced from here multiple times)
            support_candidates = []
            for j in range(len(recent_lows)):
                low_price = recent_lows.iloc[j]
                # Count how many times price touched near this level
                touch_count = sum(abs(recent_lows - low_price) / low_price < 0.002)
                if touch_count >= 3:  # At least 3 touches
                    support_candidates.append(low_price)
            
            # Find significant resistance
            resistance_candidates = []
            for j in range(len(recent_highs)):
                high_price = recent_highs.iloc[j]
                touch_count = sum(abs(recent_highs - high_price) / high_price < 0.002)
                if touch_count >= 3:
                    resistance_candidates.append(high_price)
            
            if support_candidates:
                df.loc[df.index[i], 'support_level'] = max(support_candidates)  # Strongest support
            if resistance_candidates:
                df.loc[df.index[i], 'resistance_level'] = min(resistance_candidates)  # Strongest resistance
            
            # Check if current price is near these levels
            current_price = df['close'].iloc[i]
            if support_candidates and abs(current_price - max(support_candidates)) / max(support_candidates) < 0.003:
                df.loc[df.index[i], 'near_support'] = 1
            if resistance_candidates and abs(current_price - min(resistance_candidates)) / min(resistance_candidates) < 0.003:
                df.loc[df.index[i], 'near_resistance'] = 1
        
        return df
    
    def calculate_price_action_signals(self, df):
        """Use price action instead of complex indicators"""
        print("📊 Calculating price action signals...")
        
        df = self.find_real_support_resistance(df)
        
        # Price action features
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['high']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['low']
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)
        df['is_hammer'] = ((df['lower_wick'] > 2 * df['body_size']) & (df['body_size'] > 0.001)).astype(int)
        df['is_shooting_star'] = ((df['upper_wick'] > 2 * df['body_size']) & (df['body_size'] > 0.001)).astype(int)
        
        # Momentum
        df['price_change_5m'] = df['close'].pct_change(5)
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(int)
        
        print("✅ Price action signals calculated")
        return df
    
    def generate_trading_signals(self, df):
        """Generate simple but effective trading signals"""
        print("🎯 Generating trading signals...")
        
        df['buy_signal'] = 0
        df['sell_signal'] = 0
        
        for i in range(20, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # BUY SIGNALS (combine multiple conditions)
            buy_conditions = 0
            
            # 1. Price at support + bullish candle
            if current['near_support'] == 1 and current['close'] > current['open']:
                buy_conditions += 1
            
            # 2. Hammer pattern at support
            if current['near_support'] == 1 and current['is_hammer'] == 1:
                buy_conditions += 1
            
            # 3. Price bouncing from support with volume
            if (current['near_support'] == 1 and 
                prev['near_support'] == 1 and 
                current['close'] > prev['close'] and
                current['volume_spike'] == 1):
                buy_conditions += 1
            
            if buy_conditions >= 2:  # Need at least 2 confirmations
                df.loc[df.index[i], 'buy_signal'] = 1
            
            # SELL SIGNALS
            sell_conditions = 0
            
            # 1. Price at resistance + bearish candle
            if current['near_resistance'] == 1 and current['close'] < current['open']:
                sell_conditions += 1
            
            # 2. Shooting star at resistance
            if current['near_resistance'] == 1 and current['is_shooting_star'] == 1:
                sell_conditions += 1
            
            # 3. Price rejecting resistance with volume
            if (current['near_resistance'] == 1 and 
                prev['near_resistance'] == 1 and 
                current['close'] < prev['close'] and
                current['volume_spike'] == 1):
                sell_conditions += 1
            
            if sell_conditions >= 2:
                df.loc[df.index[i], 'sell_signal'] = 1
        
        print(f"✅ Generated {df['buy_signal'].sum()} buy and {df['sell_signal'].sum()} sell signals")
        return df

def run_new_approach():
    """Test the completely new approach"""
    print("🔄 COMPLETELY NEW APPROACH")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=1000)
    engine = NewApproachEngine()
    
    # Calculate new features
    data = engine.calculate_price_action_signals(data)
    data = engine.generate_trading_signals(data)
    
    # Backtest the new approach
    balance = 1000
    trades = []
    position = None
    
    print("\n🎯 Backtesting new approach...")
    
    for i in range(25, len(data) - 5):
        current = data.iloc[i]
        
        # EXIT LOGIC
        if position:
            # Simple exit: 1% profit or 0.5% loss or 10 minutes
            current_price = current['close']
            pnl_percent = (current_price - position['entry_price']) / position['entry_price'] if position['type'] == 'LONG' else (position['entry_price'] - current_price) / position['entry_price']
            
            if (pnl_percent >= 0.01 or pnl_percent <= -0.005 or 
                (data.index[i] - position['entry_time']).total_seconds() > 600):
                
                pnl_amount = balance * 0.02 * pnl_percent
                balance += pnl_amount
                
                trades.append({
                    'time': position['entry_time'],
                    'type': position['type'],
                    'entry': position['entry_price'],
                    'exit': current_price,
                    'pnl_percent': pnl_percent,
                    'balance': balance,
                    'hold_time': (data.index[i] - position['entry_time']).total_seconds() / 60
                })
                
                pnl_color = "🟢" if pnl_percent > 0 else "🔴"
                print(f"{pnl_color} EXIT {position['type']} - PnL: {pnl_percent*100:+.2f}%")
                position = None
        
        # ENTRY LOGIC
        if not position:
            if current['buy_signal'] == 1:
                position = {
                    'type': 'LONG',
                    'entry_price': current['close'],
                    'entry_time': data.index[i]
                }
                print(f"🟢 LONG at ${current['close']:.2f} (Support bounce)")
            
            elif current['sell_signal'] == 1:
                position = {
                    'type': 'SHORT', 
                    'entry_price': current['close'],
                    'entry_time': data.index[i]
                }
                print(f"🔴 SHORT at ${current['close']:.2f} (Resistance rejection)")
    
    # Results
    if trades:
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        win_rate = len(winning_trades) / len(trades_df) * 100
        total_return = (balance - 1000) / 1000 * 100
        
        print(f"\n💰 NEW APPROACH RESULTS:")
        print(f"Trades: {len(trades_df)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${balance:.2f}")
        
        return True, win_rate, total_return, len(trades_df)
    else:
        print("❌ No trades with new approach")
        return False, 0, 0, 0

def analyze_market_regimes():
    """Analyze which market conditions work best"""
    print("\n🌡️  ANALYZING MARKET REGIMES")
    print("=" * 50)
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=2000)
    
    # Calculate volatility
    data['volatility'] = data['close'].pct_change().rolling(20).std() * np.sqrt(252 * 24 * 60)  # Annualized
    
    # Define regimes
    high_vol = data['volatility'] > data['volatility'].quantile(0.7)
    low_vol = data['volatility'] < data['volatility'].quantile(0.3)
    medium_vol = ~high_vol & ~low_vol
    
    print(f"Market Regime Analysis:")
    print(f"High Volatility periods: {high_vol.sum()} candles")
    print(f"Medium Volatility periods: {medium_vol.sum()} candles") 
    print(f"Low Volatility periods: {low_vol.sum()} candles")
    
    # Analyze returns in each regime
    returns = data['close'].pct_change()
    print(f"\nAverage returns by regime:")
    print(f"High Vol: {returns[high_vol].mean()*100:.4f}% per candle")
    print(f"Medium Vol: {returns[medium_vol].mean()*100:.4f}% per candle")
    print(f"Low Vol: {returns[low_vol].mean()*100:.4f}% per candle")

def main():
    print("🚀 COMPLETELY NEW TRADING APPROACH")
    print("Fixing fundamental issues with trendline detection")
    print("=" * 60)
    
    # Analyze market first
    analyze_market_regimes()
    
    # Test new approach
    success, win_rate, total_return, num_trades = run_new_approach()
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if success and num_trades >= 10 and win_rate >= 55:
        print("• New approach works well - continue developing")
        print("• Focus on risk management and position sizing")
    elif success and num_trades >= 5:
        print("• New approach shows promise")
        print("• Need more data to validate")
        print("• Consider different timeframes or instruments")
    else:
        print("• Fundamental strategy issues detected")
        print("• Consider completely different approach:")
        print("  - Breakout strategies")
        print("  - Mean reversion") 
        print("  - Momentum strategies")
        print("  - Multi-timeframe analysis")

if __name__ == "__main__":
    main()
