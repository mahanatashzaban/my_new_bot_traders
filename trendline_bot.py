#!/usr/bin/env python3
"""
BITCOIN TRENDLINE TRADING BOT
Strategy: Trade when price hits trendlines connecting peaks and valleys
Timeframe: 1 minute candles
Exchange: Bitfinex
"""

import requests
import pandas as pd
import numpy as np
import time
import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TrendlineTradingBot:
    def __init__(self):
        self.symbol = "tBTCUSD"
        self.base_url = "https://api-pub.bitfinex.com/v2"
        self.timeframe = "1m"
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.position = None
        
        # Trading parameters
        self.tp_percent = 0.0012  # 0.12%
        self.sl_percent = 0.0007  # 0.07%
        self.position_size = 0.1  # 10% of balance per trade
        
        # Trendline parameters
        self.lookback_candles = 500
        self.min_trendline_length = 20  # Minimum candles for valid trendline
        self.trendline_buffer = 0.0005  # 0.05% buffer for trendline hits
        
        # Performance tracking
        self.trade_log = []
        self.equity_curve = [self.initial_balance]
        self.trendlines = []
        self.last_trendline_update = None
        
        print("🚀 BITCOIN TRENDLINE TRADING BOT INITIALIZED")
        print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        print(f"📈 Timeframe: {self.timeframe}")
        print(f"🎯 TP/SL: {self.tp_percent*100}%/{self.sl_percent*100}%")
        print(f"📊 Lookback: {self.lookback_candles} candles")
        print("=" * 60)

    def fetch_candles(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Bitfinex"""
        try:
            url = f"{self.base_url}/candles/trade:{self.timeframe}:{self.symbol}/hist?limit={limit}&sort=-1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not data:
                return None
                
            df = pd.DataFrame(data, columns=["timestamp", "open", "close", "high", "low", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Convert to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
                
            return df
            
        except Exception as e:
            print(f"❌ Error fetching candles: {e}")
            return None

    def get_current_price(self) -> Optional[float]:
        """Get current BTC price from Bitfinex"""
        try:
            url = f"{self.base_url}/ticker/{self.symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            return float(data[6])  # Last price is at index 6
        except Exception as e:
            print(f"❌ Error getting current price: {e}")
            return None

    def find_peaks_valleys(self, df: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """Find peaks and valleys in price data"""
        highs = df['high'].values
        lows = df['low'].values
        
        peaks = []
        valleys = []
        
        for i in range(window, len(df) - window):
            # Check for peak
            if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
               all(highs[i] > highs[i+j] for j in range(1, window+1)):
                peaks.append(i)
            
            # Check for valley
            if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
               all(lows[i] < lows[i+j] for j in range(1, window+1)):
                valleys.append(i)
                
        return peaks, valleys

    def create_trendlines(self, df: pd.DataFrame) -> List[Dict]:
        """Create trendlines connecting significant peaks and valleys"""
        peaks, valleys = self.find_peaks_valleys(df, window=7)
        trendlines = []
        
        print(f"📊 Found {len(peaks)} peaks and {len(valleys)} valleys")
        
        # Connect peaks (resistance lines - descending)
        for i in range(len(peaks)):
            for j in range(i+1, min(i+6, len(peaks))):  # Connect to next 5 peaks max
                idx1, idx2 = peaks[i], peaks[j]
                if idx2 - idx1 < self.min_trendline_length:
                    continue
                    
                price1, price2 = df['high'].iloc[idx1], df['high'].iloc[idx2]
                slope = (price2 - price1) / (idx2 - idx1)
                
                # Only consider descending resistance lines
                if slope < 0:
                    trendlines.append({
                        'type': 'RESISTANCE',
                        'start_idx': idx1,
                        'end_idx': idx2,
                        'start_price': price1,
                        'end_price': price2,
                        'slope': slope,
                        'intercept': price1 - slope * idx1,
                        'strength': abs(idx2 - idx1)  # Longer lines are stronger
                    })
        
        # Connect valleys (support lines - ascending)
        for i in range(len(valleys)):
            for j in range(i+1, min(i+6, len(valleys))):  # Connect to next 5 valleys max
                idx1, idx2 = valleys[i], valleys[j]
                if idx2 - idx1 < self.min_trendline_length:
                    continue
                    
                price1, price2 = df['low'].iloc[idx1], df['low'].iloc[idx2]
                slope = (price2 - price1) / (idx2 - idx1)
                
                # Only consider ascending support lines
                if slope > 0:
                    trendlines.append({
                        'type': 'SUPPORT',
                        'start_idx': idx1,
                        'end_idx': idx2,
                        'start_price': price1,
                        'end_price': price2,
                        'slope': slope,
                        'intercept': price1 - slope * idx1,
                        'strength': abs(idx2 - idx1)  # Longer lines are stronger
                    })
        
        # Sort by strength (longer lines first)
        trendlines.sort(key=lambda x: x['strength'], reverse=True)
        
        print(f"📈 Generated {len([t for t in trendlines if t['type'] == 'SUPPORT'])} support lines")
        print(f"📉 Generated {len([t for t in trendlines if t['type'] == 'RESISTANCE'])} resistance lines")
        
        return trendlines[:20]  # Keep top 20 strongest trendlines

    def calculate_trendline_price(self, trendline: Dict, current_index: int) -> float:
        """Calculate trendline price at current index"""
        return trendline['slope'] * current_index + trendline['intercept']

    def check_trendline_break(self, df: pd.DataFrame, current_price: float, 
                            current_index: int) -> Optional[Dict]:
        """Check if price breaks any trendline"""
        if not self.trendlines:
            return None
            
        for trendline in self.trendlines:
            trendline_price = self.calculate_trendline_price(trendline, current_index)
            price_diff_pct = abs(current_price - trendline_price) / trendline_price
            
            # Check if price is within buffer of trendline
            if price_diff_pct <= self.trendline_buffer:
                # Determine break direction
                if trendline['type'] == 'RESISTANCE':
                    # Price breaking resistance from below → BUY
                    if current_price > trendline_price:
                        return {
                            'signal': 'BUY',
                            'trendline_type': trendline['type'],
                            'trendline_price': trendline_price,
                            'current_price': current_price,
                            'strength': trendline['strength'],
                            'price_diff_pct': price_diff_pct
                        }
                else:  # SUPPORT
                    # Price breaking support from above → SELL
                    if current_price < trendline_price:
                        return {
                            'signal': 'SELL',
                            'trendline_type': trendline['type'],
                            'trendline_price': trendline_price,
                            'current_price': current_price,
                            'strength': trendline['strength'],
                            'price_diff_pct': price_diff_pct
                        }
        
        return None

    def should_update_trendlines(self) -> bool:
        """Check if trendlines should be updated"""
        if not self.last_trendline_update:
            return True
            
        time_since_update = datetime.datetime.now() - self.last_trendline_update
        return time_since_update.total_seconds() > 300  # Update every 5 minutes

    def execute_trade(self, signal: Dict, current_price: float):
        """Execute trade based on trendline break signal"""
        if self.position:
            return False
            
        trade_amount = self.balance * self.position_size / current_price
        
        self.position = {
            'type': signal['signal'],
            'entry_price': current_price,
            'size': trade_amount,
            'entry_time': datetime.datetime.now(),
            'trendline_type': signal['trendline_type'],
            'trendline_strength': signal['strength'],
            'tp_price': current_price * (1 + self.tp_percent) if signal['signal'] == 'BUY' else current_price * (1 - self.tp_percent),
            'sl_price': current_price * (1 - self.sl_percent) if signal['signal'] == 'BUY' else current_price * (1 + self.sl_percent)
        }
        
        print(f"🎯 OPEN {signal['signal']} at ${current_price:.2f}")
        print(f"   Trendline: {signal['trendline_type']} | Strength: {signal['strength']}")
        print(f"   TP: ${self.position['tp_price']:.2f} | SL: ${self.position['sl_price']:.2f}")
        print(f"   Size: {trade_amount:.6f} BTC")
        
        return True

    def check_exit_conditions(self, current_price: float) -> bool:
        """Check if position should be closed via TP/SL"""
        if not self.position:
            return False
            
        position_type = self.position['type']
        entry_price = self.position['entry_price']
        
        if position_type == 'BUY':
            # Take Profit
            if current_price >= self.position['tp_price']:
                self.close_position(current_price, "TP")
                return True
            # Stop Loss
            elif current_price <= self.position['sl_price']:
                self.close_position(current_price, "SL")
                return True
        else:  # SELL
            # Take Profit
            if current_price <= self.position['tp_price']:
                self.close_position(current_price, "TP")
                return True
            # Stop Loss
            elif current_price >= self.position['sl_price']:
                self.close_position(current_price, "SL")
                return True
                
        return False

    def close_position(self, current_price: float, reason: str):
        """Close current position"""
        if not self.position:
            return
            
        entry_price = self.position['entry_price']
        position_type = self.position['type']
        size = self.position['size']
        
        if position_type == 'BUY':
            pnl = (current_price - entry_price) * size
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - current_price) * size
            pnl_pct = (entry_price - current_price) / entry_price * 100
            
        self.balance += pnl
        
        # Log trade
        trade_record = {
            'entry_time': self.position['entry_time'],
            'exit_time': datetime.datetime.now(),
            'type': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'trendline_type': self.position['trendline_type'],
            'trendline_strength': self.position['trendline_strength']
        }
        self.trade_log.append(trade_record)
        self.equity_curve.append(self.balance)
        
        emoji = "🎯" if reason == "TP" else "🛑"
        print(f"{emoji} CLOSE {position_type}: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        self.position = None

    def calculate_smooth_price(self, df: pd.DataFrame, method: str = 'ema') -> pd.Series:
        """Calculate smoothed price to reduce noise"""
        if method == 'ema':
            return df['close'].ewm(span=5).mean()
        elif method == 'sma':
            return df['close'].rolling(3).mean()
        else:
            return df['close']

    def show_dashboard(self, df: pd.DataFrame, current_price: float, signal: Optional[Dict]):
        """Show real-time trading dashboard"""
        current_equity = self.balance
        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']
            size = self.position['size']
            
            if position_type == 'BUY':
                unrealized_pnl = (current_price - entry_price) * size
            else:
                unrealized_pnl = (entry_price - current_price) * size
                
            current_equity += unrealized_pnl
        else:
            unrealized_pnl = 0
            
        total_return = (current_equity - self.initial_balance) / self.initial_balance * 100
        
        # Calculate performance metrics
        total_trades = len(self.trade_log)
        winning_trades = len([t for t in self.trade_log if t['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        print("\n" + "="*80)
        print("📊 TRENDLINE TRADING BOT - REAL-TIME DASHBOARD")
        print("="*80)
        print(f"🕒 Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Price: ${current_price:.2f} | Equity: ${current_equity:.2f}")
        print(f"📈 Return: {total_return:+.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%")
        print("-"*80)
        
        # Position info
        print("📊 POSITION:")
        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']
            pnl_color = "🟢" if unrealized_pnl > 0 else "🔴"
            
            print(f"   {position_type} | Entry: ${entry_price:.2f}")
            print(f"   P&L: {pnl_color} ${unrealized_pnl:+.2f}")
            print(f"   TP: ${self.position['tp_price']:.2f} | SL: ${self.position['sl_price']:.2f}")
        else:
            print("   No active position")
            
        # Trendline info
        print(f"\n📈 TRENDLINES:")
        support_lines = len([t for t in self.trendlines if t['type'] == 'SUPPORT'])
        resistance_lines = len([t for t in self.trendlines if t['type'] == 'RESISTANCE'])
        print(f"   Active: {support_lines} support, {resistance_lines} resistance")
        
        if signal:
            print(f"\n🎯 SIGNAL DETECTED:")
            print(f"   {signal['signal']} | {signal['trendline_type']} break")
            print(f"   Strength: {signal['strength']} | Diff: {signal['price_diff_pct']*100:.3f}%")
        else:
            print(f"\n⏸️  No trendline breaks detected")
            
        print("="*80)

    def run_bot(self):
        """Main trading loop"""
        print("\n🤖 TRENDLINE BOT STARTED")
        print("🎯 Trading Bitcoin on 1-minute trendline breaks")
        print("⏰ Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                # Fetch market data
                df = self.fetch_candles(self.lookback_candles)
                if df is None:
                    print("❌ Failed to fetch market data")
                    time.sleep(60)
                    continue
                    
                current_price = self.get_current_price()
                if current_price is None:
                    print("❌ Failed to get current price")
                    time.sleep(60)
                    continue
                
                # Update trendlines periodically
                if self.should_update_trendlines():
                    print("🔄 Updating trendlines...")
                    self.trendlines = self.create_trendlines(df)
                    self.last_trendline_update = datetime.datetime.now()
                
                # Use smoothed price for trendline detection
                smoothed_price = self.calculate_smooth_price(df, 'ema').iloc[-1]
                current_index = len(df) - 1
                
                # Check for trendline breaks
                signal = self.check_trendline_break(df, smoothed_price, current_index)
                
                # Check exit conditions first
                if self.check_exit_conditions(current_price):
                    time.sleep(30)
                    continue
                
                # Show dashboard
                self.show_dashboard(df, current_price, signal)
                
                # Execute new trade if signal detected
                if signal and not self.position:
                    self.execute_trade(signal, current_price)
                
                print(f"⏰ Next check in 30 seconds... (Iteration: {iteration})\n")
                time.sleep(30)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Bot stopped after {iteration} iterations")
            self.show_final_summary()

    def show_final_summary(self):
        """Display final performance summary"""
        if self.trade_log:
            total_pnl = sum(trade['pnl'] for trade in self.trade_log)
            total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
            winning_trades = len([t for t in self.trade_log if t['pnl'] > 0])
            total_trades = len(self.trade_log)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            print("\n🎯 FINAL PERFORMANCE SUMMARY")
            print("="*50)
            print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
            print(f"💰 Final Balance: ${self.balance:.2f}")
            print(f"📈 Total Return: {total_return:+.2f}%")
            print(f"💵 Total P&L: ${total_pnl:+.2f}")
            print(f"📊 Total Trades: {total_trades}")
            print(f"🎯 Win Rate: {win_rate:.1f}%")
            print(f"🔥 Winning Trades: {winning_trades}")
            print(f"💀 Losing Trades: {total_trades - winning_trades}")
            
            if total_trades > 0:
                avg_win = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] > 0])
                avg_loss = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] < 0])
                print(f"📈 Average Win: ${avg_win:+.2f}")
                print(f"📉 Average Loss: ${avg_loss:+.2f}")
            
            print("="*50)

def main():
    bot = TrendlineTradingBot()
    bot.run_bot()

if __name__ == "__main__":
    main()
