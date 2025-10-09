#!/usr/bin/env python3
"""
PROFESSIONAL ORDER BLOCK TRADING BOT
Implements Order Block + FVG + Liquidity Strategy
Maximum 5 positions with TP/SL management
"""

import requests
import pandas as pd
import numpy as np
import time
import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class OrderBlockTradingBot:
    def __init__(self):
        self.symbol = "tBTCUSD"
        self.base_url = "https://api-pub.bitfinex.com/v2"
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.positions = []  # Max 5 positions
        self.max_positions = 5
        
        # Trading parameters
        self.tp_percent = 0.0080  # 0.80% Take Profit
        self.sl_percent = 0.0040  # 0.40% Stop Loss
        self.position_size = 0.08  # 8% per position
        
        # Strategy parameters
        self.lookback_candles = 100
        self.atr_period = 14
        self.volume_sma_period = 20
        
        # Performance tracking
        self.trade_log = []
        self.equity_curve = [self.initial_balance]
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.daily_trades_count = 0
        self.last_trade_date = None
        
        # Order Blocks tracking
        self.bullish_obs = []
        self.bearish_obs = []
        
        print("🚀 PROFESSIONAL ORDER BLOCK TRADING BOT")
        print("=" * 60)
        print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        print(f"🎯 TP/SL: {self.tp_percent*100}%/{self.sl_percent*100}%")
        print(f"📊 Position Size: {self.position_size*100}%")
        print(f"📈 Max Positions: {self.max_positions}")
        print(f"🤖 Strategy: Order Blocks + FVG + Liquidity")
        print("=" * 60)

    def reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades_count = 0
            self.last_trade_date = today
            print("🔄 Daily counters reset")

    def fetch_candles(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Bitfinex"""
        try:
            url = f"{self.base_url}/candles/trade:1m:{self.symbol}/hist?limit={limit}&sort=-1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not data:
                return None
                
            df = pd.DataFrame(data, columns=["timestamp", "open", "close", "high", "low", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            
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
            return float(data[6])  # Last price
        except Exception as e:
            print(f"❌ Error getting current price: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # ATR for volatility
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(self.atr_period).mean()
        
        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(self.volume_sma_period).mean()
        
        # Liquidity levels
        df['recent_high'] = df['high'].rolling(20).max()
        df['recent_low'] = df['low'].rolling(20).min()
        df['prev_high'] = df['high'].rolling(50).max()
        df['prev_low'] = df['low'].rolling(50).min()
        
        return df

    def detect_order_blocks(self, df: pd.DataFrame) -> Tuple[List, List]:
        """Detect bullish and bearish order blocks"""
        bullish_obs = []
        bearish_obs = []
        
        current_atr = df['atr'].iloc[-1] if not df['atr'].isna().iloc[-1] else 0
        
        for i in range(2, len(df)):
            # Bullish OB: Strong down candle followed by up move
            if (df['close'].iloc[i-1] < df['open'].iloc[i-1] and 
                (df['high'].iloc[i-1] - df['low'].iloc[i-1]) > current_atr * 0.8 and
                df['close'].iloc[i] > df['open'].iloc[i] and 
                df['close'].iloc[i] > df['high'].iloc[i-1]):
                
                bullish_obs.append({
                    'index': i-1,
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'timestamp': df['timestamp'].iloc[i-1],
                    'strength': abs(i-1 - len(df))
                })
            
            # Bearish OB: Strong up candle followed by down move
            if (df['close'].iloc[i-1] > df['open'].iloc[i-1] and 
                (df['high'].iloc[i-1] - df['low'].iloc[i-1]) > current_atr * 0.8 and
                df['close'].iloc[i] < df['open'].iloc[i] and 
                df['close'].iloc[i] < df['low'].iloc[i-1]):
                
                bearish_obs.append({
                    'index': i-1,
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'timestamp': df['timestamp'].iloc[i-1],
                    'strength': abs(i-1 - len(df))
                })
        
        # Keep only recent OBs
        bullish_obs = [ob for ob in bullish_obs if ob['index'] > len(df) - self.lookback_candles]
        bearish_obs = [ob for ob in bearish_obs if ob['index'] > len(df) - self.lookback_candles]
        
        return bullish_obs, bearish_obs

    def detect_fvg(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Detect Fair Value Gaps"""
        current_atr = df['atr'].iloc[-1] if not df['atr'].isna().iloc[-1] else 0
        fvg_size = current_atr * 0.5
        
        bullish_fvg = (df['low'].iloc[-1] > df['high'].iloc[-2] + fvg_size and 
                      df['low'].iloc[-1] > df['high'].iloc[-3] + fvg_size)
        
        bearish_fvg = (df['high'].iloc[-1] < df['low'].iloc[-2] - fvg_size and 
                      df['high'].iloc[-1] < df['low'].iloc[-3] - fvg_size)
        
        return bullish_fvg, bearish_fvg

    def check_market_structure(self, df: pd.DataFrame) -> Dict:
        """Check market structure conditions"""
        if len(df) < 3:
            return {'higher_high': False, 'lower_low': False, 'higher_low': False, 'lower_high': False}
        
        higher_high = (df['high'].iloc[-1] > df['high'].iloc[-2] and 
                      df['high'].iloc[-2] > df['high'].iloc[-3])
        
        lower_low = (df['low'].iloc[-1] < df['low'].iloc[-2] and 
                    df['low'].iloc[-2] < df['low'].iloc[-3])
        
        higher_low = (df['low'].iloc[-1] > df['low'].iloc[-2] and 
                     df['low'].iloc[-2] > df['low'].iloc[-3])
        
        lower_high = (df['high'].iloc[-1] < df['high'].iloc[-2] and 
                     df['high'].iloc[-2] < df['high'].iloc[-3])
        
        return {
            'higher_high': higher_high,
            'lower_low': lower_low,
            'higher_low': higher_low,
            'lower_high': lower_high
        }

    def check_breaker_blocks(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Check for breaker blocks"""
        if len(df) < 2:
            return False, False
        
        current_volume = df['volume'].iloc[-1]
        volume_sma = df['volume_sma'].iloc[-1] if not df['volume_sma'].isna().iloc[-1] else current_volume
        
        bullish_breaker = (df['close'].iloc[-1] > df['open'].iloc[-1] and 
                          df['close'].iloc[-1] > df['high'].iloc[-2] and 
                          current_volume > volume_sma * 1.5)
        
        bearish_breaker = (df['close'].iloc[-1] < df['open'].iloc[-1] and 
                          df['close'].iloc[-1] < df['low'].iloc[-2] and 
                          current_volume > volume_sma * 1.5)
        
        return bullish_breaker, bearish_breaker

    def generate_signals(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Generate trading signals based on Order Block strategy"""
        # Update Order Blocks
        self.bullish_obs, self.bearish_obs = self.detect_order_blocks(df)
        
        # Get current market data
        current_atr = df['atr'].iloc[-1] if not df['atr'].isna().iloc[-1] else 0
        recent_high = df['recent_high'].iloc[-1] if not df['recent_high'].isna().iloc[-1] else current_price
        recent_low = df['recent_low'].iloc[-1] if not df['recent_low'].isna().iloc[-1] else current_price
        
        # Detect FVG and Breaker Blocks
        bullish_fvg, bearish_fvg = self.detect_fvg(df)
        bullish_breaker, bearish_breaker = self.check_breaker_blocks(df)
        market_structure = self.check_market_structure(df)
        
        signal = None
        signal_text = ""
        
        # BUY SIGNALS
        # Signal 1: OB/FVG + Higher Low
        if ((len(self.bullish_obs) > 0 or bullish_fvg) and market_structure['higher_low']):
            signal = 'BUY'
            signal_text = "OB/FVG + Higher Low"
        
        # Signal 2: Breaker Block above recent low
        elif bullish_breaker and current_price > recent_low:
            signal = 'BUY'
            signal_text = "Breaker Block"
        
        # SELL SIGNALS  
        # Signal 1: OB/FVG + Lower High
        elif ((len(self.bearish_obs) > 0 or bearish_fvg) and market_structure['lower_high']):
            signal = 'SELL'
            signal_text = "OB/FVG + Lower High"
        
        # Signal 2: Breaker Block below recent high
        elif bearish_breaker and current_price < recent_high:
            signal = 'SELL'
            signal_text = "Breaker Block"
        
        if signal:
            return {
                'signal': signal,
                'reason': signal_text,
                'bullish_obs': len(self.bullish_obs),
                'bearish_obs': len(self.bearish_obs),
                'bullish_fvg': bullish_fvg,
                'bearish_fvg': bearish_fvg,
                'market_structure': market_structure
            }
        
        return None

    def should_enter_trade(self, signal: Dict) -> Tuple[bool, str]:
        """Check if we should enter a new trade"""
        if len(self.positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        
        # Check if we already have too many positions in same direction
        same_direction_positions = len([p for p in self.positions if p['type'] == signal['signal']])
        if same_direction_positions >= 3:  # Max 3 positions in same direction
            return False, f"Too many {signal['signal']} positions"
        
        # Minimum balance check
        if self.balance < self.initial_balance * 0.8:
            return False, "Balance below 80% of initial"
        
        return True, "OK"

    def execute_trade(self, signal: Dict, current_price: float):
        """Execute a new trade"""
        validation, reason = self.should_enter_trade(signal)
        if not validation:
            print(f"⏸️  Trade skipped: {reason}")
            return False

        # Calculate position size
        trade_amount = self.balance * self.position_size / current_price
        
        position = {
            'id': len(self.positions) + 1,
            'type': signal['signal'],
            'entry_price': current_price,
            'size': trade_amount,
            'entry_time': datetime.datetime.now(),
            'tp_price': current_price * (1 + self.tp_percent) if signal['signal'] == 'BUY' else current_price * (1 - self.tp_percent),
            'sl_price': current_price * (1 - self.sl_percent) if signal['signal'] == 'BUY' else current_price * (1 + self.sl_percent),
            'reason': signal['reason'],
            'unrealized_pnl': 0.0,
            'pnl_percent': 0.0
        }
        
        self.positions.append(position)
        self.daily_trades_count += 1
        
        print(f"🎯 OPEN {signal['signal']} at ${current_price:.2f}")
        print(f"   Reason: {signal['reason']}")
        print(f"   TP: ${position['tp_price']:.2f} | SL: ${position['sl_price']:.2f}")
        print(f"   Size: {trade_amount:.6f} BTC")
        print(f"   Active Positions: {len(self.positions)}/{self.max_positions}")
        
        return True

    def check_exit_conditions(self, current_price: float):
        """Check TP/SL for all positions"""
        positions_to_remove = []
        
        for i, position in enumerate(self.positions):
            entry_price = position['entry_price']
            position_type = position['type']
            
            if position_type == 'BUY':
                # Take Profit
                if current_price >= position['tp_price']:
                    self.close_position(i, current_price, "TP")
                    positions_to_remove.append(i)
                # Stop Loss
                elif current_price <= position['sl_price']:
                    self.close_position(i, current_price, "SL")
                    positions_to_remove.append(i)
            else:  # SELL
                # Take Profit
                if current_price <= position['tp_price']:
                    self.close_position(i, current_price, "TP")
                    positions_to_remove.append(i)
                # Stop Loss
                elif current_price >= position['sl_price']:
                    self.close_position(i, current_price, "SL")
                    positions_to_remove.append(i)
        
        # Remove closed positions (in reverse to avoid index issues)
        for i in sorted(positions_to_remove, reverse=True):
            if i < len(self.positions):
                self.positions.pop(i)

    def close_position(self, position_index: int, current_price: float, reason: str):
        """Close a specific position"""
        if position_index >= len(self.positions):
            return
            
        position = self.positions[position_index]
        entry_price = position['entry_price']
        position_type = position['type']
        size = position['size']
        
        # Calculate PnL
        if position_type == 'BUY':
            pnl = (current_price - entry_price) * size
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - current_price) * size
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        # Update balance and statistics
        self.balance += pnl
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update equity curve
        self.equity_curve.append(self.balance)
        if self.balance > self.peak_equity:
            self.peak_equity = self.balance
        
        # Calculate drawdown
        current_drawdown = (self.peak_equity - self.balance) / self.peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Log trade
        trade_record = {
            'id': position['id'],
            'type': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': position['reason'],
            'exit_reason': reason,
            'entry_time': position['entry_time'],
            'exit_time': datetime.datetime.now(),
            'duration': (datetime.datetime.now() - position['entry_time']).total_seconds() / 60  # minutes
        }
        self.trade_log.append(trade_record)
        
        emoji = "🎯" if reason == "TP" else "🛑"
        print(f"{emoji} CLOSE {position_type} #{position['id']}: ${pnl:+.2f} ({pnl_pct:+.2f}%)")

    def update_position_pnl(self, current_price: float):
        """Update unrealized PnL for all positions"""
        for position in self.positions:
            entry_price = position['entry_price']
            position_type = position['type']
            size = position['size']
            
            if position_type == 'BUY':
                pnl_percent = (current_price - entry_price) / entry_price * 100
            else:
                pnl_percent = (entry_price - current_price) / entry_price * 100
            
            unrealized_pnl = pnl_percent / 100 * self.balance * self.position_size
            
            position['unrealized_pnl'] = unrealized_pnl
            position['pnl_percent'] = pnl_percent

    def show_comprehensive_dashboard(self, df: pd.DataFrame, current_price: float, signal: Optional[Dict]):
        """Show detailed real-time dashboard"""
        self.update_position_pnl(current_price)
        
        # Calculate current equity with unrealized PnL
        current_equity = self.balance
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions)
        current_equity += total_unrealized_pnl
        
        # Performance metrics
        total_return = (current_equity - self.initial_balance) / self.initial_balance * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
        
        # Additional metrics
        avg_win = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] > 0]) if self.winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] < 0]) if self.losing_trades > 0 else 0
        profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if self.losing_trades > 0 and avg_loss != 0 else float('inf')
        
        print("\n" + "="*100)
        print("🚀 ORDER BLOCK TRADING BOT - REAL-TIME DASHBOARD")
        print("="*100)
        print(f"🕒 Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Price: ${current_price:.2f}")
        print("-"*100)
        
        # Balance & Equity Section
        print("💵 BALANCE & EQUITY:")
        print(f"   Balance: ${self.balance:.2f}")
        print(f"   Equity: ${current_equity:.2f}")
        print(f"   Unrealized P&L: ${total_unrealized_pnl:+.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Peak Equity: ${self.peak_equity:.2f}")
        print(f"   Current Drawdown: {current_drawdown:.2f}%")
        print(f"   Max Drawdown: {self.max_drawdown:.2f}%")
        
        # Trading Statistics
        print("\n📊 TRADING STATISTICS:")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Winning Trades: {self.winning_trades}")
        print(f"   Losing Trades: {self.losing_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total P&L: ${self.total_pnl:+.2f}")
        print(f"   Average Win: ${avg_win:+.2f}")
        print(f"   Average Loss: ${avg_loss:+.2f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Daily Trades: {self.daily_trades_count}")
        
        # Current Positions
        print(f"\n📈 ACTIVE POSITIONS: {len(self.positions)}/{self.max_positions}")
        if self.positions:
            for i, pos in enumerate(self.positions):
                pnl_color = "🟢" if pos['pnl_percent'] > 0 else "🔴"
                print(f"   #{pos['id']} {pos['type']} | Entry: ${pos['entry_price']:.2f}")
                print(f"     P&L: {pnl_color} {pos['pnl_percent']:+.2f}% (${pos['unrealized_pnl']:+.2f})")
                print(f"     TP: ${pos['tp_price']:.2f} | SL: ${pos['sl_price']:.2f}")
                print(f"     Reason: {pos['reason']}")
        else:
            print("   No active positions")
        
        # Order Blocks Info
        print(f"\n🎯 ORDER BLOCKS DETECTED:")
        print(f"   Bullish OBs: {len(self.bullish_obs)}")
        print(f"   Bearish OBs: {len(self.bearish_obs)}")
        
        # Market Structure
        if signal and 'market_structure' in signal:
            ms = signal['market_structure']
            print(f"\n📊 MARKET STRUCTURE:")
            print(f"   Higher High: {'✅' if ms['higher_high'] else '❌'}")
            print(f"   Lower Low: {'✅' if ms['lower_low'] else '❌'}")
            print(f"   Higher Low: {'✅' if ms['higher_low'] else '❌'}")
            print(f"   Lower High: {'✅' if ms['lower_high'] else '❌'}")
        
        # Trading Signal
        print(f"\n🎯 TRADING SIGNAL:")
        if signal:
            signal_color = "🟢" if signal['signal'] == 'BUY' else "🔴"
            print(f"   Signal: {signal_color} {signal['signal']}")
            print(f"   Reason: {signal['reason']}")
            print(f"   Bullish FVG: {'✅' if signal['bullish_fvg'] else '❌'}")
            print(f"   Bearish FVG: {'✅' if signal['bearish_fvg'] else '❌'}")
        else:
            print("   No signal detected")
            print("   Waiting for Order Block setup...")
        
        print("="*100)

    def run_bot(self):
        """Main trading loop"""
        print("\n🤖 ORDER BLOCK BOT STARTED")
        print("🎯 Trading Strategy: Order Blocks + FVG + Liquidity")
        print(f"📈 Max Positions: {self.max_positions} | TP: {self.tp_percent*100}% | SL: {self.sl_percent*100}%")
        print("⏰ Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                self.reset_daily_counters()
                
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
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Generate signals
                signal = self.generate_signals(df, current_price)
                
                # Check exit conditions first
                self.check_exit_conditions(current_price)
                
                # Show comprehensive dashboard
                self.show_comprehensive_dashboard(df, current_price, signal)
                
                # Execute new trade if signal detected
                if signal and len(self.positions) < self.max_positions:
                    self.execute_trade(signal, current_price)
                
                print(f"⏰ Next check in 30 seconds... (Iteration: {iteration})\n")
                time.sleep(30)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Bot stopped after {iteration} iterations")
            self.show_final_summary()

    def show_final_summary(self):
        """Display final performance summary"""
        final_equity = self.equity_curve[-1] if self.equity_curve else self.balance
        total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print("\n🎯 FINAL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        print(f"💰 Final Balance: ${final_equity:.2f}")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"📊 Total Trades: {self.total_trades}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"🔥 Winning Trades: {self.winning_trades}")
        print(f"💀 Losing Trades: {self.losing_trades}")
        print(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")
        print(f"💵 Total P&L: ${self.total_pnl:+.2f}")
        
        if self.total_trades > 0:
            avg_win = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] > 0])
            avg_loss = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] < 0])
            profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if self.losing_trades > 0 and avg_loss != 0 else float('inf')
            
            print(f"📈 Average Win: ${avg_win:+.2f}")
            print(f"📉 Average Loss: ${avg_loss:+.2f}")
            print(f"💰 Profit Factor: {profit_factor:.2f}")
            
        print("="*60)

def main():
    bot = OrderBlockTradingBot()
    bot.run_bot()

if __name__ == "__main__":
    main()
