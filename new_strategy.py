#!/usr/bin/env python3
"""
HIGH WIN RATE BITCOIN TRADING BOT (70%+ Target)
Multi-Timeframe Analysis + Advanced Risk Management
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

class HighWinRateBot:
    def __init__(self):
        self.symbol = "tBTCUSD"
        self.base_url = "https://api-pub.bitfinex.com/v2"
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.position = None
        
        # Advanced Risk Management for 70% Win Rate
        self.tp_percent = 0.0030  # 0.30% - Smaller TP for higher win rate
        self.sl_percent = 0.0015  # 0.15% - Tight SL
        self.position_size = 0.08  # 8% of balance
        self.max_daily_trades = 15
        self.daily_trades_count = 0
        self.last_trade_date = None
        
        # Multi-timeframe analysis
        self.timeframes = ['5m', '15m', '1h']  # Multiple timeframe confirmation
        
        # Performance tracking - INITIALIZE ALL ATTRIBUTES
        self.trade_log = []
        self.equity_curve = [self.initial_balance]
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_pnl = 0.0  # FIXED: Added missing attribute
        self.last_trade_time = None  # FIXED: Added missing attribute
        
        # Strategy parameters
        self.min_confidence = 0.75
        self.volume_threshold = 1.2
        self.trend_strength_threshold = 0.5
        
        print("🚀 HIGH WIN RATE BITCOIN BOT (70%+ Target)")
        print("=" * 60)
        print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        print(f"🎯 TP/SL: {self.tp_percent*100}%/{self.sl_percent*100}%")
        print(f"📊 Position Size: {self.position_size*100}%")
        print(f"📈 Multi-Timeframe: 5m, 15m, 1h")
        print(f"🎯 Target Win Rate: 70%+")
        print("=" * 60)

    def reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades_count = 0
            self.last_trade_date = today
            print("🔄 Daily counters reset")

    def fetch_candles(self, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for specified timeframe"""
        try:
            url = f"{self.base_url}/candles/trade:{timeframe}:{self.symbol}/hist?limit={limit}&sort=-1"
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
            print(f"❌ Error fetching {timeframe} data: {e}")
            return None

    def get_current_price(self) -> Optional[float]:
        """Get current BTC price"""
        try:
            url = f"{self.base_url}/ticker/{self.symbol}"
            response = requests.get(url, timeout=10)
            data = response.json()
            return float(data[6])
        except Exception as e:
            print(f"❌ Error getting current price: {e}")
            return None

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = df.copy()
        
        # Trend indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI with different periods
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Support/Resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        
        # Volatility
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        
        return df

    def analyze_multi_timeframe(self) -> Dict:
        """Multi-timeframe analysis for high-confidence signals"""
        signals = {}
        confidences = {}
        
        for timeframe in self.timeframes:
            df = self.fetch_candles(timeframe, 100)
            if df is None or len(df) < 50:
                continue
                
            df = self.calculate_advanced_indicators(df)
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Bullish conditions
            bullish_score = 0
            bearish_score = 0
            
            # Trend alignment
            if current['sma_20'] > current['sma_50']:
                bullish_score += 2
            else:
                bearish_score += 2
                
            # MACD momentum
            if current['macd'] > current['macd_signal'] and current['macd_histogram'] > prev['macd_histogram']:
                bullish_score += 2
            elif current['macd'] < current['macd_signal'] and current['macd_histogram'] < prev['macd_histogram']:
                bearish_score += 2
                
            # RSI conditions
            if 40 < current['rsi_14'] < 60:  # Neutral RSI for continuation
                if current['close'] > current['sma_20']:
                    bullish_score += 1
                else:
                    bearish_score += 1
                    
            # Bollinger Bands position
            if current['bb_position'] < 0.3:
                bullish_score += 1
            elif current['bb_position'] > 0.7:
                bearish_score += 1
                
            # Volume confirmation
            if current['volume_ratio'] > self.volume_threshold:
                if current['close'] > current['open']:
                    bullish_score += 1
                else:
                    bearish_score += 1
            
            # Determine signal for this timeframe
            total_score = bullish_score + bearish_score
            if total_score > 0:
                if bullish_score > bearish_score * 1.5:  # Strong bullish bias
                    signals[timeframe] = 'BUY'
                    confidences[timeframe] = min(bullish_score / total_score, 1.0)
                elif bearish_score > bullish_score * 1.5:  # Strong bearish bias
                    signals[timeframe] = 'SELL'
                    confidences[timeframe] = min(bearish_score / total_score, 1.0)
                else:
                    signals[timeframe] = 'HOLD'
                    confidences[timeframe] = 0.5
            else:
                signals[timeframe] = 'HOLD'
                confidences[timeframe] = 0.5
                
        return {'signals': signals, 'confidences': confidences}

    def generate_high_confidence_signal(self) -> Optional[Dict]:
        """Generate high-confidence trading signal (70%+ win rate target)"""
        mtf_analysis = self.analyze_multi_timeframe()
        signals = mtf_analysis['signals']
        confidences = mtf_analysis['confidences']
        
        print("📊 Multi-Timeframe Analysis:")
        for tf in self.timeframes:
            signal = signals.get(tf, 'HOLD')
            conf = confidences.get(tf, 0)
            print(f"   {tf}: {signal} (Conf: {conf:.2f})")
        
        # Require consensus across timeframes
        buy_signals = [s for s in signals.values() if s == 'BUY']
        sell_signals = [s for s in signals.values() if s == 'SELL']
        
        total_signals = len(signals)
        buy_ratio = len(buy_signals) / total_signals if total_signals > 0 else 0
        sell_ratio = len(sell_signals) / total_signals if total_signals > 0 else 0
        
        # Calculate average confidence
        valid_confidences = [c for c in confidences.values() if c > 0]
        avg_confidence = np.mean(valid_confidences) if valid_confidences else 0
        
        # High-confidence BUY signal (all timeframes aligned)
        if buy_ratio >= 0.67 and avg_confidence >= self.min_confidence:  # At least 2/3 timeframes
            return {
                'signal': 'BUY',
                'confidence': avg_confidence,
                'timeframe_alignment': f"{len(buy_signals)}/{total_signals}",
                'reason': f"Strong bullish alignment across {len(buy_signals)} timeframes"
            }
        
        # High-confidence SELL signal (all timeframes aligned)
        elif sell_ratio >= 0.67 and avg_confidence >= self.min_confidence:
            return {
                'signal': 'SELL', 
                'confidence': avg_confidence,
                'timeframe_alignment': f"{len(sell_signals)}/{total_signals}",
                'reason': f"Strong bearish alignment across {len(sell_signals)} timeframes"
            }
        
        return None

    def adaptive_position_sizing(self, signal_confidence: float) -> float:
        """Adjust position size based on confidence and recent performance"""
        base_size = self.position_size
        
        # Increase size with higher confidence
        if signal_confidence > 0.85:
            base_size *= 1.2
        elif signal_confidence > 0.75:
            base_size *= 1.1
            
        # Reduce size after consecutive losses
        if self.consecutive_losses >= 2:
            base_size *= 0.7
        elif self.consecutive_losses >= 3:
            base_size *= 0.5
            
        # Increase size after consecutive wins
        if self.consecutive_wins >= 3:
            base_size *= 1.1
            
        return min(base_size, 0.15)  # Cap at 15%

    def should_enter_trade(self, signal: Dict) -> Tuple[bool, str]:
        """Comprehensive trade validation"""
        if self.position:
            return False, "Already in position"
            
        if self.daily_trades_count >= self.max_daily_trades:
            return False, "Daily trade limit reached"
            
        if signal['confidence'] < self.min_confidence:
            return False, f"Low confidence: {signal['confidence']:.2f}"
            
        # Check if we're in a strong drawdown
        current_drawdown = (self.peak_equity - self.balance) / self.peak_equity * 100
        if current_drawdown > 5.0:  # 5% drawdown protection
            return False, f"High drawdown: {current_drawdown:.1f}%"
            
        # Trade cooldown after loss
        if self.consecutive_losses >= 2:
            return False, f"Cooling down after {self.consecutive_losses} losses"
            
        return True, "All checks passed"

    def execute_trade(self, signal: Dict, current_price: float):
        """Execute validated trade"""
        validation, reason = self.should_enter_trade(signal)
        if not validation:
            print(f"⏸️  Trade skipped: {reason}")
            return False

        # Adaptive position sizing
        position_size = self.adaptive_position_sizing(signal['confidence'])
        trade_amount = self.balance * position_size / current_price
        
        self.position = {
            'type': signal['signal'],
            'entry_price': current_price,
            'size': trade_amount,
            'position_size': position_size,
            'entry_time': datetime.datetime.now(),
            'confidence': signal['confidence'],
            'tp_price': current_price * (1 + self.tp_percent) if signal['signal'] == 'BUY' else current_price * (1 - self.tp_percent),
            'sl_price': current_price * (1 - self.sl_percent) if signal['signal'] == 'BUY' else current_price * (1 + self.sl_percent),
            'timeframe_alignment': signal['timeframe_alignment'],
            'reason': signal['reason']
        }
        
        print(f"🎯 OPEN {signal['signal']} at ${current_price:.2f}")
        print(f"   Confidence: {signal['confidence']:.2f} | Timeframes: {signal['timeframe_alignment']}")
        print(f"   Position Size: {position_size*100:.1f}% | TP: {self.tp_percent*100}% | SL: {self.sl_percent*100}%")
        print(f"   Reason: {signal['reason']}")
        
        self.last_trade_time = datetime.datetime.now()
        self.daily_trades_count += 1
        
        return True

    def check_exit_conditions(self, current_price: float) -> bool:
        """Check TP/SL and other exit conditions"""
        if not self.position:
            return False
            
        position_type = self.position['type']
        entry_price = self.position['entry_price']
        position_size = self.position['position_size']
        
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
        """Close position and update statistics"""
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
        
        # Update performance metrics
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
        # Update equity curve and drawdown
        self.equity_curve.append(self.balance)
        if self.balance > self.peak_equity:
            self.peak_equity = self.balance
            
        current_drawdown = (self.peak_equity - self.balance) / self.peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
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
            'confidence': self.position['confidence'],
            'timeframe_alignment': self.position['timeframe_alignment']
        }
        self.trade_log.append(trade_record)
        
        emoji = "🎯" if reason == "TP" else "🛑"
        print(f"{emoji} CLOSE {position_type}: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        self.position = None

    def show_comprehensive_dashboard(self, current_price: float, signal: Optional[Dict]):
        """Show detailed real-time dashboard"""
        # Calculate current equity with unrealized PnL
        current_equity = self.balance
        unrealized_pnl = 0
        pnl_percent = 0
        
        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']
            size = self.position['size']
            
            if position_type == 'BUY':
                pnl_percent = (current_price - entry_price) / entry_price * 100
            else:
                pnl_percent = (entry_price - current_price) / entry_price * 100
                
            unrealized_pnl = pnl_percent / 100 * self.balance * self.position['position_size']
            current_equity += unrealized_pnl
        
        # Calculate performance metrics
        total_return = (current_equity - self.initial_balance) / self.initial_balance * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        
        # Calculate additional metrics
        avg_win = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] > 0]) if self.winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] < 0]) if self.losing_trades > 0 else 0
        profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if self.losing_trades > 0 and avg_loss != 0 else float('inf')
        
        print("\n" + "="*90)
        print("🚀 HIGH WIN RATE BITCOIN BOT - REAL-TIME DASHBOARD")
        print("="*90)
        print(f"🕒 Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Price: ${current_price:.2f}")
        print("-"*90)
        
        print("💵 BALANCE & EQUITY:")
        print(f"   Balance: ${self.balance:.2f}")
        print(f"   Equity: ${current_equity:.2f}")
        print(f"   Unrealized P&L: ${unrealized_pnl:+.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Peak Equity: ${self.peak_equity:.2f}")
        print(f"   Current Drawdown: {current_drawdown:.2f}%")
        print(f"   Max Drawdown: {self.max_drawdown:.2f}%")
        
        print("\n📊 TRADING STATISTICS:")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Winning Trades: {self.winning_trades}")
        print(f"   Losing Trades: {self.losing_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Consecutive Wins: {self.consecutive_wins}")
        print(f"   Consecutive Losses: {self.consecutive_losses}")
        print(f"   Total P&L: ${self.total_pnl:+.2f}")
        print(f"   Average Win: ${avg_win:+.2f}")
        print(f"   Average Loss: ${avg_loss:+.2f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Daily Trades: {self.daily_trades_count}/{self.max_daily_trades}")
        
        print("\n📈 CURRENT POSITION:")
        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']
            size = self.position['size']
            confidence = self.position['confidence']
            
            pnl_color = "🟢" if pnl_percent > 0 else "🔴"
            
            print(f"   Type: {position_type}")
            print(f"   Entry Price: ${entry_price:.2f}")
            print(f"   Size: {size:.6f} BTC")
            print(f"   P&L: {pnl_color} {pnl_percent:+.2f}% (${unrealized_pnl:+.2f})")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Timeframe Alignment: {self.position['timeframe_alignment']}")
            print(f"   TP: ${self.position['tp_price']:.2f}")
            print(f"   SL: ${self.position['sl_price']:.2f}")
            print(f"   Reason: {self.position['reason']}")
        else:
            print("   No active position")
            print("   Status: READY FOR TRADING")
            
        print(f"\n🎯 TRADING SIGNAL:")
        if signal:
            signal_color = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "⚪"
            print(f"   Signal: {signal_color} {signal['signal']}")
            print(f"   Confidence: {signal['confidence']:.2f}")
            print(f"   Timeframe Alignment: {signal['timeframe_alignment']}")
            print(f"   Reason: {signal['reason']}")
        else:
            print("   No high-confidence signal detected")
            print("   Waiting for multi-timeframe alignment...")
            
        print("="*90)

    def run_bot(self):
        """Main trading loop"""
        print("\n🤖 HIGH WIN RATE BOT STARTED")
        print("🎯 Targeting 70%+ Win Rate with Multi-Timeframe Analysis")
        print("⏰ Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                self.reset_daily_counters()
                
                # Fetch current price
                current_price = self.get_current_price()
                if current_price is None:
                    print("❌ Failed to get current price, retrying...")
                    time.sleep(60)
                    continue
                
                # Generate high-confidence signal
                signal = self.generate_high_confidence_signal()
                
                # Check exit conditions first
                if self.check_exit_conditions(current_price):
                    time.sleep(30)
                    continue
                
                # Show comprehensive dashboard
                self.show_comprehensive_dashboard(current_price, signal)
                
                # Execute new trade if high-confidence signal
                if signal and not self.position:
                    self.execute_trade(signal, current_price)
                
                print(f"⏰ Next check in 60 seconds... (Iteration: {iteration})\n")
                time.sleep(60)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Bot stopped after {iteration} iterations")
            self.show_final_summary()
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("🔄 Restarting bot in 10 seconds...")
            time.sleep(10)
            self.run_bot()

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
    bot = HighWinRateBot()
    bot.run_bot()

if __name__ == "__main__":
    main()
