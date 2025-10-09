#!/usr/bin/env python3
"""
ML Forced Signals Bot - WITH REAL-TIME DASHBOARD
AGGRESSIVE TRADING VERSION - $100 BALANCE
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_manager_simple import DataManager
from ml_trainer_bidirectional_fixed import BidirectionalMLTrainer
from simple_indicators import SimpleMLIndicatorEngine

class OptimizedMLBot:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = BidirectionalMLTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = 'BTC/USDT'
        self.position = None
        self.initial_balance = 100.0  # REDUCED TO $100
        self.balance = 100.0  # REDUCED TO $100
        self.trade_log = []

        # Performance tracking
        self.equity_curve = [100.0]  # REDUCED TO $100
        self.peak_equity = 100.0  # REDUCED TO $100
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        # **MORE AGGRESSIVE** Risk Management
        self.leverage = 10
        self.tp_percent = 0.014    # 0.8% Take Profit
        self.sl_percent = 0.006    # 0.6% Stop Loss
        self.position_size = 0.5   # INCREASED to 50% of balance

        # Trading fees (commented out for now)
        self.maker_fee = 0.0002  # 0.1%
        self.taker_fee = 0.0005  # 0.1%

        # **INCREASED** Daily trade limit
        self.max_daily_trades = 30  # Increased from 20
        self.daily_trades_count = 0
        self.last_trade_date = None

        # **MORE AGGRESSIVE** Signal filtering
        self.signal_streak = 0
        self.last_signal = None
        self.min_confidence = 0.50  # REDUCED from 0.60 to 0.50
        self.last_trade_time = None
        self.trade_cooldown = 120   # REDUCED to 2 minutes

        # Market state tracking
        self.market_volatility = 0.0
        self.trend_direction = "NEUTRAL"

    def initialize(self):
        print("🚀 AGGRESSIVE ML TRADING BOT - $100 BALANCE")
        print("=" * 60)

        if not self.ml_trainer.load_ml_models():
            print("❌ No trained ML models found.")
            return False

        print("✅ ML Models Loaded Successfully")
        print(f"💰 Initial Balance: ${self.balance:.2f}")
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"🎯 Position Size: {self.position_size*100}%")
        print(f"📈 TP/SL: {self.tp_percent*100}%/{self.sl_percent*100}%")
        print(f"🎯 Min Confidence: {self.min_confidence} (AGGRESSIVE)")
        print(f"📊 Daily Trade Limit: {self.max_daily_trades}")
        print(f"💰 Trading Fees: {self.taker_fee*100:.1f}% (taker)")
        print("=" * 60)
        return True

    def reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades_count = 0
            self.last_trade_date = today
            print("🔄 Daily trade counter reset")

    def update_equity_curve(self, current_price):
        """Update equity curve with unrealized PnL"""
        current_equity = self.balance

        # Add unrealized PnL if in position
        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']

            if position_type == 'LONG':
                unrealized_pnl = (current_price - entry_price) / entry_price * self.leverage
            else:  # SHORT
                unrealized_pnl = (entry_price - current_price) / entry_price * self.leverage

            current_equity += self.balance * self.position_size * unrealized_pnl

        self.equity_curve.append(current_equity)

        # Update peak equity and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            return current_drawdown

        return 0.0

    def show_real_time_dashboard(self, current_price, signal_data):
        """Show comprehensive real-time dashboard with all metrics"""
        # Calculate current equity with unrealized PnL
        current_equity = self.balance
        unrealized_pnl = 0
        pnl_percent = 0
        position_age = None

        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']

            if position_type == 'LONG':
                pnl_percent = (current_price - entry_price) / entry_price * self.leverage
            else:
                pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price'] * self.leverage

            unrealized_pnl = self.balance * self.position_size * pnl_percent
            current_equity += unrealized_pnl
            position_age = datetime.now() - self.position['entry_time']

        # Calculate performance metrics
        total_return = (current_equity - self.initial_balance) / self.initial_balance * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        current_drawdown = self.update_equity_curve(current_price)

        print("\n" + "="*80)
        print("📊 REAL-TIME TRADING DASHBOARD - $100 AGGRESSIVE")
        print("="*80)
        print(f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Price: ${current_price:.2f}")
        print("-"*80)

        # Balance & Equity Section
        print("💵 BALANCE & EQUITY:")
        print(f"   Balance: ${self.balance:.2f}")
        print(f"   Equity: ${current_equity:.2f}")
        print(f"   Unrealized P&L: ${unrealized_pnl:+.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Max Drawdown: {self.max_drawdown:.2f}%")
        print(f"   Current Drawdown: {current_drawdown:.2f}%")

        # Trading Statistics
        print("\n📈 TRADING STATISTICS:")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Winning Trades: {self.winning_trades}")
        print(f"   Losing Trades: {self.losing_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total P&L: ${self.total_pnl:+.2f}")
        print(f"   Daily Trades: {self.daily_trades_count}/{self.max_daily_trades}")

        # Current Position Details
        print("\n📊 CURRENT POSITION:")
        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']
            size = self.position['size']
            confidence = self.position.get('confidence', 0)

            pnl_color = "🟢" if pnl_percent > 0 else "🔴"

            print(f"   Type: {position_type}")
            print(f"   Entry Price: ${entry_price:.2f}")
            print(f"   Size: {size:.6f} BTC")
            print(f"   P&L: {pnl_color} {pnl_percent*100:+.2f}% (${unrealized_pnl:+.2f})")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Age: {position_age}")

            # Calculate distance to TP/SL
            if position_type == 'LONG':
                to_tp = (self.tp_percent - pnl_percent) * 100
                to_sl = (self.sl_percent + pnl_percent) * 100
            else:
                to_tp = (self.tp_percent - pnl_percent) * 100
                to_sl = (self.sl_percent + pnl_percent) * 100

            print(f"   To TP: {to_tp:.2f}% | To SL: {to_sl:.2f}%")
        else:
            print("   No active position")
            print("   Status: READY FOR TRADING")

        # ML Signal Information
        print(f"\n🎯 ML TRADING SIGNAL:")
        if signal_data:
            signal_color = "🟢" if signal_data['prediction'] == 'LONG' else "🔴" if signal_data['prediction'] == 'SHORT' else "⚪"
            print(f"   Signal: {signal_color} {signal_data['prediction']}")
            print(f"   Confidence: {signal_data['confidence']:.3f}")
            print(f"   Streak: {signal_data.get('streak', 0)}")

            # Show probabilities
            probs = signal_data.get('probabilities', {})
            if probs:
                print(f"   Probabilities - LONG: {probs.get('LONG', 0):.3f} | SHORT: {probs.get('SHORT', 0):.3f} | HOLD: {probs.get('HOLD', 0):.3f}")
        else:
            print("   No signal data available")

        # Market Conditions
        print(f"\n🌡️  MARKET CONDITIONS:")
        print(f"   Volatility: {self.market_volatility:.2f}%")
        print(f"   Trend: {self.trend_direction}")
        print(f"   Signal Streak: {self.signal_streak}")

        print("="*80)

    def force_ml_signals(self, df):
        """HIGHLY AGGRESSIVE ML signal generation - Forces more trades"""
        if not self.ml_trainer.models:
            return None

        if len(df) < 100:
            print("❌ Insufficient data for ML analysis")
            return None

        # Prepare features
        X, _ = self.ml_trainer.prepare_features(df, self.indicator_engine)
        if X is None or len(X) < 2:
            return None

        try:
            X_scaled = self.ml_trainer.scaler.transform(X)
        except Exception as e:
            print(f"❌ Feature scaling error: {e}")
            return None

        # Get probabilities from models
        current_probs = []
        previous_probs = []

        for name, model in self.ml_trainer.models.items():
            try:
                current_prob = model.predict_proba(X_scaled)[-1]
                current_probs.append(current_prob)

                if len(X_scaled) > 1:
                    previous_prob = model.predict_proba(X_scaled)[-2]
                    previous_probs.append(previous_prob)

                print(f"📊 {name}: SHORT:{current_prob[0]:.3f} HOLD:{current_prob[1]:.3f} LONG:{current_prob[2]:.3f}")
            except Exception as e:
                print(f"❌ Model {name} error: {e}")
                continue

        if not current_probs:
            return None

        # Average probabilities
        avg_current = np.mean(current_probs, axis=0)
        short_now, hold_now, long_now = avg_current

        # Calculate momentum
        if previous_probs:
            avg_previous = np.mean(previous_probs, axis=0)
            long_momentum = long_now - avg_previous[2]
            short_momentum = short_now - avg_previous[0]
        else:
            long_momentum = 0
            short_momentum = 0

        print(f"📊 MODEL CONSENSUS: SHORT:{short_now:.3f} HOLD:{hold_now:.3f} LONG:{long_now:.3f}")
        print(f"📈 PROBABILITY MOMENTUM: LONG{long_momentum:+.3f} SHORT{short_momentum:+.3f}")

        # **HIGHLY AGGRESSIVE** Signal Generation - Forces Trading
        signal = 'HOLD'
        confidence = hold_now

        # Calculate signal strength
        signal_strength = abs(long_now - short_now)

        # **NEW: FORCE SIGNALS WHEN CLEAR DIRECTION**
        if signal_strength > 0.10:  # Clear directional bias
            if long_now > short_now and long_now > 0.25:
                signal = 'LONG'
                confidence = max(long_now + 0.15, 0.60)  # Force higher confidence
                print("🔥 FORCING LONG - Clear directional bias!")
            elif short_now > long_now and short_now > 0.25:
                signal = 'SHORT'
                confidence = max(short_now + 0.15, 0.60)  # Force higher confidence
                print("🔥 FORCING SHORT - Clear directional bias!")

        # **RELAXED** LONG conditions
        long_conditions = [
            long_now > 0.25,                      # Reduced from 0.30
            long_now > short_now,
            long_now > 0.25,                      # Simple threshold
            long_momentum > -0.12,                # More tolerant of declines
        ]

        # **RELAXED** SHORT conditions
        short_conditions = [
            short_now > 0.25,                     # Reduced from 0.30
            short_now > long_now,
            short_now > 0.25,                     # Simple threshold
            short_momentum > -0.12,               # More tolerant of declines
        ]

        # Check for strong signals first
        if long_now > 0.40 and long_now > short_now:
            signal = 'LONG'
            confidence = min(long_now + 0.12, 0.85)
        elif short_now > 0.40 and short_now > long_now:
            signal = 'SHORT'
            confidence = min(short_now + 0.12, 0.85)
        # Then check relaxed conditions (only if no forced signal yet)
        elif signal == 'HOLD' and all(long_conditions):
            signal = 'LONG'
            confidence = min(long_now + 0.10, 0.80)
        elif signal == 'HOLD' and all(short_conditions):
            signal = 'SHORT'
            confidence = min(short_now + 0.10, 0.80)

        # **NEW: ULTRA-AGGRESSIVE MODE** - Force trade on any momentum
        if signal == 'HOLD' and self.daily_trades_count < 10:  # Only in first 10 trades
            if long_momentum > 0.05 and long_now > 0.20:
                signal = 'LONG'
                confidence = 0.55
                print("⚡ ULTRA-AGGRESSIVE LONG - Momentum detected!")
            elif short_momentum > 0.05 and short_now > 0.20:
                signal = 'SHORT'
                confidence = 0.55
                print("⚡ ULTRA-AGGRESSIVE SHORT - Momentum detected!")

        # Track signal streaks with confidence boost
        if signal == self.last_signal and signal != 'HOLD':
            self.signal_streak += 1
            confidence = min(confidence + (self.signal_streak * 0.05), 0.95)
            print(f"🔥 Signal Streak: {self.signal_streak} - Confidence Boost!")
        else:
            self.signal_streak = 1

        self.last_signal = signal

        return {
            'prediction': signal,
            'confidence': confidence,
            'probabilities': {'SHORT': short_now, 'HOLD': hold_now, 'LONG': long_now},
            'momentum': {'LONG': long_momentum, 'SHORT': short_momentum},
            'streak': self.signal_streak
        }

    def should_enter_trade(self, signal, market_context):
        """HIGHLY AGGRESSIVE trade validation - Minimal restrictions"""
        if signal['prediction'] == 'HOLD':
            return False, "No valid signal"

        # **REDUCED** confidence threshold
        if signal['confidence'] < self.min_confidence:
            return False, f"Low confidence: {signal['confidence']:.3f}"

        # Daily trade limit
        if self.daily_trades_count >= self.max_daily_trades:
            return False, "Daily trade limit reached"

        # **REDUCED** cooldown check
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.trade_cooldown:
                return False, f"In cooldown: {int(self.trade_cooldown - time_since_last)}s remaining"

        # **RELAXED** volatility filter
        if market_context and market_context['volatility'] > 5.0:  # Increased from 4.0%
            return False, f"High volatility: {market_context['volatility']:.2f}%"

        return True, "All checks passed"

    def execute_trade(self, signal, market_context, current_price):
        """Execute validated trade"""
        validation, reason = self.should_enter_trade(signal, market_context)
        if not validation:
            print(f"⏸️  Trade skipped: {reason}")
            return False

        # Calculate position size
        leveraged_size = self.balance * self.position_size * self.leverage
        btc_size = leveraged_size / current_price

        # Calculate entry fee (commented out for now)
        entry_fee = leveraged_size * self.taker_fee

        if signal['prediction'] == 'LONG':
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'streak': signal['streak'],
                'entry_fee': entry_fee
            }

            # Apply entry fee (commented out for now)
            self.balance -= entry_fee

            print(f"💚 ENTERED LONG at ${current_price:.2f}")
            print(f"   Confidence: {signal['confidence']:.3f} | Streak: {signal['streak']}")
            print(f"   Entry Fee: ${entry_fee:.4f}")

        else:  # SHORT
            self.position = {
                'type': 'SHORT',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'streak': signal['streak'],
                'entry_fee': entry_fee
            }

            # Apply entry fee (commented out for now)
            self.balance -= entry_fee

            print(f"🔴 ENTERED SHORT at ${current_price:.2f}")
            print(f"   Confidence: {signal['confidence']:.3f} | Streak: {signal['streak']}")
            print(f"   Entry Fee: ${entry_fee:.4f}")

        self.last_trade_time = datetime.now()
        self.daily_trades_count += 1
        print(f"📊 Daily trades: {self.daily_trades_count}/{self.max_daily_trades}")
        return True

    def check_tp_sl(self, current_price):
        """Check Take Profit and Stop Loss"""
        if not self.position:
            return False

        if self.position['type'] == 'LONG':
            pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
        else:
            pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price']

        pnl_percent *= self.leverage
        pnl_amount = self.balance * self.position_size * pnl_percent

        # Take Profit
        if pnl_percent >= self.tp_percent:
            # Calculate exit fee (commented out for now)
            position_value = self.position['size'] * current_price
            exit_fee = position_value * self.taker_fee
            total_fee = self.position.get('entry_fee', 0) + exit_fee

            # Apply PnL and deduct fees
            self.balance += pnl_amount
            self.balance -= exit_fee

            self.total_trades += 1
            self.total_pnl += pnl_amount

            if pnl_amount > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            trade_record = {
                'action': 'TP_' + self.position['type'],
                'price': current_price,
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'timestamp': datetime.now(),
                'confidence': self.position['confidence'],
                'exit_fee': exit_fee,
                'total_fee': total_fee
            }
            self.trade_log.append(trade_record)

            print(f"🎯 TAKE PROFIT {self.position['type']}: {pnl_percent*100:+.2f}% (${pnl_amount:+.2f})")
            print(f"       Exit Fee: ${exit_fee:.4f} | Total Fees: ${total_fee:.4f}")
            self.position = None
            self.signal_streak = 0
            return True

        # Stop Loss
        elif pnl_percent <= -self.sl_percent:
            # Calculate exit fee (commented out for now)
            position_value = self.position['size'] * current_price
            exit_fee = position_value * self.taker_fee
            total_fee = self.position.get('entry_fee', 0) + exit_fee

            # Apply PnL and deduct fees
            self.balance += pnl_amount
            self.balance -= exit_fee

            self.total_trades += 1
            self.total_pnl += pnl_amount

            if pnl_amount > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            trade_record = {
                'action': 'SL_' + self.position['type'],
                'price': current_price,
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'timestamp': datetime.now(),
                'confidence': self.position['confidence'],
                'exit_fee': exit_fee,
                'total_fee': total_fee
            }
            self.trade_log.append(trade_record)

            print(f"🛑 STOP LOSS {self.position['type']}: {pnl_percent*100:+.2f}% (${pnl_amount:+.2f})")
            print(f"       Exit Fee: ${exit_fee:.4f} | Total Fees: ${total_fee:.4f}")
            self.position = None
            self.signal_streak = 0
            return True

        return False

    def analyze_market_conditions(self, data):
        """Basic market analysis"""
        try:
            current_price = data['close'].iloc[-1]

            # Simple trend
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            self.trend_direction = "BULLISH" if current_price > sma_20 else "BEARISH"

            # Volatility
            high_low = data['high'] - data['low']
            atr = high_low.rolling(14).mean().iloc[-1]
            self.market_volatility = (atr / current_price) * 100

            return {
                'trend': self.trend_direction,
                'volatility': self.market_volatility
            }
        except:
            return None

    def run_bot(self):
        """Main bot loop with real-time dashboard"""
        if not self.initialize():
            return

        print("\n🤖 AGGRESSIVE ML BOT STARTED - $100 BALANCE")
        print("Press Ctrl+C to stop")
        print("=" * 60)

        iteration = 0

        try:
            while True:
                iteration += 1
                self.reset_daily_counters()

                # Fetch data
                data = self.data_manager.fetch_historical_data(limit=150)
                if data is None or len(data) < 100:
                    print("❌ Insufficient data")
                    time.sleep(300)
                    continue

                current_price = data['close'].iloc[-1]
                current_time = datetime.now().strftime('%H:%M:%S')

                print(f"\n🕒 {current_time} | Iter {iteration} | Trades: {self.daily_trades_count}/{self.max_daily_trades}")
                print("-" * 50)

                # Check TP/SL first
                if self.check_tp_sl(current_price):
                    time.sleep(120)  # Shorter wait after position close
                    continue

                # Analyze market conditions
                market_context = self.analyze_market_conditions(data)

                # Generate signals
                signal_data = self.force_ml_signals(data)

                # Show REAL-TIME DASHBOARD with all metrics
                self.show_real_time_dashboard(current_price, signal_data)

                # Execute trade
                if signal_data and signal_data['prediction'] != 'HOLD':
                    self.execute_trade(signal_data, market_context, current_price)

                print(f"⏰ Next check in 5 minutes...\n")
                time.sleep(300)  # 5 minutes between checks

        except KeyboardInterrupt:
            print(f"\n🛑 Bot stopped after {iteration} iterations")
            # Show final dashboard
            if 'current_price' in locals():
                self.show_real_time_dashboard(current_price, None)

            # Calculate total fees (commented out for now)
            total_fees = sum(trade.get('total_fee', 0) for trade in self.trade_log if 'total_fee' in trade)
            print(f"💰 TOTAL FEES PAID: ${total_fees:.4f}")

if __name__ == "__main__":
    bot = OptimizedMLBot()
    bot.run_bot()
