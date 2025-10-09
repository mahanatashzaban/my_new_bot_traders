#!/usr/bin/env python3
"""
ULTRA HIGH-FREQUENCY BOT - MAXIMUM TRADING FREQUENCY
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

class UltraHighFrequencyBot:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = BidirectionalMLTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = 'BTC/USDT'
        self.position = None
        self.initial_balance = 100.0
        self.balance = 100.0
        self.trade_log = []

        # Performance tracking
        self.equity_curve = [1000.0]
        self.peak_equity = 1000.0
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

        # **ULTRA AGGRESSIVE** Settings
        self.leverage = 10           # Increased leverage
        self.tp_percent = 0.005      # Smaller TP for faster hits (0.5%)
        self.sl_percent = 0.004      # Smaller SL (0.4%)
        self.position_size = 0.35    # Larger position size

        # **EXTREMELY RELAXED** Signal quality
        self.min_confidence = 0.55   # Very low confidence threshold
        self.min_probability = 0.25  # Very low probability threshold
        self.require_consensus = False  # No consensus required

        # **MAXIMUM FREQUENCY** Trade management
        self.max_daily_trades = 50   # Much higher daily limit
        self.daily_trades_count = 0
        self.last_trade_date = None
        self.trade_cooldown = 30     # Only 30 seconds between trades

        # **MINIMAL** Market filters
        self.min_volume_ratio = 0.1  # Almost no volume requirement
        self.max_volatility = 8.0    # Very high volatility allowed

        # Force trading mode
        self.force_trade_mode = True
        self.consecutive_holds = 0
        self.max_consecutive_holds = 3  # Force trade after 3 holds

    def initialize(self):
        print("🚀 ULTRA HIGH-FREQUENCY BOT - MAXIMUM TRADING")
        print("=" * 60)

        if not self.ml_trainer.load_ml_models():
            print("❌ No trained ML models found.")
            return False

        print("✅ ML Models Loaded Successfully")
        print(f"💰 Initial Balance: ${self.balance:.2f}")
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"🎯 Position Size: {self.position_size*100}%")
        print(f"📈 TP/SL: {self.tp_percent*100}%/{self.sl_percent*100}%")
        print(f"🎯 Min Confidence: {self.min_confidence} (VERY LOW)")
        print(f"📊 Min Volume: {self.min_volume_ratio}x (MINIMAL)")
        print(f"📅 Max Daily Trades: {self.max_daily_trades}")
        print(f"⏰ Cooldown: {self.trade_cooldown}s")
        print("🤖 Strategy: FORCE TRADING MODE - MAXIMUM FREQUENCY")
        print("=" * 60)
        return True

    def reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades_count = 0
            self.last_trade_date = today
            print("🔄 Daily counters reset")

    def update_equity_curve(self, current_price):
        """Update equity curve with unrealized PnL"""
        current_equity = self.balance

        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']

            if position_type == 'LONG':
                unrealized_pnl = (current_price - entry_price) / entry_price * self.leverage
            else:
                unrealized_pnl = (self.position['entry_price'] - current_price) / self.position['entry_price'] * self.leverage

            current_equity += self.balance * self.position_size * unrealized_pnl

        self.equity_curve.append(current_equity)

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            return current_drawdown

        return 0.0

    def show_real_time_dashboard(self, current_price, signal_data, market_conditions):
        """Ultra-compact real-time dashboard"""
        current_equity = self.balance
        unrealized_pnl = 0
        pnl_percent = 0

        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']

            if position_type == 'LONG':
                pnl_percent = (current_price - entry_price) / entry_price * self.leverage
            else:
                pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price'] * self.leverage

            unrealized_pnl = self.balance * self.position_size * pnl_percent
            current_equity += unrealized_pnl

        total_return = (current_equity - self.initial_balance) / self.initial_balance * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        current_drawdown = self.update_equity_curve(current_price)

        print(f"\n🔄 {datetime.now().strftime('%H:%M:%S')} | ${current_price:.0f} | Bal: ${self.balance:.2f} | Equity: ${current_equity:.2f}")
        print(f"📊 Trades: {self.total_trades} | Win: {win_rate:.1f}% | PnL: ${self.total_pnl:+.2f} | Daily: {self.daily_trades_count}/{self.max_daily_trades}")

        if self.position:
            pnl_color = "🟢" if pnl_percent > 0 else "🔴"
            print(f"📈 POSITION: {self.position['type']} | P&L: {pnl_color} {pnl_percent*100:+.2f}% (${unrealized_pnl:+.2f})")
        else:
            print("💤 NO POSITION | READY TO TRADE")

        if signal_data and signal_data['prediction'] != 'HOLD':
            signal_color = "🟢" if signal_data['prediction'] == 'LONG' else "🔴"
            print(f"🎯 SIGNAL: {signal_color} {signal_data['prediction']} | Conf: {signal_data['confidence']:.2f}")
        else:
            print(f"⏸️  HOLD | Consecutive: {self.consecutive_holds}")

        print("-" * 80)

    def analyze_market_conditions(self, data):
        """Minimal market analysis - allow almost all conditions"""
        try:
            current_price = data['close'].iloc[-1]

            # Volume check (minimal requirement)
            current_volume = data['volume'].iloc[-1]
            volume_sma = data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1

            # Volatility check (very permissive)
            high_low = data['high'] - data['low']
            atr = high_low.rolling(14).mean().iloc[-1]
            atr_percent = (atr / current_price) * 100

            # Always suitable for trading in ultra mode
            conditions = {
                'volume_ratio': volume_ratio,
                'volatility': atr_percent,
                'suitable': True,  # Always true in ultra mode
                'reasons': []
            }

            return conditions

        except Exception as e:
            print(f"❌ Market analysis error: {e}")
            return {'suitable': True, 'reasons': []}  # Still allow trading

    def generate_forced_signal(self, data):
        """Generate signals with FORCED TRADING logic"""
        if not self.ml_trainer.models:
            return self._generate_random_signal()

        # Prepare features
        X, _ = self.ml_trainer.prepare_features(data, self.indicator_engine)
        if X is None or len(X) < 2:
            return self._generate_random_signal()

        try:
            X_scaled = self.ml_trainer.scaler.transform(X)
        except Exception as e:
            print(f"❌ Feature scaling error: {e}")
            return self._generate_random_signal()

        # Get probabilities from models
        current_probs = []
        for name, model in self.ml_trainer.models.items():
            try:
                current_prob = model.predict_proba(X_scaled)[-1]
                current_probs.append(current_prob)
                print(f"🤖 {name}: S:{current_prob[0]:.2f} H:{current_prob[1]:.2f} L:{current_prob[2]:.2f}")
            except Exception as e:
                continue

        if not current_probs:
            return self._generate_random_signal()

        # Calculate average probabilities
        avg_current = np.mean(current_probs, axis=0)
        short_now, hold_now, long_now = avg_current

        print(f"📊 AVERAGE: S:{short_now:.2f} H:{hold_now:.2f} L:{long_now:.2f}")

        # **FORCEFUL** Signal Generation
        signal = 'HOLD'
        confidence = hold_now

        # Force trade after too many consecutive holds
        if self.consecutive_holds >= self.max_consecutive_holds:
            if long_now > short_now:
                signal = 'LONG'
                confidence = max(long_now, 0.6)
                print("🔥 FORCING LONG - Too many consecutive holds!")
            else:
                signal = 'SHORT'
                confidence = max(short_now, 0.6)
                print("🔥 FORCING SHORT - Too many consecutive holds!")

        # Very relaxed LONG conditions
        elif (long_now > self.min_probability or short_now > self.min_probability):
            if long_now > short_now:
                signal = 'LONG'
                confidence = max(long_now + 0.1, 0.55)
            else:
                signal = 'SHORT'
                confidence = max(short_now + 0.1, 0.55)

        # Update consecutive holds counter
        if signal == 'HOLD':
            self.consecutive_holds += 1
        else:
            self.consecutive_holds = 0

        return {
            'prediction': signal,
            'confidence': confidence,
            'probabilities': {'SHORT': short_now, 'HOLD': hold_now, 'LONG': long_now},
            'forced': self.consecutive_holds >= self.max_consecutive_holds
        }

    def _generate_random_signal(self):
        """Fallback: Generate random signal when ML fails"""
        import random
        signals = ['LONG', 'SHORT', 'HOLD']
        weights = [0.4, 0.4, 0.2]  # Prefer trading over holding
        
        signal = random.choices(signals, weights=weights)[0]
        confidence = random.uniform(0.6, 0.8) if signal != 'HOLD' else 0.5
        
        print(f"🎲 RANDOM SIGNAL: {signal} (Conf: {confidence:.2f})")
        
        if signal == 'HOLD':
            self.consecutive_holds += 1
        else:
            self.consecutive_holds = 0
            
        return {
            'prediction': signal,
            'confidence': confidence,
            'probabilities': {'SHORT': 0.33, 'HOLD': 0.33, 'LONG': 0.33},
            'forced': False
        }

    def should_enter_trade(self, signal, market_conditions):
        """MINIMAL trade validation"""
        if signal['prediction'] == 'HOLD':
            return False, "No signal"

        if self.position:
            return False, "In position"

        if self.daily_trades_count >= self.max_daily_trades:
            return False, "Daily limit"

        # **VERY LOW** confidence threshold
        if signal['confidence'] < self.min_confidence:
            return False, f"Low conf: {signal['confidence']:.2f}"

        # **MINIMAL** cooldown
        if hasattr(self, 'last_trade_time'):
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.trade_cooldown:
                return False, f"Cooldown: {int(self.trade_cooldown - time_since_last)}s"

        return True, "OK"

    def execute_trade(self, signal, current_price, market_conditions):
        """Execute trade instantly"""
        validation, reason = self.should_enter_trade(signal, market_conditions)
        if not validation:
            print(f"⏸️  Skip: {reason}")
            return False

        # Position sizing
        leveraged_size = self.balance * self.position_size * self.leverage
        btc_size = leveraged_size / current_price

        if signal['prediction'] == 'LONG':
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'forced': signal.get('forced', False)
            }
            print(f"💚 LONG @ ${current_price:.0f} | Size: {btc_size:.4f} BTC")

        else:  # SHORT
            self.position = {
                'type': 'SHORT',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'forced': signal.get('forced', False)
            }
            print(f"🔴 SHORT @ ${current_price:.0f} | Size: {btc_size:.4f} BTC")

        self.last_trade_time = datetime.now()
        self.daily_trades_count += 1
        return True

    def check_tp_sl(self, current_price):
        """Check Take Profit and Stop Loss quickly"""
        if not self.position:
            return False

        entry_price = self.position['entry_price']
        position_type = self.position['type']

        if position_type == 'LONG':
            pnl_percent = (current_price - entry_price) / entry_price * self.leverage
        else:
            pnl_percent = (entry_price - current_price) / entry_price * self.leverage

        pnl_amount = self.balance * self.position_size * pnl_percent

        # Take Profit
        if pnl_percent >= self.tp_percent:
            self._close_position(current_price, pnl_percent, pnl_amount, "TP")
            return True

        # Stop Loss
        elif pnl_percent <= -self.sl_percent:
            self._close_position(current_price, pnl_percent, pnl_amount, "SL")
            return True

        return False

    def _close_position(self, current_price, pnl_percent, pnl_amount, reason):
        """Close position quickly"""
        position_type = self.position['type']
        
        self.balance += pnl_amount
        self.total_trades += 1
        self.total_pnl += pnl_amount

        if pnl_amount > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        emoji = "🎯" if reason == "TP" else "🛑"
        print(f"{emoji} {reason} {position_type}: {pnl_percent*100:+.2f}% (${pnl_amount:+.2f})")
        self.position = None

    def run_bot(self):
        """Main bot loop - ULTRA FAST"""
        if not self.initialize():
            return

        print("\n🤖 ULTRA HIGH-FREQUENCY BOT STARTED")
        print("⚡ MAXIMUM TRADING FREQUENCY")
        print("🎯 FORCE TRADING MODE ACTIVE")
        print("Press Ctrl+C to stop\n")

        iteration = 0

        try:
            while True:
                iteration += 1
                self.reset_daily_counters()

                # Fetch data quickly
                data = self.data_manager.fetch_historical_data(limit=100)  # Smaller data
                if data is None or len(data) < 50:
                    time.sleep(10)
                    continue

                current_price = data['close'].iloc[-1]

                # Quick market analysis
                market_conditions = self.analyze_market_conditions(data)

                # Generate signal (forced if needed)
                signal_data = self.generate_forced_signal(data)

                # Show compact dashboard
                self.show_real_time_dashboard(current_price, signal_data, market_conditions)

                # Check TP/SL first
                if self.check_tp_sl(current_price):
                    time.sleep(10)  # Very short wait after close
                    continue

                # Execute trade aggressively
                if signal_data and signal_data['prediction'] != 'HOLD' and not self.position:
                    self.execute_trade(signal_data, current_price, market_conditions)

                # Shorter sleep for maximum frequency
                sleep_time = 15 if not self.position else 5  # Check more often when in position
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n🛑 Bot stopped after {iteration} iterations")
            final_equity = self.equity_curve[-1] if self.equity_curve else self.balance
            total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
            print(f"💰 Final Balance: ${final_equity:.2f}")
            print(f"📈 Total Return: {total_return:+.2f}%")
            print(f"📊 Total Trades: {self.total_trades}")

if __name__ == "__main__":
    bot = UltraHighFrequencyBot()
    bot.run_bot()
