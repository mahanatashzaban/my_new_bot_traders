#!/usr/bin/env python3
"""
PROFESSIONAL Real-time Monitoring Bot - FIXED VERSION
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_manager_simple import DataManager
from ml_trainer_bidirectional_fixed import BidirectionalMLTrainer
from simple_indicators import SimpleMLIndicatorEngine

class ProfessionalRealTimeBot:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = BidirectionalMLTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = 'BTC/USDT'
        self.position = None
        self.initial_balance = 100.0  # Start with $1000
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

        # **OPTIMIZED** Trading parameters
        self.leverage = 5           # Reduced from 10x
        self.tp_percent = 0.010     # 1.0% Take Profit
        self.sl_percent = 0.008     # 0.8% Stop Loss
        self.position_size = 0.3    # 30% position size
        
        # **OPTIMIZED** Monitoring intervals
        self.monitor_interval = 30   # Check TP/SL every 30 seconds
        self.signal_interval = 300   # Check signals every 5 minutes (NOT 10 seconds!)
        
        # Trade management
        self.max_daily_trades = 20
        self.daily_trades_count = 0
        self.last_trade_date = None
        
        # Signal quality filters
        self.min_confidence = 0.65
        self.min_probability = 0.35
        self.require_consensus = True

    def initialize(self):
        print("🚀 PROFESSIONAL REAL-TIME BOT - OPTIMIZED")
        print("=" * 60)

        if not self.ml_trainer.load_ml_models():
            print("❌ No trained ML models found.")
            return False

        print("✅ ML Models Loaded Successfully")
        print(f"💰 Initial Balance: ${self.balance:.2f}")
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"🎯 Position Size: {self.position_size*100}%")
        print(f"📈 TP/SL: {self.tp_percent*100}%/{self.sl_percent*100}%")
        print(f"🔍 TP/SL Monitoring: Every {self.monitor_interval}s")
        print(f"📊 Signal Checks: Every {self.signal_interval//60} minutes")
        print(f"🎯 Min Confidence: {self.min_confidence}")
        print("=" * 60)
        return True

    def reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades_count = 0
            self.last_trade_date = today
            print("🔄 Daily trade counter reset")

    def update_equity_curve(self):
        """Update equity curve with unrealized PnL"""
        current_equity = self.balance
        
        # Add unrealized PnL if in position
        if self.position:
            current_price = self.get_current_price()
            if current_price:
                if self.position['type'] == 'LONG':
                    unrealized_pnl = (current_price - self.position['entry_price']) / self.position['entry_price'] * self.leverage
                else:
                    unrealized_pnl = (self.position['entry_price'] - current_price) / self.position['entry_price'] * self.leverage
                current_equity += self.balance * self.position_size * unrealized_pnl

        self.equity_curve.append(current_equity)

        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            return current_drawdown
        
        return 0.0

    def show_performance_report(self):
        """Show comprehensive performance report"""
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            avg_trade = self.total_pnl / self.total_trades
        else:
            win_rate = 0.0
            avg_trade = 0.0

        # Calculate current equity with unrealized PnL
        current_equity = self.balance
        if self.position:
            current_price = self.get_current_price()
            if current_price:
                if self.position['type'] == 'LONG':
                    unrealized = (current_price - self.position['entry_price']) / self.position['entry_price'] * self.leverage
                else:
                    unrealized = (self.position['entry_price'] - current_price) / self.position['entry_price'] * self.leverage
                current_equity += self.balance * self.position_size * unrealized

        print("\n" + "=" * 70)
        print("📊 LIVE PERFORMANCE REPORT")
        print("=" * 70)
        print(f"💰 Current Equity: ${current_equity:.2f}")
        print(f"💰 Balance: ${self.balance:.2f}")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")
        print(f"🔢 Total Trades: {self.total_trades}")
        print(f"🎯 Win Rate: {win_rate:.1f}% ({self.winning_trades}/{self.total_trades})")
        print(f"📊 Total P&L: ${self.total_pnl:+.2f}")
        print(f"📈 Avg Trade: ${avg_trade:+.2f}")
        print(f"📅 Daily Trades: {self.daily_trades_count}/{self.max_daily_trades}")

        # Show recent trades
        if self.trade_log:
            print(f"\n🔄 Recent Trades (Last 5):")
            print("-" * 50)
            for trade in self.trade_log[-5:]:
                action = trade.get('action', 'UNKNOWN')
                price = trade.get('price', 0)
                pnl = trade.get('pnl_percent', 0)
                pnl_amount = trade.get('pnl_amount', 0)
                timestamp = trade.get('timestamp', 'N/A')
                confidence = trade.get('confidence', 0)

                if isinstance(timestamp, datetime):
                    timestamp = timestamp.strftime('%H:%M:%S')

                pnl_color = "🟢" if pnl > 0 else "🔴"
                action_icon = "💚" if "LONG" in action else "🔴" if "SHORT" in action else "⚪"
                
                print(f"  {timestamp} | {action_icon} {action:10} | ${price:8.2f} | {pnl_color}{pnl*100:+.2f}% (${pnl_amount:+.2f}) | Conf: {confidence:.3f}")

    def show_position_report(self):
        """Show current position status"""
        if self.position:
            current_price = self.get_current_price()
            if current_price:
                if self.position['type'] == 'LONG':
                    pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price'] * self.leverage
                else:
                    pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price'] * self.leverage

                pnl_amount = self.balance * self.position_size * pnl_percent
                pnl_color = "🟢" if pnl_percent > 0 else "🔴"
                position_age = datetime.now() - self.position['entry_time']

                print(f"\n📈 ACTIVE POSITION:")
                print("-" * 50)
                print(f"  Type: {self.position['type']}")
                print(f"  Entry: ${self.position['entry_price']:.2f}")
                print(f"  Current: ${current_price:.2f}")
                print(f"  PnL: {pnl_color}{pnl_percent*100:+.2f}% (${pnl_amount:+.2f})")
                print(f"  Age: {position_age}")
                print(f"  Confidence: {self.position.get('confidence', 0):.3f}")
        else:
            print(f"\n📭 NO ACTIVE POSITION")

    def get_current_price(self):
        """Get real-time current price"""
        try:
            ticker = self.data_manager.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            print(f"❌ Error getting current price: {e}")
            # Fallback to historical data
            try:
                data = self.data_manager.fetch_historical_data(limit=5)
                return data['close'].iloc[-1] if data is not None else None
            except:
                return None

    def check_tp_sl_realtime(self):
        """Enhanced TP/SL checking with real-time price"""
        if not self.position:
            return False

        current_price = self.get_current_price()
        if current_price is None:
            return False

        if self.position['type'] == 'LONG':
            pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
        else:  # SHORT
            pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price']

        # Apply leverage
        pnl_percent *= self.leverage
        pnl_amount = self.balance * self.position_size * pnl_percent

        current_time = datetime.now().strftime('%H:%M:%S')

        # Take Profit
        if pnl_percent >= self.tp_percent:
            self.balance += pnl_amount
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
                'entry_price': self.position['entry_price'],
                'confidence': self.position.get('confidence', 0)
            }
            self.trade_log.append(trade_record)

            print(f"🎯 [{current_time}] TP {self.position['type']}: {pnl_percent*100:+.2f}% (${pnl_amount:+.2f})")
            self.position = None
            return True

        # Stop Loss
        elif pnl_percent <= -self.sl_percent:
            self.balance += pnl_amount
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
                'entry_price': self.position['entry_price'],
                'confidence': self.position.get('confidence', 0)
            }
            self.trade_log.append(trade_record)

            print(f"🛑 [{current_time}] SL {self.position['type']}: {pnl_percent*100:+.2f}% (${pnl_amount:+.2f})")
            self.position = None
            return True

        # Show PnL update every minute (not every check to reduce spam)
        if datetime.now().second == 0:  # Only at the start of each minute
            pnl_color = "🟢" if pnl_percent > 0 else "🔴"
            print(f"📊 [{current_time}] {self.position['type']} PnL: {pnl_color}{pnl_percent*100:+.2f}%")

        return False

    def generate_ml_signal(self, df):
        """ENHANCED ML signal generation with better filtering"""
        if not self.ml_trainer.models:
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

        # Get probabilities from all models
        model_predictions = {}
        current_probs = []

        for name, model in self.ml_trainer.models.items():
            try:
                current_prob = model.predict_proba(X_scaled)[-1]
                model_predictions[name] = current_prob
                current_probs.append(current_prob)
                
                print(f"📊 {name}: SHORT:{current_prob[0]:.3f} HOLD:{current_prob[1]:.3f} LONG:{current_prob[2]:.3f}")
            except Exception as e:
                print(f"❌ Model {name} error: {e}")
                continue

        if not current_probs:
            return None

        # Calculate average probabilities
        avg_current = np.mean(current_probs, axis=0)
        short_now, hold_now, long_now = avg_current

        print(f"📊 MODEL CONSENSUS: SHORT:{short_now:.3f} HOLD:{hold_now:.3f} LONG:{long_now:.3f}")

        # **ENHANCED Signal Logic**
        signal = 'HOLD'
        confidence = hold_now

        # Check for model consensus
        long_models = 0
        short_models = 0
        
        for name, probs in model_predictions.items():
            if probs[2] > probs[0] and probs[2] > probs[1]:  # LONG highest
                long_models += 1
            elif probs[0] > probs[2] and probs[0] > probs[1]:  # SHORT highest
                short_models += 1

        total_models = len(model_predictions)
        long_consensus = long_models / total_models if total_models > 0 else 0
        short_consensus = short_models / total_models if total_models > 0 else 0

        print(f"🤝 Model Consensus: LONG {long_models}/{total_models} | SHORT {short_models}/{total_models}")

        # **STRONG Signal Conditions**
        # LONG signal requires: high probability + model consensus + minimum threshold
        if (long_now > self.min_probability and 
            long_now > short_now and 
            long_now > hold_now and
            long_consensus >= 0.5):  # At least 50% model consensus
            
            signal = 'LONG'
            confidence = min(long_now + 0.1, 0.95)  # Boost confidence

        # SHORT signal requires: high probability + model consensus + minimum threshold
        elif (short_now > self.min_probability and 
              short_now > long_now and 
              short_now > hold_now and
              short_consensus >= 0.5):  # At least 50% model consensus
            
            signal = 'SHORT'
            confidence = min(short_now + 0.1, 0.95)  # Boost confidence

        return {
            'prediction': signal,
            'confidence': confidence,
            'probabilities': {'SHORT': short_now, 'HOLD': hold_now, 'LONG': long_now},
            'consensus': {'LONG': long_consensus, 'SHORT': short_consensus}
        }

    def should_enter_trade(self, signal):
        """Enhanced trade validation"""
        if signal['prediction'] == 'HOLD':
            return False, "No valid signal"

        if self.position:
            return False, "Already in position"

        if self.daily_trades_count >= self.max_daily_trades:
            return False, "Daily trade limit reached"

        if signal['confidence'] < self.min_confidence:
            return False, f"Low confidence: {signal['confidence']:.3f}"

        # Check probability threshold
        if signal['prediction'] == 'LONG':
            prob = signal['probabilities']['LONG']
        else:
            prob = signal['probabilities']['SHORT']

        if prob < self.min_probability:
            return False, f"Low probability: {prob:.3f}"

        return True, "All checks passed"

    def execute_trade(self, signal, current_price):
        """Execute validated trade"""
        validation, reason = self.should_enter_trade(signal)
        if not validation:
            print(f"⏸️  Trade skipped: {reason}")
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
                'consensus': signal['consensus']['LONG']
            }

            print(f"💚 REALTIME LONG at ${current_price:.2f}")
            print(f"   Confidence: {signal['confidence']:.3f} | Consensus: {signal['consensus']['LONG']:.1%}")
            print(f"   Size: {btc_size:.6f} BTC (${leveraged_size:.2f})")

        elif signal['prediction'] == 'SHORT':
            self.position = {
                'type': 'SHORT',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'consensus': signal['consensus']['SHORT']
            }

            print(f"🔴 REALTIME SHORT at ${current_price:.2f}")
            print(f"   Confidence: {signal['confidence']:.3f} | Consensus: {signal['consensus']['SHORT']:.1%}")
            print(f"   Size: {btc_size:.6f} BTC (${leveraged_size:.2f})")

        self.daily_trades_count += 1
        print(f"📊 Daily trades: {self.daily_trades_count}/{self.max_daily_trades}")
        return True

    def run_realtime_bot(self):
        """Optimized real-time bot loop"""
        if not self.initialize():
            return

        print("\n🤖 PROFESSIONAL REAL-TIME BOT STARTED")
        print("Press Ctrl+C to stop")
        print("=" * 60)

        last_signal_check = 0
        iteration = 0
        start_time = datetime.now()

        try:
            while True:
                current_time = time.time()
                iteration += 1
                self.reset_daily_counters()

                # Update equity curve
                current_drawdown = self.update_equity_curve()
                
                # Show performance every 15 minutes
                if iteration % 30 == 0:  # Every 15 minutes (30 * 30s = 900s)
                    runtime = datetime.now() - start_time
                    print(f"\n📊 Performance Update | Iteration: {iteration} | Runtime: {runtime}")
                    self.show_performance_report()
                    self.show_position_report()
                    print(f"📉 Current Drawdown: {current_drawdown:.2f}%")

                # Check TP/SL every monitor_interval seconds
                if self.position:
                    self.check_tp_sl_realtime()

                # Check for new signals every signal_interval seconds (5 minutes)
                if current_time - last_signal_check >= self.signal_interval and not self.position:
                    last_signal_check = current_time

                    try:
                        data = self.data_manager.fetch_historical_data(limit=150)
                        if data is not None and len(data) >= 100:
                            current_price = data['close'].iloc[-1]
                            current_time_str = datetime.now().strftime('%H:%M:%S')
                            runtime = datetime.now() - start_time

                            print(f"\n🕒 {current_time_str} | Signal Check | Runtime: {runtime}")
                            print("-" * 50)

                            signal_data = self.generate_ml_signal(data)
                            if signal_data:
                                print(f"📊 Price: ${current_price:.2f}")
                                print(f"🎯 Signal: {signal_data['prediction']} (Conf: {signal_data['confidence']:.3f})")
                                
                                self.execute_trade(signal_data, current_price)
                            else:
                                print("❌ No valid signal generated")

                            print(f"💰 Balance: ${self.balance:.2f}")
                        else:
                            print("❌ Insufficient data for signal generation")

                    except Exception as e:
                        print(f"❌ Signal check error: {e}")

                time.sleep(self.monitor_interval)

        except KeyboardInterrupt:
            print(f"\n🛑 Bot stopped after {iteration} iterations")
            runtime = datetime.now() - start_time
            print(f"⏰ Total Runtime: {runtime}")
            self.show_performance_report()
            self.show_position_report()

if __name__ == "__main__":
    bot = ProfessionalRealTimeBot()
    bot.run_realtime_bot()
