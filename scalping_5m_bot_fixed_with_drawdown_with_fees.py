#!/usr/bin/env python3
"""
5-Minute Scalping Bot with Fixed Bidirectional ML
"""

import time
import pandas as pd
from datetime import datetime
from data_manager_simple import DataManager
from ml_trainer_bidirectional_fixed import BidirectionalMLTrainer
from simple_indicators import SimpleMLIndicatorEngine

class Scalping5mBotFixed:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = BidirectionalMLTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = 'BTC/USDT'
        self.position = None
        self.balance = 100
        self.initial_balance = 100
        self.trade_log = []
        self.peak_balance = 100  # Track peak for drawdown calculation
        self.max_drawdown = 0  # Track maximum drawdown during session
        self.total_fees_paid = 0.0  # Track total fees paid

        # Scalping parameters
        self.leverage = 10
        self.tp_percent = 0.012  # 0.8%
        self.sl_percent = 0.006 # 0.4%
        self.confidence_threshold = 0.60  # Slightly lower for more signals
        self.position_size = 0.2  # 20% of balance per trade

        # Trading fees (Binance spot trading fees) - ENABLED
        self.maker_fee = 0.0002  # 0.1%
        self.taker_fee = 0.0005  # 0.2%

    def initialize(self):
        print("🚀 5-MINUTE SCALPING BOT WITH FIXED BIDIRECTIONAL ML")
        print("=" * 50)

        if not self.ml_trainer.load_ml_models():
            print("❌ No trained models found. Please train first.")
            return False

        print("✅ Fixed Bidirectional ML Models Loaded")
        print(f"💰 Initial Balance: ${self.balance:.2f}")
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"📊 Position Size: {self.position_size*100}% of balance")
        print(f"🎯 Take Profit: {self.tp_percent*100:.1f}%")
        print(f"🛑 Stop Loss: {self.sl_percent*100:.1f}%")
        print(f"📈 Confidence Threshold: > {self.confidence_threshold}")
        print(f"💰 Trading Fees: {self.taker_fee*100:.1f}% (taker) / {self.maker_fee*100:.1f}% (maker)")
        print("🤖 Trading: LONG & SHORT positions")
        return True

    def execute_trade(self, signal, current_price):
        """Execute LONG or SHORT trade with leverage"""
        if signal['confidence'] < self.confidence_threshold:
            return False

        # Position sizing with leverage - using 20% of balance
        position_size = self.balance * self.position_size
        leveraged_size = position_size * self.leverage
        btc_size = leveraged_size / current_price

        # Calculate trading fee
        entry_fee = leveraged_size * self.taker_fee

        # Check if we have enough balance for the fee
        if entry_fee >= self.balance:
            print(f"❌ Insufficient balance for entry fee: ${entry_fee:.4f}")
            return False

        if signal['prediction'] == 'LONG' and not self.position:
            # ENTER LONG
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'leverage': self.leverage,
                'entry_fee': entry_fee
            }

            # Apply entry fee
            self.balance -= entry_fee
            self.total_fees_paid += entry_fee

            trade = {
                'action': 'LONG',
                'price': current_price,
                'size_btc': btc_size,
                'size_usd': leveraged_size,
                'confidence': signal['confidence'],
                'timestamp': datetime.now(),
                'leverage': self.leverage,
                'fee': entry_fee
            }
            self.trade_log.append(trade)

            print(f"💚 SCALP LONG: {btc_size:.6f} BTC at ${current_price:.2f}")
            print(f"       Leverage: {self.leverage}x | Position: ${leveraged_size:.2f}")
            print(f"       Entry Fee: ${entry_fee:.4f}")
            return True

        elif signal['prediction'] == 'SHORT' and not self.position:
            # ENTER SHORT
            self.position = {
                'type': 'SHORT',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'leverage': self.leverage,
                'entry_fee': entry_fee
            }

            # Apply entry fee
            self.balance -= entry_fee
            self.total_fees_paid += entry_fee

            trade = {
                'action': 'SHORT',
                'price': current_price,
                'size_btc': btc_size,
                'size_usd': leveraged_size,
                'confidence': signal['confidence'],
                'timestamp': datetime.now(),
                'leverage': self.leverage,
                'fee': entry_fee
            }
            self.trade_log.append(trade)

            print(f"🔴 SCALP SHORT: {btc_size:.6f} BTC at ${current_price:.2f}")
            print(f"       Leverage: {self.leverage}x | Position: ${leveraged_size:.2f}")
            print(f"       Entry Fee: ${entry_fee:.4f}")
            return True

        elif self.position and signal['prediction'] in ['LONG', 'SHORT']:
            # EXIT logic - close position on opposite signal
            should_exit = False

            if self.position['type'] == 'LONG' and signal['prediction'] == 'SHORT':
                should_exit = True
                exit_reason = "ML REVERSAL to SHORT"
            elif self.position['type'] == 'SHORT' and signal['prediction'] == 'LONG':
                should_exit = True
                exit_reason = "ML REVERSAL to LONG"

            if should_exit:
                position_value = self.position['size'] * current_price
                if self.position['type'] == 'LONG':
                    pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
                else:  # SHORT
                    pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price']

                # Apply leverage to PnL
                pnl_percent *= self.leverage
                pnl_amount = position_value * pnl_percent

                # Calculate exit fee
                exit_fee = position_value * self.taker_fee
                total_fee = self.position.get('entry_fee', 0) + exit_fee

                # Apply PnL and deduct exit fee
                self.balance += pnl_amount
                self.balance -= exit_fee
                self.total_fees_paid += exit_fee

                trade = {
                    'action': 'EXIT_' + self.position['type'],
                    'price': current_price,
                    'pnl_percent': pnl_percent,
                    'pnl_amount': pnl_amount,
                    'timestamp': datetime.now(),
                    'exit_reason': exit_reason,
                    'exit_fee': exit_fee,
                    'total_fee': total_fee
                }
                self.trade_log.append(trade)

                pnl_color = "🟢" if pnl_percent > 0 else "🔴"
                print(f"{pnl_color} EXIT {self.position['type']}: {exit_reason}")
                print(f"       PnL: {pnl_percent*100:+.2f}% | Amount: ${pnl_amount:+.2f}")
                print(f"       Exit Fee: ${exit_fee:.4f} | Total Fees: ${total_fee:.4f}")

                self.position = None
                return True

        return False

    def check_tp_sl(self, current_price):
        """Check Take Profit and Stop Loss conditions"""
        if not self.position:
            return False

        if self.position['type'] == 'LONG':
            pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
        else:  # SHORT
            pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price']

        # Apply leverage to PnL
        pnl_percent *= self.leverage

        position_value = self.position['size'] * current_price
        pnl_amount = position_value * pnl_percent

        # Check TP/SL
        if pnl_percent >= self.tp_percent:
            # Calculate exit fee
            exit_fee = position_value * self.taker_fee
            total_fee = self.position.get('entry_fee', 0) + exit_fee

            # Apply PnL and deduct exit fee
            self.balance += pnl_amount
            self.balance -= exit_fee
            self.total_fees_paid += exit_fee

            trade = {
                'action': 'TP_' + self.position['type'],
                'price': current_price,
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'timestamp': datetime.now(),
                'exit_reason': 'TAKE PROFIT',
                'exit_fee': exit_fee,
                'total_fee': total_fee
            }
            self.trade_log.append(trade)

            print(f"🎯 TAKE PROFIT {self.position['type']}: {pnl_percent*100:+.2f}%")
            print(f"       Exit Fee: ${exit_fee:.4f} | Total Fees: ${total_fee:.4f}")
            self.position = None
            return True

        elif pnl_percent <= -self.sl_percent:
            # Calculate exit fee
            exit_fee = position_value * self.taker_fee
            total_fee = self.position.get('entry_fee', 0) + exit_fee

            # Apply PnL and deduct exit fee
            self.balance += pnl_amount
            self.balance -= exit_fee
            self.total_fees_paid += exit_fee

            trade = {
                'action': 'SL_' + self.position['type'],
                'price': current_price,
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'timestamp': datetime.now(),
                'exit_reason': 'STOP LOSS',
                'exit_fee': exit_fee,
                'total_fee': total_fee
            }
            self.trade_log.append(trade)

            print(f"🛑 STOP LOSS {self.position['type']}: {pnl_percent*100:+.2f}%")
            print(f"       Exit Fee: ${exit_fee:.4f} | Total Fees: ${total_fee:.4f}")
            self.position = None
            return True

        return False

    def run_scalping(self):
        """Run scalping bot"""
        if not self.initialize():
            return

        print("\n🤖 5-MINUTE SCALPING STARTED")
        print("Trading LONG & SHORT with 10x Leverage")
        print("=" * 50)

        iteration = 0
        while iteration < 48:  # Run for 4 hours
            iteration += 1

            try:
                # Get 5-minute data
                data = self.data_manager.fetch_historical_data(limit=300)
                if data is None or data.empty:
                    print("❌ No data, waiting 5 minutes...")
                    time.sleep(300)
                    continue

                current_price = data['close'].iloc[-1]
                current_time = datetime.now().strftime('%H:%M:%S')

                print(f"\n🕒 {current_time} | Iteration {iteration}/48")
                print("-" * 40)

                # Check TP/SL first
                if self.check_tp_sl(current_price):
                    # Position was closed by TP/SL, continue to next iteration
                    time.sleep(300)
                    continue

                # Get ML signal
                signal_data = self.ml_trainer.predict_direction(data, self.indicator_engine)

                if signal_data:
                    print(f"📊 Price: ${current_price:.2f}")
                    print(f"🎯 Signal: {signal_data['prediction']}")
                    print(f"📈 Confidence: {signal_data['confidence']:.3f}")
                    print(f"🤖 Votes: {signal_data.get('vote_count', {})}")
                    print(f"💰 Total Fees Paid: ${self.total_fees_paid:.4f}")

                    # Execute trading logic
                    trade_executed = self.execute_trade(signal_data, current_price)

                    # Position status
                    if self.position:
                        if self.position['type'] == 'LONG':
                            pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
                        else:
                            pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price']

                        pnl_percent *= self.leverage
                        pnl_color = "🟢" if pnl_percent > 0 else "🔴"
                        print(f"📈 Position: {self.position['type']} | PnL: {pnl_color}{pnl_percent*100:+.2f}%")

                    # Update peak balance and calculate drawdown
                    if self.balance > self.peak_balance:
                        self.peak_balance = self.balance

                    current_drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
                    if current_drawdown > self.max_drawdown:
                        self.max_drawdown = current_drawdown

                    drawdown_color = "🔴" if current_drawdown > 0 else "🟢"

                    print(f"💰 Balance: ${self.balance:.2f}")
                    print(f"📉 Current Drawdown: {drawdown_color}{current_drawdown:.2f}%")
                    print(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")
                else:
                    print(f"📊 Price: ${current_price:.2f}")
                    print("⏳ No trading signal")
                    print(f"💰 Total Fees Paid: ${self.total_fees_paid:.4f}")
                    
                    if self.position:
                        if self.position['type'] == 'LONG':
                            pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
                        else:
                            pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price']
                        pnl_percent *= self.leverage
                        pnl_color = "🟢" if pnl_percent > 0 else "🔴"
                        print(f"📈 Position: {self.position['type']} | PnL: {pnl_color}{pnl_percent*100:+.2f}%")

                    # Update peak balance and calculate drawdown even when no signal
                    if self.balance > self.peak_balance:
                        self.peak_balance = self.balance

                    current_drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
                    if current_drawdown > self.max_drawdown:
                        self.max_drawdown = current_drawdown

                    drawdown_color = "🔴" if current_drawdown > 0 else "🟢"
                    print(f"💰 Balance: ${self.balance:.2f}")
                    print(f"📉 Current Drawdown: {drawdown_color}{current_drawdown:.2f}%")
                    print(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")

                print("⏰ Next check in 5 minutes...")
                time.sleep(300)

            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(300)

        self.show_final_summary()

    def show_final_summary(self):
        """Show trading session summary"""
        print("\n" + "=" * 50)
        print("🎯 SCALPING SESSION COMPLETE")
        print("=" * 50)

        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100

        print(f"💰 FINAL BALANCE: ${self.balance:.2f}")
        print(f"📈 TOTAL RETURN: {total_return:+.2f}%")
        print(f"🔢 TOTAL TRADES: {len(self.trade_log)}")
        print(f"💰 TOTAL FEES PAID: ${self.total_fees_paid:.4f}")

        if self.trade_log:
            winning_trades = [t for t in self.trade_log if 'pnl_percent' in t and t['pnl_percent'] > 0]
            win_rate = len(winning_trades) / len(self.trade_log) * 100

            print(f"🎯 WIN RATE: {win_rate:.1f}% ({len(winning_trades)}/{len(self.trade_log)})")

            long_trades = [t for t in self.trade_log if t.get('action', '').startswith('LONG')]
            short_trades = [t for t in self.trade_log if t.get('action', '').startswith('SHORT')]

            print(f"📈 LONG Trades: {len(long_trades)}")
            print(f"📉 SHORT Trades: {len(short_trades)}")

            # Calculate fee breakdown
            entry_fees = sum(trade.get('fee', 0) for trade in self.trade_log if 'fee' in trade)
            exit_fees = sum(trade.get('exit_fee', 0) for trade in self.trade_log if 'exit_fee' in trade)
            print(f"💰 Entry Fees: ${entry_fees:.4f}")
            print(f"💰 Exit Fees: ${exit_fees:.4f}")

        # Final drawdown calculation
        print(f"📉 MAX DRAWDOWN: {self.max_drawdown:.2f}%")

        # Calculate net return after fees
        net_return_after_fees = total_return - (self.total_fees_paid / self.initial_balance * 100)
        print(f"📊 NET RETURN (after fees): {net_return_after_fees:+.2f}%")

if __name__ == "__main__":
    bot = Scalping5mBotFixed()
    bot.run_scalping()
