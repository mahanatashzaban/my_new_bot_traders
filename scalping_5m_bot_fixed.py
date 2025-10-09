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
        
        # Scalping parameters
        self.leverage = 10
        self.tp_percent = 0.008  # 0.8%
        self.sl_percent = 0.004  # 0.4%
        self.confidence_threshold = 0.60  # Slightly lower for more signals
        
    def initialize(self):
        print("🚀 5-MINUTE SCALPING BOT WITH FIXED BIDIRECTIONAL ML")
        print("=" * 50)
        
        if not self.ml_trainer.load_ml_models():
            print("❌ No trained models found. Please train first.")
            return False
            
        print("✅ Fixed Bidirectional ML Models Loaded")
        print(f"💰 Initial Balance: ${self.balance:.2f}")
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"🎯 Take Profit: {self.tp_percent*100:.1f}%")
        print(f"🛑 Stop Loss: {self.sl_percent*100:.1f}%")
        print(f"📈 Confidence Threshold: > {self.confidence_threshold}")
        print("🤖 Trading: LONG & SHORT positions")
        return True
    
    def execute_trade(self, signal, current_price):
        """Execute LONG or SHORT trade with leverage"""
        if signal['confidence'] < self.confidence_threshold:
            return False
            
        # Position sizing with leverage
        position_size = self.balance * 0.15  # 15% of balance
        leveraged_size = position_size * self.leverage
        btc_size = leveraged_size / current_price
        
        if signal['prediction'] == 'LONG' and not self.position:
            # ENTER LONG
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'leverage': self.leverage
            }
            
            trade = {
                'action': 'LONG',
                'price': current_price,
                'size_btc': btc_size,
                'size_usd': leveraged_size,
                'confidence': signal['confidence'],
                'timestamp': datetime.now(),
                'leverage': self.leverage
            }
            self.trade_log.append(trade)
            
            print(f"💚 SCALP LONG: {btc_size:.6f} BTC at ${current_price:.2f}")
            print(f"       Leverage: {self.leverage}x | Position: ${leveraged_size:.2f}")
            return True
            
        elif signal['prediction'] == 'SHORT' and not self.position:
            # ENTER SHORT
            self.position = {
                'type': 'SHORT', 
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'leverage': self.leverage
            }
            
            trade = {
                'action': 'SHORT',
                'price': current_price,
                'size_btc': btc_size,
                'size_usd': leveraged_size,
                'confidence': signal['confidence'],
                'timestamp': datetime.now(),
                'leverage': self.leverage
            }
            self.trade_log.append(trade)
            
            print(f"🔴 SCALP SHORT: {btc_size:.6f} BTC at ${current_price:.2f}")
            print(f"       Leverage: {self.leverage}x | Position: ${leveraged_size:.2f}")
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
                
                self.balance += pnl_amount
                
                trade = {
                    'action': 'EXIT_' + self.position['type'],
                    'price': current_price,
                    'pnl_percent': pnl_percent,
                    'pnl_amount': pnl_amount,
                    'timestamp': datetime.now(),
                    'exit_reason': exit_reason
                }
                self.trade_log.append(trade)
                
                pnl_color = "🟢" if pnl_percent > 0 else "🔴"
                print(f"{pnl_color} EXIT {self.position['type']}: {exit_reason}")
                print(f"       PnL: {pnl_percent*100:+.2f}% | Amount: ${pnl_amount:+.2f}")
                
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
            self.balance += pnl_amount
            trade = {
                'action': 'TP_' + self.position['type'],
                'price': current_price,
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'timestamp': datetime.now(),
                'exit_reason': 'TAKE PROFIT'
            }
            self.trade_log.append(trade)
            
            print(f"🎯 TAKE PROFIT {self.position['type']}: {pnl_percent*100:+.2f}%")
            self.position = None
            return True
            
        elif pnl_percent <= -self.sl_percent:
            self.balance += pnl_amount
            trade = {
                'action': 'SL_' + self.position['type'],
                'price': current_price,
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'timestamp': datetime.now(),
                'exit_reason': 'STOP LOSS'
            }
            self.trade_log.append(trade)
            
            print(f"🛑 STOP LOSS {self.position['type']}: {pnl_percent*100:+.2f}%")
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
                    
                    print(f"💰 Balance: ${self.balance:.2f}")
                else:
                    print(f"📊 Price: ${current_price:.2f}")
                    print("⏳ No trading signal")
                    if self.position:
                        if self.position['type'] == 'LONG':
                            pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
                        else:
                            pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price']
                        pnl_percent *= self.leverage
                        pnl_color = "🟢" if pnl_percent > 0 else "🔴"
                        print(f"📈 Position: {self.position['type']} | PnL: {pnl_color}{pnl_percent*100:+.2f}%")
                
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
        
        if self.trade_log:
            winning_trades = [t for t in self.trade_log if 'pnl_percent' in t and t['pnl_percent'] > 0]
            win_rate = len(winning_trades) / len(self.trade_log) * 100
            
            print(f"🎯 WIN RATE: {win_rate:.1f}% ({len(winning_trades)}/{len(self.trade_log)})")
            
            long_trades = [t for t in self.trade_log if t.get('action', '').startswith('LONG')]
            short_trades = [t for t in self.trade_log if t.get('action', '').startswith('SHORT')]
            
            print(f"📈 LONG Trades: {len(long_trades)}")
            print(f"📉 SHORT Trades: {len(short_trades)}")

if __name__ == "__main__":
    bot = Scalping5mBotFixed()
    bot.run_scalping()
