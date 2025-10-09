#!/usr/bin/env python3
"""
5-Minute Live Trading Bot
"""

import time
import pandas as pd
from datetime import datetime
from data_manager_simple import DataManager
from ml_trainer_simple import SimpleMLTrainer
from simple_indicators import SimpleMLIndicatorEngine
from simple_config import TRADING_CONFIG

class FiveMinuteTrader:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = SimpleMLTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = TRADING_CONFIG['symbol']
        self.position = None
        self.balance = TRADING_CONFIG['initial_balance']
        self.initial_balance = TRADING_CONFIG['initial_balance']
        self.trade_log = []
        self.iteration = 0
        
    def initialize(self):
        print("🚀 5-MINUTE LIVE TRADING BOT")
        print("=" * 50)
        
        if not self.ml_trainer.load_ml_models():
            return False
            
        print("✅ 5-minute ML Models Loaded")
        print("💰 Initial Balance: ${:.2f}".format(self.balance))
        print("⏰ Timeframe: 5 minutes")
        print("🎯 Confidence Threshold: > 0.60")
        print("📊 Models: Random Forest + Gradient Boosting")
        return True
    
    def execute_trade(self, signal):
        """Execute trade with 10% position sizing"""
        if signal['confidence'] < 0.60:  # Slightly lower threshold for 5m
            return False
            
        current_price = signal['current_price']
        
        if signal['prediction'] == 'UP' and not self.position:
            # BUY - 10% position
            position_size = self.balance * 0.10
            btc_size = position_size / current_price
            
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence']
            }
            self.balance -= position_size
            
            trade = {
                'action': 'BUY',
                'price': current_price,
                'size_btc': btc_size,
                'size_usd': position_size,
                'confidence': signal['confidence'],
                'timestamp': datetime.now()
            }
            self.trade_log.append(trade)
            
            print("💚 5m BUY: {:.6f} BTC at ${:.2f}".format(btc_size, current_price))
            print("       Position: ${:.2f} | Confidence: {:.3f}".format(position_size, signal['confidence']))
            return True
                
        elif signal['prediction'] == 'DOWN' and self.position and self.position['type'] == 'LONG':
            # SELL
            position_value = self.position['size'] * current_price
            pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
            pnl_amount = position_value * pnl_percent
            
            self.balance += position_value + pnl_amount
            
            trade = {
                'action': 'SELL',
                'price': current_price,
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'timestamp': datetime.now(),
                'hold_time': (datetime.now() - self.position['entry_time']).total_seconds() / 60
            }
            self.trade_log.append(trade)
            
            pnl_color = "🟢" if pnl_percent > 0 else "🔴"
            print("{} 5m SELL | PnL: {:.2f}% | Amount: ${:.2f}".format(
                pnl_color, pnl_percent*100, pnl_amount))
                
            self.position = None
            return True
            
        return False
    
    def run_5m_trading(self):
        """Run 5-minute trading session"""
        if not self.initialize():
            return
            
        print("\n🤖 5-MINUTE TRADING STARTED")
        print("Will check for signals every 5 minutes")
        print("=" * 50)
        
        self.iteration = 0
        while self.iteration < 24:  # Run for 2 hours (24 * 5min)
            self.iteration += 1
            
            try:
                # Get 5-minute data
                data = self.data_manager.fetch_historical_data(limit=200)
                if data is None or data.empty:
                    print("❌ No data, waiting 5 minutes...")
                    time.sleep(300)
                    continue
                    
                current_price = data['close'].iloc[-1]
                current_time = datetime.now().strftime('%H:%M:%S')
                
                print(f"\n🕒 {current_time} | Iteration {self.iteration}/24")
                print("-" * 40)
                
                # Get ML signal
                signal_data = self.ml_trainer.predict_direction(data, self.indicator_engine)
                
                if signal_data:
                    print(f"📊 Price: ${current_price:.2f}")
                    print(f"🎯 Signal: {signal_data['prediction']}")
                    print(f"📈 Confidence: {signal_data['confidence']:.3f}")
                    print(f"🤖 Models: {signal_data['model_predictions']}")
                    
                    # Execute trading logic
                    trade_executed = self.execute_trade({
                        'prediction': signal_data['prediction'],
                        'confidence': signal_data['confidence'],
                        'current_price': current_price
                    })
                    
                    # Position status
                    if self.position:
                        current_pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
                        pnl_color = "🟢" if current_pnl > 0 else "🔴"
                        print(f"📈 Position: LONG | PnL: {pnl_color}{current_pnl*100:.2f}%")
                    else:
                        print("💤 No Position")
                    
                    print(f"💰 Balance: ${self.balance:.2f}")
                else:
                    print(f"📊 Price: ${current_price:.2f}")
                    print("⏳ No trading signal")
                    if self.position:
                        current_pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
                        pnl_color = "🟢" if current_pnl > 0 else "🔴"
                        print(f"📈 Position: LONG | PnL: {pnl_color}{current_pnl*100:.2f}%")
                
                print("⏰ Next check in 5 minutes...")
                time.sleep(300)  # Wait 5 minutes
                
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
        
        # Final summary
        self.show_final_summary()
    
    def show_final_summary(self):
        """Show trading session summary"""
        print("\n" + "=" * 50)
        print("🎯 5-MINUTE TRADING SESSION COMPLETE")
        print("=" * 50)
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        print(f"💰 FINAL BALANCE: ${self.balance:.2f}")
        print(f"📈 TOTAL RETURN: {total_return:+.2f}%")
        print(f"🔢 TOTAL TRADES: {len(self.trade_log)}")
        
        if self.trade_log:
            winning_trades = [t for t in self.trade_log if 'pnl_percent' in t and t['pnl_percent'] > 0]
            win_rate = len(winning_trades) / len(self.trade_log) * 100 if self.trade_log else 0
            
            print(f"🎯 WIN RATE: {win_rate:.1f}% ({len(winning_trades)}/{len(self.trade_log)})")
            
            if winning_trades:
                avg_win = sum(t['pnl_percent'] for t in winning_trades) / len(winning_trades) * 100
                print(f"📊 AVERAGE WIN: {avg_win:+.2f}%")
        
        print(f"\n⏰ TIMEFRAME: 5 minutes")
        print(f"🤖 MODELS: Random Forest + Gradient Boosting")
        print(f"🎯 CONFIDENCE THRESHOLD: > 0.60")

if __name__ == "__main__":
    trader = FiveMinuteTrader()
    trader.run_5m_trading()
