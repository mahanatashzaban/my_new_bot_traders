#!/usr/bin/env python3
"""
Quick Fix Bot - Lower thresholds for immediate testing
"""

import time
import pandas as pd
from datetime import datetime
from data_manager_simple import DataManager
from ml_trainer_bidirectional_fixed import BidirectionalMLTrainer
from simple_indicators import SimpleMLIndicatorEngine

class QuickFixBot:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = BidirectionalMLTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = 'BTC/USDT'
        self.position = None
        self.balance = 1000
        self.initial_balance = 1000
        self.trade_log = []
        
        # LOWER thresholds for testing
        self.leverage = 10
        self.tp_percent = 0.008
        self.sl_percent = 0.004
        self.confidence_threshold = 0.55  # LOWER confidence
        self.min_votes = 1  # Only need 1 vote
        
    def initialize(self):
        print("🚀 QUICK FIX BOT - LOWER THRESHOLDS")
        print("=" * 50)
        
        if not self.ml_trainer.load_ml_models():
            print("❌ No trained models found.")
            return False
            
        print("✅ Models Loaded")
        print(f"💰 Balance: ${self.balance:.2f}")
        print(f"⚡ Leverage: {self.leverage}x")
        print(f"🎯 Confidence Threshold: > {self.confidence_threshold} (LOWER)")
        print(f"🤖 Min Votes: {self.min_votes} (LOWER)")
        return True
    
    def execute_trade(self, signal, current_price):
        """Execute trades with lower thresholds"""
        if signal['confidence'] < self.confidence_threshold:
            return False
            
        # Check vote count
        long_votes = signal['vote_count']['LONG']
        short_votes = signal['vote_count']['SHORT']
        
        if long_votes < self.min_votes and short_votes < self.min_votes:
            return False
            
        # Position sizing
        position_size = self.balance * 0.15
        leveraged_size = position_size * self.leverage
        btc_size = leveraged_size / current_price
        
        if long_votes >= self.min_votes and signal['prediction'] == 'LONG' and not self.position:
            # ENTER LONG
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now()
            }
            
            print(f"💚 QUICK LONG at ${current_price:.2f}")
            print(f"       Confidence: {signal['confidence']:.3f}")
            return True
            
        elif short_votes >= self.min_votes and signal['prediction'] == 'SHORT' and not self.position:
            # ENTER SHORT
            self.position = {
                'type': 'SHORT', 
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now()
            }
            
            print(f"🔴 QUICK SHORT at ${current_price:.2f}")
            print(f"       Confidence: {signal['confidence']:.3f}")
            return True
            
        return False
    
    def run_bot(self):
        """Run quick fix bot"""
        if not self.initialize():
            return
            
        print("\n🤖 QUICK FIX BOT STARTED")
        print("With Lower Thresholds for Testing")
        print("=" * 50)
        
        iteration = 0
        while iteration < 12:  # Run for 1 hour
            iteration += 1
            
            try:
                # Get data - use more data for better features
                data = self.data_manager.fetch_historical_data(limit=200)
                if data is None:
                    continue
                    
                current_price = data['close'].iloc[-1]
                current_time = datetime.now().strftime('%H:%M:%S')
                
                print(f"\n🕒 {current_time} | Iter {iteration}/12")
                print("-" * 30)
                
                # Get ML signal
                signal_data = self.ml_trainer.predict_direction(data, self.indicator_engine)
                
                if signal_data:
                    print(f"📊 Price: ${current_price:.2f}")
                    print(f"🎯 Signal: {signal_data['prediction']}")
                    print(f"📈 Confidence: {signal_data['confidence']:.3f}")
                    print(f"🤖 Votes: {signal_data.get('vote_count', {})}")
                    
                    # Try to execute trade
                    self.execute_trade(signal_data, current_price)
                    
                    print(f"💰 Balance: ${self.balance:.2f}")
                else:
                    print(f"📊 Price: ${current_price:.2f}")
                    print("⏳ No signal data")
                
                print("⏰ Next check in 5 minutes...")
                time.sleep(300)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(300)
        
        print(f"\n💰 FINAL BALANCE: ${self.balance:.2f}")

if __name__ == "__main__":
    bot = QuickFixBot()
    bot.run_bot()
