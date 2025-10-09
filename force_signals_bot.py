#!/usr/bin/env python3
"""
Force Signals Bot - Makes Conservative Models Give Trading Signals
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime
from data_manager_simple import DataManager
from ml_trainer_bidirectional_fixed import BidirectionalMLTrainer
from simple_indicators import SimpleMLIndicatorEngine

class ForceSignalsBot:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = BidirectionalMLTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = 'BTC/USDT'
        self.position = None
        self.balance = 1000
        self.initial_balance = 1000
        self.trade_log = []
        
        # Aggressive parameters
        self.leverage = 10
        self.tp_percent = 0.008
        self.sl_percent = 0.004
        
    def initialize(self):
        print("🚀 FORCE SIGNALS BOT - AGGRESSIVE MODE")
        print("=" * 50)
        
        if not self.ml_trainer.load_ml_models():
            print("❌ No trained models found.")
            return False
            
        print("✅ Models Loaded - Forcing Aggressive Signals")
        print(f"💰 Balance: ${self.balance:.2f}")
        print(f"⚡ Leverage: {self.leverage}x")
        print("🎯 Strategy: Force LONG/SHORT from probability scores")
        return True
    
    def force_prediction_from_probabilities(self, df):
        """Force trading signals from model probabilities instead of predictions"""
        if not self.ml_trainer.models:
            return None

        # Prepare features
        X, _ = self.ml_trainer.prepare_features(df, self.indicator_engine)
        if X is None:
            return None

        # Scale features
        try:
            X_scaled = self.ml_trainer.scaler.transform(X)
        except Exception as e:
            print(f"❌ Error scaling features: {e}")
            return None

        # Get probabilities from both models
        all_probabilities = []
        
        for name, model in self.ml_trainer.models.items():
            try:
                probabilities = model.predict_proba(X_scaled)[-1]  # [SHORT, HOLD, LONG]
                all_probabilities.append(probabilities)
                print(f"📊 {name} probabilities: SHORT:{probabilities[0]:.3f}, HOLD:{probabilities[1]:.3f}, LONG:{probabilities[2]:.3f}")
            except Exception as e:
                print(f"❌ Probability error with {name}: {e}")
                continue

        if not all_probabilities:
            return None

        # Average probabilities across models
        avg_probabilities = np.mean(all_probabilities, axis=0)
        short_prob, hold_prob, long_prob = avg_probabilities
        
        print(f"📊 AVG Probabilities: SHORT:{short_prob:.3f}, HOLD:{hold_prob:.3f}, LONG:{long_prob:.3f}")
        
        # Force signal based on probabilities (ignore HOLD)
        if long_prob > short_prob and long_prob > 0.4:  # Lower threshold
            return {
                'prediction': 'LONG',
                'confidence': long_prob,
                'probabilities': {'SHORT': short_prob, 'HOLD': hold_prob, 'LONG': long_prob},
                'reason': f'LONG probability {long_prob:.3f} > SHORT {short_prob:.3f}'
            }
        elif short_prob > long_prob and short_prob > 0.4:  # Lower threshold
            return {
                'prediction': 'SHORT', 
                'confidence': short_prob,
                'probabilities': {'SHORT': short_prob, 'HOLD': hold_prob, 'LONG': long_prob},
                'reason': f'SHORT probability {short_prob:.3f} > LONG {long_prob:.3f}'
            }
        else:
            return {
                'prediction': 'HOLD',
                'confidence': hold_prob,
                'probabilities': {'SHORT': short_prob, 'HOLD': hold_prob, 'LONG': long_prob},
                'reason': 'No clear directional bias'
            }
    
    def get_technical_bias(self, data):
        """Get technical bias to confirm ML signals"""
        try:
            # Simple RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Price momentum (last 3 candles)
            current_price = data['close'].iloc[-1]
            price_3_candles_ago = data['close'].iloc[-4]
            momentum = (current_price - price_3_candles_ago) / price_3_candles_ago * 100
            
            # Volume trend
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].tail(10).mean()
            volume_ratio = current_volume / avg_volume
            
            print(f"📈 Technicals - RSI: {rsi:.1f}, Momentum: {momentum:+.2f}%, Volume: {volume_ratio:.2f}x")
            
            # Generate technical bias
            if rsi < 40 and momentum > 0:
                return 'LONG'
            elif rsi > 60 and momentum < 0:
                return 'SHORT'
            elif volume_ratio > 1.2 and momentum > 0.1:
                return 'LONG'
            elif volume_ratio > 1.2 and momentum < -0.1:
                return 'SHORT'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            print(f"❌ Technical bias error: {e}")
            return 'NEUTRAL'
    
    def execute_trade(self, signal, technical_bias, current_price):
        """Execute trade with technical confirmation"""
        # Only trade if we have a clear signal
        if signal['prediction'] == 'HOLD':
            return False
            
        # Check if technicals confirm
        if technical_bias != signal['prediction'] and technical_bias != 'NEUTRAL':
            print(f"⚠️  ML {signal['prediction']} but Technicals {technical_bias} - Skipping")
            return False
            
        # Position sizing
        position_size = self.balance * 0.15
        leveraged_size = position_size * self.leverage
        btc_size = leveraged_size / current_price
        
        if signal['prediction'] == 'LONG' and not self.position:
            # ENTER LONG
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence']
            }
            
            print(f"💚 FORCED LONG at ${current_price:.2f}")
            print(f"       Confidence: {signal['confidence']:.3f}")
            print(f"       Reason: {signal['reason']}")
            return True
            
        elif signal['prediction'] == 'SHORT' and not self.position:
            # ENTER SHORT
            self.position = {
                'type': 'SHORT', 
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence']
            }
            
            print(f"🔴 FORCED SHORT at ${current_price:.2f}")
            print(f"       Confidence: {signal['confidence']:.3f}")
            print(f"       Reason: {signal['reason']}")
            return True
            
        return False
    
    def check_tp_sl(self, current_price):
        """Check Take Profit and Stop Loss"""
        if not self.position:
            return False
            
        if self.position['type'] == 'LONG':
            pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
        else:  # SHORT
            pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price']
        
        # Apply leverage
        pnl_percent *= self.leverage
        
        if pnl_percent >= self.tp_percent:
            pnl_amount = self.balance * 0.15 * pnl_percent
            self.balance += pnl_amount
            print(f"🎯 TP {self.position['type']}: {pnl_percent*100:+.2f}%")
            self.position = None
            return True
        elif pnl_percent <= -self.sl_percent:
            pnl_amount = self.balance * 0.15 * pnl_percent
            self.balance += pnl_amount
            print(f"🛑 SL {self.position['type']}: {pnl_percent*100:+.2f}%")
            self.position = None
            return True
            
        return False
    
    def run_bot(self):
        """Run force signals bot"""
        if not self.initialize():
            return
            
        print("\n🤖 FORCE SIGNALS BOT STARTED")
        print("Using Probability Scores Instead of Predictions")
        print("=" * 50)
        
        iteration = 0
        while iteration < 12:  # Run for 1 hour
            iteration += 1
            
            try:
                # Get data
                data = self.data_manager.fetch_historical_data(limit=150)
                if data is None:
                    continue
                    
                current_price = data['close'].iloc[-1]
                current_time = datetime.now().strftime('%H:%M:%S')
                
                print(f"\n🕒 {current_time} | Iter {iteration}/12")
                print("-" * 40)
                
                # Check TP/SL first
                if self.check_tp_sl(current_price):
                    time.sleep(300)
                    continue
                
                # Get forced signal from probabilities
                signal_data = self.force_prediction_from_probabilities(data)
                
                # Get technical bias
                technical_bias = self.get_technical_bias(data)
                
                if signal_data:
                    print(f"📊 Price: ${current_price:.2f}")
                    print(f"🎯 ML Signal: {signal_data['prediction']}")
                    
                    # Execute trade
                    self.execute_trade(signal_data, technical_bias, current_price)
                    
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
        print(f"📈 TOTAL RETURN: {(self.balance - self.initial_balance) / self.initial_balance * 100:+.2f}%")

if __name__ == "__main__":
    bot = ForceSignalsBot()
    bot.run_bot()
