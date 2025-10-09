#!/usr/bin/env python3
"""
Live ML Trading Bot - Real Trading Version
"""

import time
import pandas as pd
from datetime import datetime
from data_manager import DataManager
from ml_trainer import MLStrategyTrainer
from simple_indicators import SimpleMLIndicatorEngine
from config import TRADING_CONFIG

class LiveMLTrader:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = MLStrategyTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = TRADING_CONFIG['symbol']
        self.position = None
        self.balance = TRADING_CONFIG['initial_balance']
        self.trade_log = []
        
    def initialize(self):
        """Initialize the live trader"""
        print("🚀 INITIALIZING LIVE ML TRADER")
        print("=" * 50)
        
        if not self.ml_trainer.load_ml_models():
            return False
            
        print("✅ ML Models Loaded")
        print("💰 Initial Balance: ${:.2f}".format(self.balance))
        print("🌐 Exchange: Binance")
        print("📈 Symbol: BTC/USDT")
        print("⏰ Timeframe: 1 minute")
        return True
    
    def execute_trade(self, signal):
        """Execute a trade based on ML signal"""
        if signal['confidence'] < 0.65:
            return  # Only trade on high confidence
            
        current_price = signal['current_price']
        
        if signal['prediction'] == 'UP' and not self.position:
            # BUY Logic
            position_size = self.balance * 0.1  # 10% of balance
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': position_size / current_price,
                'entry_time': datetime.now(),
                'confidence': signal['confidence']
            }
            self.balance -= position_size
            
            trade = {
                'action': 'BUY',
                'price': current_price,
                'size': position_size / current_price,
                'confidence': signal['confidence'],
                'timestamp': datetime.now(),
                'balance': self.balance
            }
            self.trade_log.append(trade)
            
            print("💚 LIVE: BUY at ${:.2f} | Conf: {:.3f}".format(
                current_price, signal['confidence']))
                
        elif signal['prediction'] == 'DOWN' and self.position and self.position['type'] == 'LONG':
            # SELL Logic
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
                'balance': self.balance
            }
            self.trade_log.append(trade)
            
            pnl_color = "🟢" if pnl_percent > 0 else "🔴"
            print("{} LIVE: SELL at ${:.2f} | PnL: {:.2f}% | Balance: ${:.2f}".format(
                pnl_color, current_price, pnl_percent*100, self.balance))
                
            self.position = None
    
    def run_live_trading(self):
        """Run live trading simulation"""
        if not self.initialize():
            return
            
        print("\n🤖 LIVE TRADING STARTED")
        print("Trading with HIGH CONFIDENCE signals only (>0.65)")
        print("=" * 50)
        
        iteration = 0
        while iteration < 50:  # Run for 50 iterations (about 50 minutes)
            iteration += 1
            
            try:
                # Get market data
                data = self.data_manager.fetch_historical_data(limit=500)
                if data is None or data.empty:
                    print("❌ No data received, waiting...")
                    time.sleep(30)
                    continue
                    
                # Get ML signal
                signal_data = self.ml_trainer.predict_direction(data, self.indicator_engine)
                
                if signal_data:
                    signal = {
                        'prediction': signal_data['prediction'],
                        'confidence': signal_data['confidence'],
                        'current_price': data['close'].iloc[-1],
                        'timestamp': datetime.now(),
                        'model_votes': signal_data.get('model_predictions', {})
                    }
                    
                    # Display market info
                    current_time = datetime.now().strftime('%H:%M:%S')
                    print("\n📊 {} | Price: ${:.2f} | Signal: {} | Conf: {:.3f}".format(
                        current_time,
                        signal['current_price'],
                        signal['prediction'],
                        signal['confidence']
                    ))
                    
                    # Show model votes
                    if 'model_votes' in signal and signal['model_votes']:
                        print("   🤖 Model Votes: {}".format(signal['model_votes']))
                    
                    # Execute trading logic
                    self.execute_trade(signal)
                    
                    # Show position status
                    if self.position:
                        current_pnl = (signal['current_price'] - self.position['entry_price']) / self.position['entry_price']
                        pnl_color = "🟢" if current_pnl > 0 else "🔴"
                        print("   📈 Position: LONG | PnL: {}{:.2f}%{}".format(
                            pnl_color, current_pnl*100, "🔴" if current_pnl < 0 else ""))
                    else:
                        print("   💰 No Position | Balance: ${:.2f}".format(self.balance))
                else:
                    print("⏳ {} | Price: ${:.2f} | No signal".format(
                        datetime.now().strftime('%H:%M:%S'),
                        data['close'].iloc[-1] if not data.empty else 0
                    ))
                
                print("   ⏰ Next check in 60 seconds...")
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                time.sleep(30)  # Wait before retrying
        
        # Final summary
        print("\n" + "=" * 50)
        print("🎯 LIVE TRADING COMPLETED")
        print("=" * 50)
        print("Final Balance: ${:.2f}".format(self.balance))
        
        initial_balance = TRADING_CONFIG['initial_balance']
        total_return = (self.balance - initial_balance) / initial_balance * 100
        print("Total Return: {:.2f}%".format(total_return))
        print("Total Trades: {}".format(len(self.trade_log)))
        
        if self.trade_log:
            winning_trades = [t for t in self.trade_log if 'pnl_percent' in t and t['pnl_percent'] > 0]
            win_rate = len(winning_trades) / len(self.trade_log) * 100
            print("Winning Trades: {}/{} ({:.1f}%)".format(
                len(winning_trades), len(self.trade_log), win_rate))
            
            if winning_trades:
                avg_win = sum(t['pnl_percent'] for t in winning_trades) / len(winning_trades) * 100
                print("Average Win: {:.2f}%".format(avg_win))
        
        print("\n💾 Models Used:")
        print("   - XGBoost (xgb_ml_model.pkl)")
        print("   - Random Forest (rf_ml_model.pkl)")
        print("   - Gradient Boosting (gb_ml_model.pkl)")
        print("   - Feature Scaler (ml_scaler.pkl)")

if __name__ == "__main__":
    print("🚀 ML LIVE TRADING BOT")
    print("Trained Models: XGBoost, Random Forest, Gradient Boosting")
    print("Exchange: Binance | Symbol: BTC/USDT")
    print("=" * 60)
    
    trader = LiveMLTrader()
    trader.run_live_trading()
