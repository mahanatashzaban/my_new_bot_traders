#!/usr/bin/env python3
"""
Enhanced Live ML Trading Bot - With Full Metrics
"""

import time
import pandas as pd
from datetime import datetime
from data_manager import DataManager
from ml_trainer import MLStrategyTrainer
from simple_indicators import SimpleMLIndicatorEngine
from config import TRADING_CONFIG

class EnhancedLiveTrader:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = MLStrategyTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = TRADING_CONFIG['symbol']
        
        # Trading state
        self.position = None
        self.balance = TRADING_CONFIG['initial_balance']
        self.initial_balance = TRADING_CONFIG['initial_balance']
        self.trade_log = []
        self.equity_curve = []
        self.peak_balance = TRADING_CONFIG['initial_balance']
        self.max_drawdown = 0
        
    def initialize(self):
        """Initialize the enhanced trader"""
        print("🚀 ENHANCED LIVE ML TRADER")
        print("=" * 50)
        
        if not self.ml_trainer.load_ml_models():
            return False
            
        print("✅ ML Models Loaded")
        print("💰 Initial Balance: ${:.2f}".format(self.initial_balance))
        print("🌐 Exchange: Binance | Symbol: BTC/USDT")
        print("⏰ Timeframe: 1 minute | Check: 60 seconds")
        return True
    
    def calculate_metrics(self, current_price):
        """Calculate real-time metrics"""
        # Current equity
        if self.position:
            position_value = self.position['size'] * current_price
            current_equity = self.balance + position_value
        else:
            current_equity = self.balance
        
        # Update peak and drawdown
        if current_equity > self.peak_balance:
            self.peak_balance = current_equity
        
        current_drawdown = (self.peak_balance - current_equity) / self.peak_balance * 100
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Total return
        total_return = (current_equity - self.initial_balance) / self.initial_balance * 100
        
        # Record equity curve
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': current_equity,
            'balance': self.balance,
            'drawdown': current_drawdown
        })
        
        return current_equity, total_return, current_drawdown
    
    def execute_trade(self, signal):
        """Execute trade with position sizing"""
        if signal['confidence'] < 0.65:
            return False
            
        current_price = signal['current_price']
        
        if signal['prediction'] == 'UP' and not self.position:
            # BUY Logic with 10% position sizing
            position_size_usd = self.balance * 0.10  # Risk 10% of balance
            btc_size = position_size_usd / current_price
            
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': btc_size,
                'entry_time': datetime.now(),
                'confidence': signal['confidence'],
                'position_value': position_size_usd
            }
            self.balance -= position_size_usd
            
            trade = {
                'action': 'BUY',
                'price': current_price,
                'size_btc': btc_size,
                'size_usd': position_size_usd,
                'confidence': signal['confidence'],
                'timestamp': datetime.now()
            }
            self.trade_log.append(trade)
            
            print("💚 LIVE: BUY {:.6f} BTC at ${:.2f}".format(btc_size, current_price))
            print("       Position Value: ${:.2f} | Confidence: {:.3f}".format(
                position_size_usd, signal['confidence']))
            return True
                
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
                'hold_time': (datetime.now() - self.position['entry_time']).total_seconds() / 60
            }
            self.trade_log.append(trade)
            
            pnl_color = "🟢" if pnl_percent > 0 else "🔴"
            print("{} LIVE: SELL | PnL: {:.2f}% | Amount: ${:.2f}".format(
                pnl_color, pnl_percent*100, pnl_amount))
                
            self.position = None
            return True
            
        return False
    
    def run_enhanced_trading(self):
        """Run enhanced live trading with full metrics"""
        if not self.initialize():
            return
            
        print("\n🤖 ENHANCED LIVE TRADING STARTED")
        print("Includes: Real-time PnL, Drawdown, Equity Curve, Position Sizing")
        print("=" * 50)
        
        iteration = 0
        while iteration < 100:  # Run for 100 iterations
            iteration += 1
            
            try:
                # Get market data
                data = self.data_manager.fetch_historical_data(limit=500)
                if data is None or data.empty:
                    print("❌ No data, waiting...")
                    time.sleep(30)
                    continue
                    
                current_price = data['close'].iloc[-1]
                
                # Calculate real-time metrics
                current_equity, total_return, current_drawdown = self.calculate_metrics(current_price)
                
                # Get ML signal
                signal_data = self.ml_trainer.predict_direction(data, self.indicator_engine)
                
                current_time = datetime.now().strftime('%H:%M:%S')
                
                print("\n" + "="*60)
                print("📊 {} | Price: ${:.2f}".format(current_time, current_price))
                print("="*60)
                
                # Display metrics
                print("💰 BALANCE: ${:.2f} | EQUITY: ${:.2f}".format(self.balance, current_equity))
                print("📈 RETURN: {:.2f}% | DRAWDOWN: {:.2f}%".format(total_return, current_drawdown))
                
                if signal_data:
                    print("🎯 SIGNAL: {} | Confidence: {:.3f}".format(
                        signal_data['prediction'], signal_data['confidence']))
                    
                    if 'model_predictions' in signal_data:
                        print("🤖 MODELS: {}".format(signal_data['model_predictions']))
                    
                    # Execute trading logic
                    trade_executed = self.execute_trade({
                        'prediction': signal_data['prediction'],
                        'confidence': signal_data['confidence'],
                        'current_price': current_price
                    })
                    
                    # Update metrics after potential trade
                    if trade_executed:
                        current_equity, total_return, current_drawdown = self.calculate_metrics(current_price)
                
                # Position status
                if self.position:
                    current_pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
                    pnl_color = "🟢" if current_pnl > 0 else "🔴"
                    print("📊 POSITION: LONG | Size: {:.6f} BTC".format(self.position['size']))
                    print("   PnL: {}{:.2f}% | Entry: ${:.2f}".format(
                        pnl_color, current_pnl*100, self.position['entry_price']))
                    
                    # Exit conditions check
                    if current_pnl <= -0.015:
                        print("🛑 STOP LOSS HIT! Would exit at -1.5%")
                    elif current_pnl >= 0.02:
                        print("🎯 TAKE PROFIT HIT! Would exit at +2.0%")
                    else:
                        time_in_trade = (datetime.now() - self.position['entry_time']).total_seconds() / 60
                        if time_in_trade > 15:
                            print("⏰ TIME EXIT! Would exit after 15 minutes")
                        else:
                            print("⏳ Time in trade: {:.1f}/15 minutes".format(time_in_trade))
                else:
                    print("💤 NO POSITION | Waiting for high-confidence signal...")
                
                print("⏰ Next check in 60 seconds...")
                time.sleep(60)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(30)
        
        # Final summary
        self.show_final_summary()
    
    def show_final_summary(self):
        """Show comprehensive trading summary"""
        print("\n" + "="*60)
        print("🎯 TRADING SESSION COMPLETE - FINAL SUMMARY")
        print("="*60)
        
        current_equity = self.balance  # Assume no position at end
        total_return = (current_equity - self.initial_balance) / self.initial_balance * 100
        
        print("💰 FINAL BALANCE: ${:.2f}".format(current_equity))
        print("📈 TOTAL RETURN: {:.2f}%".format(total_return))
        print("📉 MAX DRAWDOWN: {:.2f}%".format(self.max_drawdown))
        print("🔢 TOTAL TRADES: {}".format(len(self.trade_log)))
        
        if self.trade_log:
            winning_trades = [t for t in self.trade_log if 'pnl_percent' in t and t['pnl_percent'] > 0]
            win_rate = len(winning_trades) / len(self.trade_log) * 100
            
            print("🎯 WIN RATE: {:.1f}% ({}/{})".format(
                win_rate, len(winning_trades), len(self.trade_log)))
            
            if winning_trades:
                avg_win = sum(t['pnl_percent'] for t in winning_trades) / len(winning_trades) * 100
                print("📊 AVERAGE WIN: {:.2f}%".format(avg_win))
        
        print("\n💾 MODELS: XGBoost + Random Forest + Gradient Boosting")
        print("⏰ TIMEFRAME: 1-minute candles | Binance BTC/USDT")

if __name__ == "__main__":
    trader = EnhancedLiveTrader()
    trader.run_enhanced_trading()
