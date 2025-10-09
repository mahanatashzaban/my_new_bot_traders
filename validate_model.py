#!/usr/bin/env python3
"""
Validate model performance with walk-forward backtesting
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from model_trainer import ModelTrainer
from indicators import IndicatorEngine
import joblib
from datetime import datetime, timedelta

class Backtester:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.model_trainer = ModelTrainer()
        self.indicator_engine = IndicatorEngine()
        
    def load_model(self):
        """Load trained model"""
        try:
            self.model_trainer.load_models()
            return True
        except:
            print("❌ No trained model found. Run train_model.py first.")
            return False
    
    def walk_forward_backtest(self, df, period_days=30):
        """Walk-forward backtesting with periodic retraining"""
        print("🔍 Running Walk-Forward Backtest...")
        
        # Split data into training and testing periods
        total_days = (df.index[-1] - df.index[0]).days
        test_periods = []
        
        for start_day in range(0, total_days - period_days, 7):  # Slide 7 days each time
            train_end = df.index[0] + timedelta(days=start_day)
            test_start = train_end
            test_end = test_start + timedelta(days=period_days)
            
            train_data = df[df.index <= train_end].tail(1000)  # Last 1000 candles for training
            test_data = df[(df.index > test_start) & (df.index <= test_end)]
            
            if len(test_data) > 100:  # Minimum test data
                test_periods.append((train_data, test_data))
        
        print(f"📊 Testing across {len(test_periods)} periods")
        
        all_results = []
        
        for i, (train_data, test_data) in enumerate(test_periods):
            print(f"🧪 Testing period {i+1}/{len(test_periods)}: {test_data.index[0].date()} to {test_data.index[-1].date()}")
            
            # Retrain model on current training period
            self.model_trainer.train_models(train_data)
            
            # Test on this period
            period_results = self.backtest_period(test_data)
            all_results.append(period_results)
        
        return self.analyze_results(all_results)
    
    def backtest_period(self, df):
        """Backtest on a specific period"""
        balance = self.initial_balance
        position = None
        trades = []
        
        for i in range(50, len(df)):  # Start from 50 to have enough history
            current_data = df.iloc[:i+1]
            current_price = df.iloc[i]['close']
            current_time = df.index[i]
            
            # Get trading signal
            signal, confidence = self.get_trading_signal(current_data)
            
            # Check exit conditions
            if position:
                pnl_percent = self.calculate_pnl(position, current_price)
                
                # Exit conditions
                if (position['type'] == 'LONG' and (pnl_percent <= -0.01 or pnl_percent >= 0.02)) or \
                   (position['type'] == 'SHORT' and (pnl_percent <= -0.01 or pnl_percent >= 0.02)):
                    
                    # Close position
                    pnl_amount = balance * pnl_percent
                    balance += pnl_amount
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_percent': pnl_percent,
                        'pnl_amount': pnl_amount,
                        'balance': balance
                    })
                    
                    position = None
            
            # Enter new position
            if not position and signal in ['BUY', 'SELL'] and confidence > 0.6:
                position_size = (balance * 0.02) / current_price  # 2% risk
                
                position = {
                    'type': 'LONG' if signal == 'BUY' else 'SHORT',
                    'entry_price': current_price,
                    'size': position_size,
                    'entry_time': current_time
                }
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'total_return': (balance - self.initial_balance) / self.initial_balance,
            'trades': trades,
            'num_trades': len(trades)
        }
    
    def get_trading_signal(self, df):
        """Get trading signal from ML model"""
        try:
            X, _ = self.model_trainer.prepare_features(df)
            if X.empty:
                return "HOLD", 0.0
            
            X_scaled = self.model_trainer.scaler.transform(X)
            model = self.model_trainer.models['xgb']
            prediction = model.predict(X_scaled)[-1]
            probability = model.predict_proba(X_scaled)[-1]
            confidence = max(probability)
            
            if prediction == 1:
                return "BUY", confidence
            elif prediction == 0:
                return "SELL", confidence
            else:
                return "HOLD", confidence
                
        except Exception as e:
            return "HOLD", 0.0
    
    def calculate_pnl(self, position, current_price):
        """Calculate P&L for current position"""
        if position['type'] == 'LONG':
            return (current_price - position['entry_price']) / position['entry_price']
        else:
            return (position['entry_price'] - current_price) / position['entry_price']
    
    def analyze_results(self, all_results):
        """Analyze backtest results"""
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        total_return = 0
        total_trades = 0
        winning_trades = 0
        all_trades = []
        
        for result in all_results:
            total_return += result['total_return']
            total_trades += result['num_trades']
            all_trades.extend(result['trades'])
            
            for trade in result['trades']:
                if trade['pnl_percent'] > 0:
                    winning_trades += 1
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            avg_return = (total_return / len(all_results)) * 100
            
            print(f"📈 Average Period Return: {avg_return:.2f}%")
            print(f"🎯 Total Trades: {total_trades}")
            print(f"✅ Winning Trades: {winning_trades}")
            print(f"📊 Win Rate: {win_rate:.1f}%")
            
            if all_trades:
                avg_trade_return = np.mean([t['pnl_percent'] for t in all_trades]) * 100
                best_trade = max([t['pnl_percent'] for t in all_trades]) * 100
                worst_trade = min([t['pnl_percent'] for t in all_trades]) * 100
                
                print(f"📋 Average Trade Return: {avg_trade_return:.2f}%")
                print(f"🔥 Best Trade: {best_trade:.2f}%")
                print(f"💧 Worst Trade: {worst_trade:.2f}%")
            
            # Validation criteria
            if win_rate > 55 and avg_return > 0:
                print("\n✅ VALIDATION PASSED - Model is profitable")
                return True
            else:
                print("\n❌ VALIDATION FAILED - Model needs improvement")
                return False
        else:
            print("❌ No trades executed during backtest")
            return False

def main():
    print("🔍 MODEL VALIDATION PHASE")
    print("=" * 50)
    
    # Load historical data for validation
    data_manager = DataManager()
    print("Fetching validation data...")
    validation_data = data_manager.fetch_historical_data(limit=2000)
    
    if validation_data is None:
        print("❌ Failed to fetch validation data")
        return False
    
    # Run backtest
    backtester = Backtester()
    
    if not backtester.load_model():
        return False
    
    # Run walk-forward validation
    is_profitable = backtester.walk_forward_backtest(validation_data)
    
    return is_profitable

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Validation successful! You can proceed to live trading.")
    else:
        print("\n⚠️  Validation failed. Review your strategy before live trading.")
