#!/usr/bin/env python3
"""
ML Strategy Backtest - Pure Indicator-Based
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ml_trainer import MLStrategyTrainer
from simple_indicators import SimpleMLIndicatorEngine

class MLStrategyBacktester:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.ml_trainer = MLStrategyTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        
    def backtest_ml_strategy(self, df):
        """Backtest ML-based trading strategy"""
        print("🤖 Backtesting ML Strategy...")
        print("=" * 50)
        
        balance = self.initial_balance
        position = None
        trades = []
        
        # Load ML models
        if not self.ml_trainer.load_ml_models():
            print("❌ Cannot run backtest without trained ML models")
            return False
        
        print("✅ ML models loaded successfully")
        
        # Start from a point where we have enough data for indicators
        start_index = 200  # Start after we have enough data for indicators
        
        for i in range(start_index, len(df)):
            current_data = df.iloc[i]
            current_price = current_data['close']
            current_time = df.index[i]
            
            # Use recent data for prediction (last 500 points)
            start_pred = max(0, i - 500)
            historical_data = df.iloc[start_pred:i+1].copy()
            
            # Get ML prediction
            prediction_result = self.ml_trainer.predict_direction(historical_data, self.indicator_engine)
            
            if not prediction_result:
                continue
            
            # EXIT LOGIC
            if position:
                pnl_percent = self._calculate_pnl(position, current_price)
                
                # Exit conditions
                exit_trade = False
                exit_reason = ""
                
                # Stop loss (1.5%)
                if pnl_percent <= -0.015:
                    exit_trade = True
                    exit_reason = "STOP LOSS"
                
                # Take profit (2%)
                elif pnl_percent >= 0.02:
                    exit_trade = True
                    exit_reason = "TAKE PROFIT"
                
                # Time exit (15 minutes)
                elif (current_time - position['entry_time']).total_seconds() > 900:
                    exit_trade = True
                    exit_reason = "TIME EXIT"
                
                # ML reversal (if prediction changes)
                elif (position['type'] == 'LONG' and prediction_result['prediction'] == 'DOWN' and 
                      prediction_result['confidence'] > 0.7):
                    exit_trade = True
                    exit_reason = "ML REVERSAL"
                elif (position['type'] == 'SHORT' and prediction_result['prediction'] == 'UP' and 
                      prediction_result['confidence'] > 0.7):
                    exit_trade = True
                    exit_reason = "ML REVERSAL"
                
                if exit_trade:
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
                        'balance': balance,
                        'exit_reason': exit_reason,
                        'ml_confidence': position['ml_confidence']
                    })
                    
                    pnl_color = "🟢" if pnl_percent > 0 else "🔴"
                    print(f"{pnl_color} EXIT {position['type']}: {exit_reason} | PnL: {pnl_percent*100:+.2f}% | Balance: ${balance:.2f}")
                    position = None
            
            # ENTRY LOGIC
            if not position:
                # Only enter if ML confidence is high enough
                if prediction_result['confidence'] > 0.55:
                    if prediction_result['prediction'] == 'UP':
                        # ENTER LONG
                        position = {
                            'type': 'LONG',
                            'entry_price': current_price,
                            'entry_time': current_time,
                            'ml_confidence': prediction_result['confidence'],
                            'model_predictions': prediction_result['model_predictions']
                        }
                        print(f"🟢 ENTER LONG at ${current_price:.2f} | Confidence: {prediction_result['confidence']:.2f}")
                    
                    elif prediction_result['prediction'] == 'DOWN':
                        # ENTER SHORT
                        position = {
                            'type': 'SHORT', 
                            'entry_price': current_price,
                            'entry_time': current_time,
                            'ml_confidence': prediction_result['confidence'],
                            'model_predictions': prediction_result['model_predictions']
                        }
                        print(f"🔴 ENTER SHORT at ${current_price:.2f} | Confidence: {prediction_result['confidence']:.2f}")
        
        # Close any open position at the end
        if position:
            current_price = df['close'].iloc[-1]
            pnl_percent = self._calculate_pnl(position, current_price)
            pnl_amount = balance * pnl_percent
            balance += pnl_amount
            
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.index[-1],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'balance': balance,
                'exit_reason': 'END OF BACKTEST',
                'ml_confidence': position['ml_confidence']
            })
            print(f"🟡 CLOSED {position['type']} at end | PnL: {pnl_percent*100:+.2f}%")
        
        # Store results
        self.balance = balance
        self.trades = trades
        
        return self._analyze_ml_performance()
    
    def _calculate_pnl(self, position, current_price):
        """Calculate P&L percentage"""
        if position['type'] == 'LONG':
            return (current_price - position['entry_price']) / position['entry_price']
        else:
            return (position['entry_price'] - current_price) / position['entry_price']
    
    def _analyze_ml_performance(self):
        """Analyze ML strategy performance"""
        if not self.trades:
            print("❌ No trades executed")
            return False
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        
        # ML-specific metrics
        avg_confidence = trades_df['ml_confidence'].mean() if len(trades_df) > 0 else 0
        winning_confidence = winning_trades['ml_confidence'].mean() if len(winning_trades) > 0 else 0
        losing_confidence = losing_trades['ml_confidence'].mean() if len(losing_trades) > 0 else 0
        
        print(f"\n🤖 ML STRATEGY RESULTS:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        print(f"Average ML Confidence: {avg_confidence:.2f}")
        print(f"Winning Trade Confidence: {winning_confidence:.2f}")
        print(f"Losing Trade Confidence: {losing_confidence:.2f}")
        
        # Exit reason analysis
        if len(trades_df) > 0:
            exit_reasons = trades_df['exit_reason'].value_counts()
            print(f"\n📋 Exit Reasons:")
            for reason, count in exit_reasons.items():
                print(f"  {reason}: {count} trades")
        
        # Strategy assessment
        if win_rate >= 55 and total_return >= 3.0:
            assessment = "🎉 EXCELLENT ML STRATEGY"
        elif win_rate >= 52 and total_return >= 1.5:
            assessment = "✅ VERY GOOD ML STRATEGY"
        elif win_rate >= 50 and total_return >= 0.5:
            assessment = "👍 GOOD ML STRATEGY"
        else:
            assessment = "⚠️ ML STRATEGY NEEDS OPTIMIZATION"
        
        print(f"\n{assessment}")
        
        return win_rate >= 50 and total_return > 0

def main():
    print("🚀 ML-BASED TRADING STRATEGY BACKTEST")
    print("Using trained ML models with technical indicators")
    print("=" * 60)
    
    # Load data - use more data for better backtest
    dm = DataManager()
    print("Fetching data for backtest...")
    data = dm.fetch_historical_data(limit=3000)  # Increased for better backtest
    
    if data is None:
        print("❌ Failed to fetch data for backtest")
        return
    
    print(f"✅ Loaded {len(data)} candles for backtest")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Run backtest
    backtester = MLStrategyBacktester(initial_balance=1000)
    success = backtester.backtest_ml_strategy(data)
    
    if success:
        print(f"\n🎉 ML Strategy shows promise!")
        print("Consider training on more data and optimizing parameters")
    else:
        print(f"\n⚠️ ML Strategy needs improvement")
        print("Try different indicator combinations or ML parameters")

if __name__ == "__main__":
    main()
