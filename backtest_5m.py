#!/usr/bin/env python3
"""
Backtest for 5-Minute Strategy
"""

import pandas as pd
from data_manager_simple import DataManager
from ml_trainer_simple import SimpleMLTrainer
from simple_indicators import SimpleMLIndicatorEngine
from simple_config import TRADING_CONFIG

class FiveMinuteBacktester:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.ml_trainer = SimpleMLTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        
    def backtest_5m_strategy(self, df):
        """Backtest 5-minute ML strategy"""
        print("🤖 BACKTESTING 5-MINUTE ML STRATEGY")
        print("=" * 50)
        
        if not self.ml_trainer.load_ml_models():
            print("❌ Please train 5m models first")
            return False
        
        print("✅ 5-minute models loaded")
        
        # Start after enough data for indicators
        start_index = 150
        
        for i in range(start_index, len(df)):
            current_data = df.iloc[i]
            current_price = current_data['close']
            current_time = df.index[i]
            
            # Use historical data for prediction
            historical_data = df.iloc[max(0, i-200):i+1].copy()
            
            # Get ML prediction
            prediction = self.ml_trainer.predict_direction(historical_data, self.indicator_engine)
            
            if not prediction or prediction['confidence'] < 0.60:
                continue
            
            # EXIT LOGIC
            if self.position:
                pnl_percent = self._calculate_pnl(self.position, current_price)
                
                # Exit conditions
                exit_trade = False
                
                # Stop loss (2%)
                if pnl_percent <= -0.02:
                    exit_trade = True
                    exit_reason = "STOP LOSS"
                
                # Take profit (3%)
                elif pnl_percent >= 0.03:
                    exit_trade = True
                    exit_reason = "TAKE PROFIT"
                
                # ML reversal
                elif (self.position['type'] == 'LONG' and prediction['prediction'] == 'DOWN' and 
                      prediction['confidence'] > 0.65):
                    exit_trade = True
                    exit_reason = "ML REVERSAL"
                
                if exit_trade:
                    pnl_amount = self.balance * pnl_percent
                    self.balance += pnl_amount
                    
                    self.trades.append({
                        'entry_time': self.position['entry_time'],
                        'exit_time': current_time,
                        'type': self.position['type'],
                        'entry_price': self.position['entry_price'],
                        'exit_price': current_price,
                        'pnl_percent': pnl_percent,
                        'pnl_amount': pnl_amount,
                        'balance': self.balance,
                        'exit_reason': exit_reason,
                        'ml_confidence': self.position['ml_confidence']
                    })
                    
                    print(f"EXIT {self.position['type']}: {exit_reason} | PnL: {pnl_percent*100:+.2f}%")
                    self.position = None
            
            # ENTRY LOGIC
            if not self.position and prediction['confidence'] > 0.60:
                if prediction['prediction'] == 'UP':
                    # ENTER LONG
                    self.position = {
                        'type': 'LONG',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'ml_confidence': prediction['confidence']
                    }
                    print(f"ENTER LONG at ${current_price:.2f} | Confidence: {prediction['confidence']:.2f}")
        
        return self._analyze_performance()
    
    def _calculate_pnl(self, position, current_price):
        if position['type'] == 'LONG':
            return (current_price - position['entry_price']) / position['entry_price']
        return 0
    
    def _analyze_performance(self):
        if not self.trades:
            print("❌ No trades executed")
            return False
        
        trades_df = pd.DataFrame(self.trades)
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        
        print(f"\n🤖 5-MINUTE BACKTEST RESULTS:")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Average Win: {avg_win:+.2f}%")
        print(f"Average Loss: {avg_loss:+.2f}%")
        
        return win_rate > 50

def main():
    print("🚀 5-MINUTE STRATEGY BACKTEST")
    print("=" * 50)
    
    dm = DataManager()
    print("Fetching 5-minute data for backtest...")
    data = dm.fetch_historical_data(limit=2000)  # ~7 days of 5m data
    
    if data is None:
        print("❌ Failed to fetch data")
        return
    
    print(f"✅ Loaded {len(data)} 5-minute candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    backtester = FiveMinuteBacktester(initial_balance=1000)
    success = backtester.backtest_5m_strategy(data)
    
    if success:
        print("\n🎉 5-minute strategy shows promise!")
    else:
        print("\n⚠️ 5-minute strategy needs improvement")

if __name__ == "__main__":
    main()
