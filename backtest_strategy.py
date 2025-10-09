#!/usr/bin/env python3
"""
Complete Backtest for YOUR Trendline Retest Strategy - FIXED VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_manager import DataManager
from model_trainer import ModelTrainer
from indicators import IndicatorEngine
import joblib
from datetime import datetime, timedelta

class StrategyBacktester:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []
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
    
    def backtest_your_strategy(self, df):
        """Backtest specifically YOUR trendline retest strategy - FIXED VERSION"""
        balance = self.initial_balance
        position = None
        trades = []
        equity_curve = []
        
        print("🔍 Backtesting YOUR Trendline Retest Strategy...")
        print("=" * 60)
        
        strategy_signals = 0
        entered_trades = 0
        
        for i in range(50, len(df)):
            current_data = df.iloc[i]
            current_price = current_data['close']
            current_time = df.index[i]
            
            # Track equity curve
            current_equity = balance
            if position:
                pnl_percent = self._calculate_pnl_percent(position, current_price)
                current_equity += balance * pnl_percent
            
            equity_curve.append({
                'timestamp': current_time,
                'balance': balance,
                'price': current_price,
                'equity': current_equity
            })
            
            # CHECK EXIT CONDITIONS FIRST (if we have a position)
            if position:
                exit_reason = None
                pnl_percent = self._calculate_pnl_percent(position, current_price)
                
                # Stop Loss Check (2%)
                if position['type'] == 'LONG' and current_price <= position['stop_loss']:
                    exit_reason = "STOP LOSS"
                elif position['type'] == 'SHORT' and current_price >= position['stop_loss']:
                    exit_reason = "STOP LOSS"
                
                # Take Profit Check (3%)
                elif position['type'] == 'LONG' and current_price >= position['take_profit']:
                    exit_reason = "TAKE PROFIT"
                elif position['type'] == 'SHORT' and current_price <= position['take_profit']:
                    exit_reason = "TAKE PROFIT"
                
                # Time-based exit (max 30 minutes hold)
                elif (current_time - position['entry_time']).total_seconds() > 1800:  # 30 minutes
                    exit_reason = "TIME EXIT"
                
                # Exit if conditions met
                if exit_reason:
                    pnl_amount = balance * pnl_percent
                    balance += pnl_amount
                    
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'size': position['size'],
                        'pnl_percent': pnl_percent,
                        'pnl_amount': pnl_amount,
                        'balance_after': balance,
                        'exit_reason': exit_reason,
                        'hold_time_minutes': (current_time - position['entry_time']).total_seconds() / 60
                    }
                    trades.append(trade_record)
                    
                    print(f"🔴 EXIT {position['type']}: {exit_reason} | "
                          f"PnL: {pnl_percent*100:+.2f}% | "
                          f"Price: ${current_price:.2f} | "
                          f"Balance: ${balance:.2f}")
                    
                    position = None
            
            # CHECK ENTRY CONDITIONS (if no position)
            if not position:
                # YOUR STRATEGY RULES - MATCH TRAINING PARAMETERS
                is_trendline_touch = current_data['trendline_touch'] == 1
                has_rejection = current_data['rejection_strength'] > 0.8  # Lower threshold like training
                good_volume = current_data['volume_ma_ratio'] > 0.5
                
                # Get ML confirmation if available
                ml_confidence = 0.0
                ml_signal = "HOLD"
                try:
                    # Prepare features for current point
                    historical_data = df.iloc[:i+1].copy()
                    X, _ = self.model_trainer.prepare_features(historical_data)
                    if not X.empty:
                        X_scaled = self.model_trainer.scaler.transform(X)
                        model = self.model_trainer.models['xgb']
                        prediction = model.predict(X_scaled)[-1]  # Get latest prediction
                        probability = model.predict_proba(X_scaled)[-1]
                        ml_confidence = max(probability)
                        
                        if prediction == 1:
                            ml_signal = "BUY"
                        else:
                            ml_signal = "SELL"
                except Exception as e:
                    # If ML fails, continue without it
                    ml_signal = "HOLD"
                    ml_confidence = 0.0
                
                if is_trendline_touch and has_rejection and good_volume:
                    strategy_signals += 1
                    
                    # Determine direction based on trendline slope
                    if current_data['trendline_slope'] < 0:  # Resistance line - SHORT
                        signal = "SHORT"
                        stop_loss = current_price * 1.02  # 2% stop loss
                        take_profit = current_price * 0.97  # 3% take profit
                    else:  # Support line - LONG
                        signal = "LONG"
                        stop_loss = current_price * 0.98  # 2% stop loss
                        take_profit = current_price * 1.03  # 3% take profit
                    
                    # Check if ML confirms the signal
                    ml_confirms = (signal == "LONG" and ml_signal == "BUY") or (signal == "SHORT" and ml_signal == "SELL")
                    
                    # Enter trade if ML confirms OR if we don't have ML (for testing)
                    if ml_confirms or ml_confidence == 0.0:
                        # Enter trade
                        position_size = (balance * 0.02) / current_price  # 2% risk
                        position = {
                            'type': signal,
                            'entry_price': current_price,
                            'size': position_size,
                            'entry_time': current_time,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'ml_confidence': ml_confidence
                        }
                        
                        entered_trades += 1
                        print(f"🎯 STRATEGY TRIGGERED: {signal} at ${current_price:.2f} | "
                              f"Rejection: {current_data['rejection_strength']:.2f} | "
                              f"ML Confidence: {ml_confidence:.2f}")
        
        # Close any open position at end
        if position:
            current_price = df.iloc[-1]['close']
            pnl_percent = self._calculate_pnl_percent(position, current_price)
            pnl_amount = balance * pnl_percent
            balance += pnl_amount
            
            trade_record = {
                'entry_time': position['entry_time'],
                'exit_time': df.index[-1],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'size': position['size'],
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'balance_after': balance,
                'exit_reason': "END OF DATA",
                'hold_time_minutes': (df.index[-1] - position['entry_time']).total_seconds() / 60
            }
            trades.append(trade_record)
            
            print(f"🔴 FORCE EXIT {position['type']}: END OF DATA | "
                  f"PnL: {pnl_percent*100:+.2f}%")
        
        # Store results
        self.balance = balance
        self.trades = trades
        self.equity_curve = equity_curve
        
        print(f"\n📊 STRATEGY PERFORMANCE SUMMARY:")
        print(f"   Signals Generated: {strategy_signals}")
        print(f"   Trades Entered: {entered_trades}")
        print(f"   Completed Trades: {len(trades)}")
        
        return self._analyze_strategy_performance(df)
    
    def _calculate_pnl_percent(self, position, current_price):
        """Calculate P&L percentage for position"""
        if position['type'] == 'LONG':
            return (current_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            return (position['entry_price'] - current_price) / position['entry_price']
    
    def _calculate_position_value(self, position, current_price):
        """Calculate current position value"""
        if not position:
            return 0
        
        pnl_percent = self._calculate_pnl_percent(position, current_price)
        return self.initial_balance * pnl_percent
    
    def _analyze_strategy_performance(self, df):
        """Analyze and print strategy performance"""
        if not self.trades:
            print("❌ No trades executed during backtest")
            return False
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] <= 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl_percent'].mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() * 100 if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl_amount'].sum() / losing_trades['pnl_amount'].sum()) if losing_trades['pnl_amount'].sum() != 0 else float('inf')
        
        # Strategy-specific metrics
        long_trades = trades_df[trades_df['type'] == 'LONG']
        short_trades = trades_df[trades_df['type'] == 'SHORT']
        
        print("\n" + "="*70)
        print("📈 YOUR TRENDLINE RETEST STRATEGY - BACKTEST RESULTS")
        print("="*70)
        
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"💰 Final Balance: ${self.balance:,.2f}")
        print(f"📊 Total Return: {total_return:+.2f}%")
        print(f"🎯 Total Trades: {len(trades_df)}")
        print(f"✅ Winning Trades: {len(winning_trades)}")
        print(f"❌ Losing Trades: {len(losing_trades)}")
        print(f"📈 Win Rate: {win_rate:.1f}%")
        print(f"🔥 Average Win: {avg_win:+.2f}%")
        print(f"💧 Average Loss: {avg_loss:+.2f}%")
        print(f"📊 Profit Factor: {profit_factor:.2f}")
        print(f"📉 Max Consecutive Wins: {self._max_consecutive_wins(trades_df)}")
        print(f"📈 Max Consecutive Losses: {self._max_consecutive_losses(trades_df)}")
        
        if len(long_trades) > 0:
            long_win_rate = len(long_trades[long_trades['pnl_percent'] > 0]) / len(long_trades) * 100
            print(f"🟢 Long Trades: {len(long_trades)} | Win Rate: {long_win_rate:.1f}%")
        
        if len(short_trades) > 0:
            short_win_rate = len(short_trades[short_trades['pnl_percent'] > 0]) / len(short_trades) * 100
            print(f"🔴 Short Trades: {len(short_trades)} | Win Rate: {short_win_rate:.1f}%")
        
        # Trade duration analysis
        avg_hold_time = trades_df['hold_time_minutes'].mean()
        print(f"⏱️  Average Hold Time: {avg_hold_time:.1f} minutes")
        
        # Drawdown analysis
        max_drawdown = self._calculate_max_drawdown()
        print(f"📉 Maximum Drawdown: {max_drawdown:.2f}%")
        
        # Exit reason analysis
        exit_reasons = trades_df['exit_reason'].value_counts()
        print(f"\n📋 Exit Reasons:")
        for reason, count in exit_reasons.items():
            print(f"   {reason}: {count} trades")
        
        # Validation criteria
        is_profitable = total_return > 0
        has_good_win_rate = win_rate > 45  # Lowered threshold for crypto
        has_positive_expectancy = (win_rate/100 * avg_win + (1-win_rate/100) * avg_loss) > 0
        
        print(f"\n🔍 STRATEGY VALIDATION:")
        print(f"   Profitable: {'✅' if is_profitable else '❌'} ({total_return:+.2f}%)")
        print(f"   Good Win Rate: {'✅' if has_good_win_rate else '❌'} ({win_rate:.1f}%)")
        print(f"   Positive Expectancy: {'✅' if has_positive_expectancy else '❌'}")
        print(f"   Overall: {'✅ PASSED' if (is_profitable and has_positive_expectancy) else '❌ FAILED'}")
        
        # Plot results
        self._plot_results(df)
        
        return is_profitable and has_positive_expectancy
    
    def _max_consecutive_wins(self, trades_df):
        """Calculate maximum consecutive winning trades"""
        max_streak = 0
        current_streak = 0
        
        for pnl in trades_df['pnl_percent']:
            if pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _max_consecutive_losses(self, trades_df):
        """Calculate maximum consecutive losing trades"""
        max_streak = 0
        current_streak = 0
        
        for pnl in trades_df['pnl_percent']:
            if pnl <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from equity curve"""
        if not self.equity_curve:
            return 0
        
        equities = [point['equity'] for point in self.equity_curve]
        peak = equities[0]
        max_dd = 0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _plot_results(self, df):
        """Plot backtest results"""
        if not self.trades:
            print("❌ No trades to plot")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Price and trades
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='BTC Price', alpha=0.7, linewidth=1)
        
        # Plot trades
        trades_df = pd.DataFrame(self.trades)
        long_entries = trades_df[trades_df['type'] == 'LONG']
        short_entries = trades_df[trades_df['type'] == 'SHORT']
        
        plt.scatter(long_entries['entry_time'], long_entries['entry_price'], 
                   color='green', marker='^', s=100, label='Long Entries', zorder=5)
        plt.scatter(short_entries['entry_time'], short_entries['entry_price'], 
                   color='red', marker='v', s=100, label='Short Entries', zorder=5)
        
        plt.scatter(trades_df['exit_time'], trades_df['exit_price'], 
                   color='orange', marker='o', s=50, label='Exits', alpha=0.7, zorder=4)
        
        plt.title('YOUR Trendline Retest Strategy - Trade Entries/Exits')
        plt.ylabel('Price (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Equity curve
        plt.subplot(2, 1, 2)
        equity_df = pd.DataFrame(self.equity_curve)
        plt.plot(equity_df['timestamp'], equity_df['equity'], label='Portfolio Equity', linewidth=2)
        plt.axhline(y=self.initial_balance, color='red', linestyle='--', label='Initial Balance')
        plt.title('Portfolio Equity Curve')
        plt.ylabel('Equity (USDT)')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('strategy_backtest_results.png', dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved as 'strategy_backtest_results.png'")

def main():
    print("🎯 YOUR TRENDLINE RETEST STRATEGY BACKTEST")
    print("=" * 50)
    
    # Load data
    data_manager = DataManager()
    print("Fetching historical data...")
    historical_data = data_manager.fetch_historical_data(limit=2000)
    
    if historical_data is None:
        print("❌ Failed to fetch historical data")
        return False
    
    print(f"📊 Loaded {len(historical_data)} candles")
    
    # Calculate indicators (including your trendline features)
    print("Calculating indicators and trendline features...")
    indicator_engine = IndicatorEngine()
    data_with_indicators = indicator_engine.calculate_all_indicators(historical_data)
    
    # Run backtest
    backtester = StrategyBacktester(initial_balance=1000)
    
    if not backtester.load_model():
        print("❌ Cannot run backtest without trained models")
        return False
    
    print("\n" + "="*50)
    success = backtester.backtest_your_strategy(data_with_indicators)
    
    if success:
        print("\n🎉 YOUR STRATEGY VALIDATION PASSED!")
        print("   Consider proceeding to live trading with small amounts.")
    else:
        print("\n⚠️  YOUR STRATEGY NEEDS OPTIMIZATION!")
        print("   Review parameters before live trading.")
    
    return success

if __name__ == "__main__":
    main()
