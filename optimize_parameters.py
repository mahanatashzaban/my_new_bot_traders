#!/usr/bin/env python3
"""
Optimize Strategy Parameters
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from indicators import IndicatorEngine

class ParameterOptimizer:
    def __init__(self):
        self.data_manager = DataManager()
        self.indicator_engine = IndicatorEngine()
    
    def test_parameter_set(self, data, distance_thresh, rejection_thresh, profit_thresh=0.005, hold_bars=5):
        """Test a specific parameter set"""
        total_trades = 0
        successful_trades = 0
        total_return = 0
        
        for i in range(50, len(data) - hold_bars):
            # Check entry conditions
            if (data['trendline_touch'].iloc[i] == 1 and 
                data['distance_to_trendline'].iloc[i] < distance_thresh and
                data['rejection_strength'].iloc[i] > rejection_thresh):
                
                current_price = data['close'].iloc[i]
                future_prices = data['close'].iloc[i+1:i+hold_bars+1]
                
                # Determine trade direction
                if data['trendline_slope'].iloc[i] < 0:  # SHORT
                    price_move = (current_price - future_prices.min()) / current_price
                    if price_move > profit_thresh:
                        successful_trades += 1
                        total_return += price_move
                    else:
                        total_return += price_move  # Still count the actual move
                    total_trades += 1
                    
                else:  # LONG
                    price_move = (future_prices.max() - current_price) / current_price
                    if price_move > profit_thresh:
                        successful_trades += 1
                        total_return += price_move
                    else:
                        total_return += price_move
                    total_trades += 1
        
        if total_trades > 0:
            win_rate = successful_trades / total_trades
            avg_return = total_return / total_trades
            return win_rate, avg_return, total_trades
        else:
            return 0, 0, 0
    
    def optimize(self):
        """Run comprehensive parameter optimization"""
        print("🔧 OPTIMIZING STRATEGY PARAMETERS")
        print("=" * 50)
        
        # Load more data for robust optimization
        data = self.data_manager.fetch_historical_data(limit=3000)
        if data is None:
            print("❌ Failed to fetch data")
            return
        
        print("Calculating indicators...")
        data = self.indicator_engine.calculate_all_indicators(data)
        
        best_score = 0
        best_params = {}
        results = []
        
        # Test various parameter combinations
        print("\n🧪 Testing parameter combinations...")
        
        for distance_thresh in [0.003, 0.005, 0.008, 0.01, 0.015]:
            for rejection_thresh in [0.5, 0.8, 1.0, 1.5, 2.0]:
                for profit_thresh in [0.003, 0.005, 0.008, 0.01]:
                    
                    win_rate, avg_return, total_trades = self.test_parameter_set(
                        data, distance_thresh, rejection_thresh, profit_thresh
                    )
                    
                    if total_trades >= 10:  # Only consider if we have enough trades
                        # Score = win_rate * avg_return * log(trades) - balances multiple factors
                        score = win_rate * avg_return * np.log1p(total_trades)
                        
                        results.append({
                            'distance': distance_thresh,
                            'rejection': rejection_thresh,
                            'profit': profit_thresh,
                            'win_rate': win_rate,
                            'avg_return': avg_return,
                            'trades': total_trades,
                            'score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = results[-1]
                        
                        print(f"  dist={distance_thresh:.3f}, rej={rejection_thresh:.1f}, profit={profit_thresh:.3f} -> "
                              f"win={win_rate:.3f}, return={avg_return:.4f}, trades={total_trades}")
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🏆 TOP 5 PARAMETER SETS:")
        print("=" * 70)
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. dist={result['distance']:.3f}, rej={result['rejection']:.1f}, profit={result['profit']:.3f}")
            print(f"   Win Rate: {result['win_rate']:.3f}, Avg Return: {result['avg_return']:.4f}, Trades: {result['trades']}")
            print(f"   Score: {result['score']:.6f}")
            print()
        
        print(f"🎯 RECOMMENDED PARAMETERS:")
        print(f"   Distance Threshold: {best_params['distance']:.3f}")
        print(f"   Rejection Threshold: {best_params['rejection']:.1f}") 
        print(f"   Profit Target: {best_params['profit']:.3f}")
        print(f"   Expected Win Rate: {best_params['win_rate']:.3f}")
        print(f"   Expected Avg Return: {best_params['avg_return']:.4f} per trade")
        print(f"   Expected Trades: {best_params['trades']}")
        
        return best_params

def main():
    optimizer = ParameterOptimizer()
    best_params = optimizer.optimize()
    
    print(f"\n💡 Next steps:")
    print(f"1. Update indicators.py with these parameters")
    print(f"2. Retrain the ML model: python train_model.py")
    print(f"3. Run full backtest: python backtest_strategy.py")

if __name__ == "__main__":
    main()
