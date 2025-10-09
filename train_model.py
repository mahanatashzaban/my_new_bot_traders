#!/usr/bin/env python3
"""
Train the ML model with MORE historical data
"""

import pandas as pd
from data_manager import DataManager
from model_trainer import ModelTrainer

def train_and_validate():
    print("🔧 TRAINING PHASE")
    print("=" * 50)
    
    # Fetch MORE historical data
    data_manager = DataManager()
    print("Fetching historical data...")
    
    # Get much more data for better training
    historical_data = data_manager.fetch_historical_data(limit=5000)
    
    if historical_data is None:
        print("❌ Failed to fetch historical data")
        return False
    
    print(f"📊 Loaded {len(historical_data)} candles of historical data")
    print(f"💰 Price range: ${historical_data['low'].min():.2f} - ${historical_data['high'].max():.2f}")
    print(f"📅 Date range: {historical_data.index[0]} to {historical_data.index[-1]}")
    
    # Train model
    trainer = ModelTrainer()
    print("\n🎯 Training ML models...")
    models = trainer.train_models(historical_data)
    
    if models:
        print("\n🎉 Model training completed successfully!")
        print("   Next step: python backtest_strategy.py")
        return True
    else:
        print("\n❌ Model training failed")
        print("   This might be because:")
        print("   - Not enough trendline setups found in the data")
        print("   - Try running again with different market conditions")
        return False

if __name__ == "__main__":
    train_and_validate()
