#!/usr/bin/env python3
"""
Train ML Models on Technical Indicators
"""

import pandas as pd
from data_manager import DataManager
from ml_trainer import MLStrategyTrainer
from indicators import MLIndicatorEngine

def train_ml_models():
    print("🤖 TRAINING ML MODELS ON TECHNICAL INDICATORS")
    print("=" * 50)
    
    # Load substantial data for training
    dm = DataManager()
    print("Fetching training data...")
    training_data = dm.fetch_historical_data(limit=5000)
    
    if training_data is None:
        print("❌ Failed to fetch training data")
        return False
    
    print(f"📊 Loaded {len(training_data)} candles for ML training")
    
    # Train ML models
    trainer = MLStrategyTrainer()
    indicator_engine = MLIndicatorEngine()
    
    print("Training ML models on 20+ technical indicators...")
    models = trainer.train_ml_models(training_data, indicator_engine)
    
    if models:
        print("\n✅ ML TRAINING COMPLETED SUCCESSFULLY!")
        print("Models are ready for backtesting and live trading")
        return True
    else:
        print("\n❌ ML training failed")
        return False

if __name__ == "__main__":
    success = train_ml_models()
    if success:
        print("\n🎯 Next: Run python ml_backtest.py to test the ML strategy")
