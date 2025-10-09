#!/usr/bin/env python3
"""
Simple 5-Minute Retraining - No Dependencies
"""

import pandas as pd
from data_manager_simple import DataManager
from ml_trainer import MLStrategyTrainer
from simple_indicators import SimpleMLIndicatorEngine

def retrain_5m_simple():
    print("🔄 SIMPLE 5-MINUTE RETRAINING")
    print("=" * 50)
    
    dm = DataManager()
    
    print("Fetching 5-minute training data...")
    data = dm.fetch_historical_data(limit=2000)  # ~7 days of 5m data
    
    if data is None or len(data) < 500:
        print("❌ Not enough 5-minute data")
        return False
    
    print(f"✅ Loaded {len(data)} 5-minute candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Train models
    indicator_engine = SimpleMLIndicatorEngine()
    trainer = MLStrategyTrainer()
    
    print("Training 5-minute models...")
    models = trainer.train_enhanced_models(data, indicator_engine)
    
    if models:
        print("🎉 5-minute models trained successfully!")
        return True
    else:
        print("❌ 5-minute training failed")
        return False

if __name__ == "__main__":
    retrain_5m_simple()
