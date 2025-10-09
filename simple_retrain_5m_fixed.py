#!/usr/bin/env python3
"""
Simple 5-Minute Retraining - With Fallback
"""

import pandas as pd
from data_manager_simple import DataManager
from simple_indicators import SimpleMLIndicatorEngine

# Try to use simple trainer first
try:
    from ml_trainer_simple import SimpleMLTrainer as MLTrainer
    print("✅ Using Simple ML Trainer (No XGBoost)")
except ImportError:
    try:
        from ml_trainer import MLStrategyTrainer as MLTrainer
        print("✅ Using Full ML Trainer (With XGBoost)")
    except ImportError:
        print("❌ No ML trainer available")
        exit(1)

def retrain_5m_fixed():
    print("🔄 5-MINUTE RETRAINING")
    print("=" * 50)
    
    dm = DataManager()
    
    print("Fetching 5-minute training data...")
    data = dm.fetch_historical_data(limit=2000)
    
    if data is None or len(data) < 500:
        print("❌ Not enough 5-minute data")
        return False
    
    print(f"✅ Loaded {len(data)} 5-minute candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Train models
    indicator_engine = SimpleMLIndicatorEngine()
    trainer = MLTrainer()
    
    # Use appropriate training method
    if hasattr(trainer, 'train_simple_models'):
        models = trainer.train_simple_models(data, indicator_engine)
    else:
        models = trainer.train_enhanced_models(data, indicator_engine)
    
    if models:
        print("🎉 5-minute models trained successfully!")
        return True
    else:
        print("❌ 5-minute training failed")
        return False

if __name__ == "__main__":
    retrain_5m_fixed()
