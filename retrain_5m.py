#!/usr/bin/env python3
"""
Retrain ML Models for 5-Minute Timeframe
"""

import pandas as pd
from data_manager import DataManager
from ml_trainer import MLStrategyTrainer
from simple_indicators import SimpleMLIndicatorEngine

def retrain_5m_models():
    print("🔄 RETRAINING FOR 5-MINUTE TIMEFRAME")
    print("=" * 50)
    
    dm = DataManager()
    
    # Fetch 5-minute data (fewer candles needed since timeframe is longer)
    print("Fetching 5-minute data for training...")
    data = dm.fetch_historical_data(limit=3000)  # ~10 days of 5m data
    
    if data is None:
        print("❌ Failed to fetch data")
        return False
    
    print(f"✅ Loaded {len(data)} 5-minute candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Train models
    indicator_engine = SimpleMLIndicatorEngine()
    trainer = MLStrategyTrainer()
    
    print("Training models for 5-minute timeframe...")
    models = trainer.train_enhanced_models(data, indicator_engine)
    
    if models:
        print("🎉 5-minute models trained successfully!")
        print("💾 Models saved: xgb_ml_model.pkl, rf_ml_model.pkl, gb_ml_model.pkl")
        return True
    else:
        print("❌ 5-minute training failed")
        return False

if __name__ == "__main__":
    retrain_5m_models()
