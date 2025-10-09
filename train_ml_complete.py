#!/usr/bin/env python3
"""
Complete ML Training Pipeline - Start from Scratch
"""

import pandas as pd
import numpy as np
import joblib
import os
from data_manager import DataManager
from ml_trainer import MLStrategyTrainer
from simple_indicators import SimpleMLIndicatorEngine

def collect_training_data():
    """Collect substantial historical data for training"""
    print("📊 COLLECTING TRAINING DATA...")
    print("=" * 50)
    
    dm = DataManager()
    
    # Fetch large dataset for robust training
    print("Fetching historical data (this may take a while)...")
    data = dm.fetch_historical_data(limit=5000)
    
    if data is None or len(data) < 1000:
        print("❌ Insufficient data for training")
        return None
    
    print(f"✅ Collected {len(data)} candles for training")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    return data

def train_complete_pipeline():
    """Complete ML training pipeline"""
    print("🤖 STARTING COMPLETE ML TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Collect data
    training_data = collect_training_data()
    if training_data is None:
        return False
    
    # Step 2: Initialize components
    indicator_engine = SimpleMLIndicatorEngine()
    ml_trainer = MLStrategyTrainer()
    
    # Step 3: Calculate all indicators
    print("\n📈 CALCULATING TECHNICAL INDICATORS...")
    data_with_indicators = indicator_engine.calculate_all_indicators(training_data.copy())
    
    # Check if we have enough data after indicator calculation
    data_clean = data_with_indicators.dropna()
    if len(data_clean) < 500:
        print(f"❌ Not enough clean data after indicator calculation: {len(data_clean)} samples")
        return False
    
    print(f"✅ Clean data samples: {len(data_clean)}")
    
    # Step 4: Train ML models
    print("\n🎯 TRAINING ML MODELS...")
    models = ml_trainer.train_enhanced_models(data_with_indicators, indicator_engine)
    
    if models:
        print("\n✅ ML TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Trained models: {list(models.keys())}")
        print(f"Feature count: {len(indicator_engine.get_feature_columns())}")
        return True
    else:
        print("\n❌ ML training failed")
        return False

def verify_training():
    """Verify that models were trained correctly"""
    print("\n🔍 VERIFYING TRAINED MODELS...")
    print("=" * 50)
    
    try:
        # Try to load models
        xgb_model = joblib.load('xgb_ml_model.pkl')
        rf_model = joblib.load('rf_ml_model.pkl')
        gb_model = joblib.load('gb_ml_model.pkl')
        scaler = joblib.load('ml_scaler.pkl')
        
        print("✅ All models loaded successfully:")
        print(f"   - XGBoost: {type(xgb_model)}")
        print(f"   - Random Forest: {type(rf_model)}")
        print(f"   - Gradient Boosting: {type(gb_model)}")
        print(f"   - Scaler: {type(scaler)}")
        
        # Test with sample data
        dm = DataManager()
        test_data = dm.fetch_historical_data(limit=100)
        if test_data is not None:
            indicator_engine = SimpleMLIndicatorEngine()
            ml_trainer = MLStrategyTrainer()
            ml_trainer.load_ml_models()
            
            prediction = ml_trainer.predict_direction(test_data, indicator_engine)
            if prediction:
                print(f"✅ Model test prediction: {prediction['prediction']} (Confidence: {prediction['confidence']:.2f})")
            else:
                print("❌ Model test prediction failed")
                
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 COMPLETE ML TRAINING FROM SCRATCH")
    print("Using SIMPLE MANUAL INDICATORS (no ta library dependency)")
    print("=" * 60)
    
    success = train_complete_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 TRAINING PIPELINE COMPLETED!")
        print("=" * 60)
        
        # Verify training
        verify_training()
        
        print("\n📝 NEXT STEPS:")
        print("1. Run backtest: python ml_backtest.py")
        print("2. Start live bot: python ml_trading_bot.py")
        print("3. Monitor performance and retrain periodically")
        
    else:
        print("\n❌ TRAINING PIPELINE FAILED")
        print("Check your data connection and try again")
