#!/usr/bin/env python3
"""
Test 5-Minute Trading Bot
"""

from data_manager_simple import DataManager
from ml_trainer_simple import SimpleMLTrainer
from simple_indicators import SimpleMLIndicatorEngine

def test_5m_bot():
    print("🧪 TESTING 5-MINUTE TRADING BOT")
    print("=" * 50)
    
    dm = DataManager()
    trainer = SimpleMLTrainer()
    indicator_engine = SimpleMLIndicatorEngine()
    
    if not trainer.load_ml_models():
        print("❌ Please train 5m models first")
        return
    
    print("✅ 5-minute models loaded")
    
    # Get 5-minute data
    data = dm.fetch_historical_data(limit=300)
    if data is None:
        print("❌ Failed to fetch 5m data")
        return
    
    print(f"✅ Loaded {len(data)} 5-minute candles")
    
    # Test multiple predictions
    print("\n📊 TESTING MULTIPLE PREDICTIONS:")
    print("-" * 40)
    
    for i in range(5):
        # Use different data slices
        test_data = data.iloc[:100 + (i * 50)]
        prediction = trainer.predict_direction(test_data, indicator_engine)
        
        if prediction:
            current_price = test_data['close'].iloc[-1]
            print(f"Test {i+1}:")
            print(f"  Signal: {prediction['prediction']}")
            print(f"  Confidence: {prediction['confidence']:.3f}")
            print(f"  Models: {prediction['model_predictions']}")
            print(f"  Price: ${current_price:.2f}")
            print()
        else:
            print(f"Test {i+1}: No prediction")
            print()

if __name__ == "__main__":
    test_5m_bot()
