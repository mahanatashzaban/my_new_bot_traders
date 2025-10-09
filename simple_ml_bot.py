#!/usr/bin/env python3
"""
Simple ML Bot - Test the trained models
"""

import pandas as pd
import numpy as np
from data_manager import DataManager
from ml_trainer import MLStrategyTrainer
from simple_indicators import SimpleMLIndicatorEngine
from config import TRADING_CONFIG

class SimpleMLBot:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_trainer = MLStrategyTrainer()
        self.indicator_engine = SimpleMLIndicatorEngine()
        self.symbol = TRADING_CONFIG['symbol']
        
    def test_prediction(self):
        """Test prediction with the trained models"""
        print("🤖 TESTING ML MODELS")
        print("=" * 50)
        
        # Load ML models
        if not self.ml_trainer.load_ml_models():
            print("❌ Please train models first: python train_ml_complete.py")
            return
        
        print("✅ ML models loaded successfully")
        
        # Get sufficient data for prediction
        print("Fetching data for prediction...")
        data = self.data_manager.fetch_historical_data(limit=500)
        if data is None:
            print("❌ Failed to fetch data")
            return
        
        print(f"✅ Loaded {len(data)} candles for prediction")
        
        # Get prediction
        prediction = self.ml_trainer.predict_direction(data, self.indicator_engine)
        
        if prediction:
            current_price = data['close'].iloc[-1]
            
            print(f"\n🎯 ML PREDICTION RESULTS:")
            print(f"   Final Prediction: {prediction['prediction']}")
            print(f"   Confidence: {prediction['confidence']:.3f}")
            print(f"   Current Price: ${current_price:.2f}")
            print(f"\n   Model Votes:")
            for model, vote in prediction['model_predictions'].items():
                confidence = prediction['model_confidences'][model]
                print(f"     {model.upper()}: {vote} (conf: {confidence:.3f})")
            
            # Trading recommendation
            if prediction['confidence'] > 0.65:
                if prediction['prediction'] == 'UP':
                    print(f"\n💚 STRONG BUY SIGNAL (Confidence: {prediction['confidence']:.3f})")
                else:
                    print(f"\n🔴 STRONG SELL SIGNAL (Confidence: {prediction['confidence']:.3f})")
            elif prediction['confidence'] > 0.60:
                if prediction['prediction'] == 'UP':
                    print(f"\n📗 MODERATE BUY SIGNAL (Confidence: {prediction['confidence']:.3f})")
                else:
                    print(f"\n📕 MODERATE SELL SIGNAL (Confidence: {prediction['confidence']:.3f})")
            elif prediction['confidence'] > 0.55:
                if prediction['prediction'] == 'UP':
                    print(f"\n🟢 WEAK BUY SIGNAL (Confidence: {prediction['confidence']:.3f})")
                else:
                    print(f"\n🔴 WEAK SELL SIGNAL (Confidence: {prediction['confidence']:.3f})")
            else:
                print(f"\n⚪ HOLD (Low Confidence: {prediction['confidence']:.3f})")
                
        else:
            print("❌ No prediction available")

    def analyze_model_performance(self):
        """Analyze model performance on recent data"""
        print("\n📊 ANALYZING MODEL PERFORMANCE")
        print("=" * 50)
        
        # Load more data for analysis
        data = self.data_manager.fetch_historical_data(limit=2000)  # Increased to 2000
        if data is None:
            print("❌ Failed to fetch data for analysis")
            return
        
        # Load models
        self.ml_trainer.load_ml_models()
        
        predictions = []
        actual_movements = []
        confidences = []
        
        # Start from a point where we have enough data for indicators
        start_index = 300  # Start after enough data for indicators
        
        # Test on recent data points
        test_data = data.iloc[start_index:]
        
        print(f"Testing on {len(test_data)} data points...")
        
        test_count = 0
        for i in range(100, len(test_data), 5):  # Test every 5th point to save time
            historical_data = test_data.iloc[:i+1]
            prediction = self.ml_trainer.predict_direction(historical_data, self.indicator_engine)
            
            if prediction and i + 1 < len(test_data):
                current_price = historical_data['close'].iloc[-1]
                next_price = test_data['close'].iloc[i + 1]
                actual_move = 'UP' if next_price > current_price else 'DOWN'
                
                predictions.append(prediction['prediction'])
                actual_movements.append(actual_move)
                confidences.append(prediction['confidence'])
                test_count += 1
                
                if test_count >= 100:  # Limit to 100 tests for speed
                    break
        
        if predictions:
            correct = sum(1 for p, a in zip(predictions, actual_movements) if p == a)
            accuracy = correct / len(predictions)
            total_tests = len(predictions)
            avg_confidence = np.mean(confidences)
            
            # Calculate performance by confidence levels
            high_conf_indices = [i for i, conf in enumerate(confidences) if conf > 0.65]
            medium_conf_indices = [i for i, conf in enumerate(confidences) if 0.55 <= conf <= 0.65]
            low_conf_indices = [i for i, conf in enumerate(confidences) if conf < 0.55]
            
            high_conf_accuracy = 0
            if high_conf_indices:
                high_conf_correct = sum(1 for i in high_conf_indices if predictions[i] == actual_movements[i])
                high_conf_accuracy = high_conf_correct / len(high_conf_indices)
            
            medium_conf_accuracy = 0
            if medium_conf_indices:
                medium_conf_correct = sum(1 for i in medium_conf_indices if predictions[i] == actual_movements[i])
                medium_conf_accuracy = medium_conf_correct / len(medium_conf_indices)
            
            low_conf_accuracy = 0
            if low_conf_indices:
                low_conf_correct = sum(1 for i in low_conf_indices if predictions[i] == actual_movements[i])
                low_conf_accuracy = low_conf_correct / len(low_conf_indices)
            
            print(f"📈 PERFORMANCE ANALYSIS RESULTS:")
            print(f"   Total Tests: {total_tests}")
            print(f"   Correct Predictions: {correct}")
            print(f"   Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"   Average Confidence: {avg_confidence:.3f}")
            
            print(f"\n📊 ACCURACY BY CONFIDENCE LEVEL:")
            if high_conf_indices:
                print(f"   High Confidence (>0.65): {high_conf_accuracy:.3f} ({len(high_conf_indices)} samples)")
            if medium_conf_indices:
                print(f"   Medium Confidence (0.55-0.65): {medium_conf_accuracy:.3f} ({len(medium_conf_indices)} samples)")
            if low_conf_indices:
                print(f"   Low Confidence (<0.55): {low_conf_accuracy:.3f} ({len(low_conf_indices)} samples)")
            
            print(f"\n🎯 PERFORMANCE ASSESSMENT:")
            if accuracy > 0.60:
                print("   ✅ EXCELLENT - Models performing very well!")
            elif accuracy > 0.55:
                print("   👍 GOOD - Models performing better than random")
            elif accuracy > 0.52:
                print("   ⚠️  MODERATE - Slightly better than random")
            elif accuracy > 0.50:
                print("   🔄 BASELINE - At random level")
            else:
                print("   ❌ POOR - Worse than random")
            
            # Calculate expected value for trading
            if accuracy > 0.50:
                win_rate = accuracy
                avg_win_pct = 0.02  # Assume 2% average win
                avg_loss_pct = 0.015  # Assume 1.5% average loss
                expected_value = (win_rate * avg_win_pct) - ((1 - win_rate) * avg_loss_pct)
                expected_return = expected_value * 100
                
                print(f"\n💹 EXPECTED TRADING PERFORMANCE:")
                print(f"   Expected Value per Trade: {expected_return:+.2f}%")
                print(f"   Projected Annual Return: {(expected_return * 250):+.1f}% (assuming 250 trades/year)")
                
                if expected_return > 1.0:
                    print("   🎉 HIGH POTENTIAL - Trading could be profitable!")
                elif expected_return > 0.5:
                    print("   📈 MODERATE POTENTIAL - Possibly profitable")
                else:
                    print("   ⚠️  LOW POTENTIAL - May not be profitable after costs")
            
            # Model agreement analysis
            print(f"\n🤝 MODEL CONSENSUS ANALYSIS:")
            up_predictions = sum(1 for p in predictions if p == 'UP')
            down_predictions = sum(1 for p in predictions if p == 'DOWN')
            consensus_strength = abs(up_predictions - down_predictions) / total_tests
            
            print(f"   UP Predictions: {up_predictions} ({up_predictions/total_tests*100:.1f}%)")
            print(f"   DOWN Predictions: {down_predictions} ({down_predictions/total_tests*100:.1f}%)")
            print(f"   Consensus Strength: {consensus_strength:.3f}")
            
            if consensus_strength > 0.3:
                print("   ✅ Strong model consensus")
            elif consensus_strength > 0.15:
                print("   ⚠️  Moderate model consensus")
            else:
                print("   🔄 Weak model consensus")
                
        else:
            print("❌ Not enough data for performance analysis")

    def test_multiple_predictions(self):
        """Test multiple predictions to see consistency"""
        print("\n🔄 TESTING MULTIPLE PREDICTIONS")
        print("=" * 50)
        
        if not self.ml_trainer.load_ml_models():
            return
        
        # Test predictions on different time periods
        time_periods = [300, 500, 800, 1000]
        
        predictions_data = []
        
        for period in time_periods:
            data = self.data_manager.fetch_historical_data(limit=period)
            if data is not None:
                prediction = self.ml_trainer.predict_direction(data, self.indicator_engine)
                if prediction:
                    predictions_data.append({
                        'period': period,
                        'prediction': prediction['prediction'],
                        'confidence': prediction['confidence'],
                        'price': data['close'].iloc[-1],
                        'model_votes': prediction['model_predictions']
                    })
        
        if predictions_data:
            print("📋 MULTIPLE PREDICTION TEST:")
            for pred in predictions_data:
                print(f"\n   Data Period: {pred['period']} candles")
                print(f"   Prediction: {pred['prediction']}")
                print(f"   Confidence: {pred['confidence']:.3f}")
                print(f"   Price: ${pred['price']:.2f}")
                print(f"   Model Votes: {pred['model_votes']}")
            
            # Check consistency
            all_same = all(p['prediction'] == predictions_data[0]['prediction'] for p in predictions_data)
            avg_confidence = np.mean([p['confidence'] for p in predictions_data])
            
            print(f"\n🎯 CONSISTENCY ANALYSIS:")
            print(f"   Consistent Predictions: {'✅ YES' if all_same else '❌ NO'}")
            print(f"   Average Confidence: {avg_confidence:.3f}")
            
            if all_same and avg_confidence > 0.60:
                print("   🎉 HIGHLY RELIABLE - Strong consistent signals!")
            elif all_same:
                print("   👍 CONSISTENT - Same direction across timeframes")
            else:
                print("   ⚠️  MIXED - Predictions vary by timeframe")

    def get_market_context(self):
        """Get current market context for the prediction"""
        print("\n📊 MARKET CONTEXT ANALYSIS")
        print("=" * 50)
        
        data = self.data_manager.fetch_historical_data(limit=200)
        if data is None:
            return
        
        current_price = data['close'].iloc[-1]
        price_1h_ago = data['close'].iloc[-60] if len(data) > 60 else data['close'].iloc[0]
        price_24h_ago = data['close'].iloc[0]
        
        change_1h = (current_price - price_1h_ago) / price_1h_ago * 100
        change_24h = (current_price - price_24h_ago) / price_24h_ago * 100
        
        # Calculate volatility
        volatility = data['close'].pct_change().std() * 100
        
        print(f"💰 CURRENT MARKET:")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   1H Change: {change_1h:+.2f}%")
        print(f"   24H Change: {change_24h:+.2f}%")
        print(f"   Volatility: {volatility:.2f}%")
        
        # Market trend
        if change_1h > 0.5:
            trend = "🟢 STRONG UPTREND"
        elif change_1h > 0.1:
            trend = "📗 MILD UPTREND"
        elif change_1h < -0.5:
            trend = "🔴 STRONG DOWNTREND"
        elif change_1h < -0.1:
            trend = "📕 MILD DOWNTREND"
        else:
            trend = "⚪ SIDEWAYS"
        
        print(f"   Short-term Trend: {trend}")
        
        # Volatility assessment
        if volatility > 1.0:
            vol_assessment = "🔥 HIGH VOLATILITY"
        elif volatility > 0.5:
            vol_assessment = "⚡ MODERATE VOLATILITY"
        else:
            vol_assessment = "💤 LOW VOLATILITY"
        
        print(f"   Volatility: {vol_assessment}")

if __name__ == "__main__":
    bot = SimpleMLBot()
    
    print("🚀 ML TRADING BOT - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Run all tests
    bot.test_prediction()
    bot.get_market_context()
    bot.test_multiple_predictions()
    bot.analyze_model_performance()
    
    print("\n" + "=" * 60)
    print("🎯 TESTING COMPLETE")
    print("=" * 60)
    print("Next: Run 'python ml_backtest.py' for full strategy backtest")
    print("Then: Run 'python ml_live_bot.py' for live simulation")
