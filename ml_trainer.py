#!/usr/bin/env python3
"""
ML Trainer for Pure Indicator-Based Strategy - Adaptive Labeling
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLStrategyTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.training_metrics = {}
    
    def prepare_features(self, df, indicator_engine):
        """Prepare features for ML training with validation"""
        try:
            # Calculate all indicators
            df_with_indicators = indicator_engine.calculate_all_indicators(df.copy())
            
            # Get feature columns
            feature_cols = indicator_engine.get_feature_columns()
            
            # Drop rows with NaN values
            df_clean = df_with_indicators.dropna()
            
            if len(df_clean) < 100:
                raise ValueError(f"Not enough clean data: {len(df_clean)} samples")
            
            # Select features
            X = df_clean[feature_cols]
            
            print(f"✅ Feature preparation: {X.shape[0]} samples, {X.shape[1]} features")
            return X, df_clean
            
        except Exception as e:
            print(f"❌ Feature preparation failed: {e}")
            return None, None
    
    def analyze_price_movements(self, df):
        """Analyze actual price movements to set appropriate thresholds"""
        print("Analyzing price movements...")
        
        # Calculate 1-minute price changes
        price_changes = df['close'].pct_change().dropna()
        
        # Calculate volatility
        volatility = price_changes.std()
        avg_change = price_changes.abs().mean()
        max_change = price_changes.abs().max()
        
        print(f"📊 Price movement analysis:")
        print(f"   Average absolute change: {avg_change*100:.4f}%")
        print(f"   Volatility (std): {volatility*100:.4f}%")
        print(f"   Maximum change: {max_change*100:.4f}%")
        
        # Set thresholds based on actual volatility
        threshold_1min = volatility * 0.5  # Half of volatility
        threshold_3min = volatility * 1.0  # Full volatility
        threshold_5min = volatility * 1.5  # 1.5x volatility
        
        return {
            'volatility': volatility,
            'threshold_1min': threshold_1min,
            'threshold_3min': threshold_3min,
            'threshold_5min': threshold_5min
        }
    
    def create_adaptive_labels(self, df):
        """
        Create labels based on actual price volatility
        """
        print("Creating adaptive labels based on volatility...")
        
        # Analyze price movements
        movement_stats = self.analyze_price_movements(df)
        threshold = movement_stats['threshold_3min']  # Use 3-minute threshold
        
        print(f"Using adaptive threshold: {threshold*100:.4f}%")
        
        labels = []
        
        for i in range(len(df) - 3):  # 3-bar lookahead
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+4]  # Next 3 bars
            
            future_max = future_prices.max()
            future_min = future_prices.min()
            
            upside = (future_max - current_price) / current_price
            downside = (current_price - future_min) / current_price
            
            # Use volatility-based threshold
            if upside > threshold and upside > downside:
                labels.append(1)  # BUY
            elif downside > threshold and downside > upside:
                labels.append(0)  # SELL
            else:
                labels.append(-1)  # HOLD
        
        # Pad the end
        labels.extend([-1] * min(3, len(df)))
        
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        print(f"Adaptive label distribution: BUY={label_dist.get(1, 0)}, SELL={label_dist.get(0, 0)}, HOLD={label_dist.get(-1, 0)}")
        
        return labels
    
    def create_trend_following_labels(self, df, window=5):
        """
        Create labels based on trend following (simpler approach)
        """
        print("Creating trend-following labels...")
        
        labels = []
        
        for i in range(len(df) - window):
            current_trend = self.calculate_trend(df['close'].iloc[:i+1])
            future_trend = self.calculate_trend(df['close'].iloc[i+1:i+window+1])
            
            # If trends align, trade in that direction
            if current_trend > 0.1 and future_trend > 0.1:
                labels.append(1)  # BUY - uptrend continues
            elif current_trend < -0.1 and future_trend < -0.1:
                labels.append(0)  # SELL - downtrend continues
            else:
                labels.append(-1)  # HOLD - unclear or reversal
        
        # Pad the end
        labels.extend([-1] * min(window, len(df)))
        
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        print(f"Trend label distribution: BUY={label_dist.get(1, 0)}, SELL={label_dist.get(0, 0)}, HOLD={label_dist.get(-1, 0)}")
        
        return labels
    
    def calculate_trend(self, prices):
        """Calculate trend strength"""
        if len(prices) < 2:
            return 0
        returns = prices.pct_change().dropna()
        if returns.std() == 0:
            return 0
        return returns.mean() / (returns.std() + 1e-8)  # Sharpe-like ratio
    
    def create_momentum_labels(self, df, window=3):
        """
        Create labels based on momentum (price acceleration)
        """
        print("Creating momentum labels...")
        
        labels = []
        
        for i in range(len(df) - window):
            current_momentum = self.calculate_momentum(df['close'].iloc[max(0, i-5):i+1])
            future_momentum = self.calculate_momentum(df['close'].iloc[i+1:i+window+1])
            
            # If momentum is strong and continues
            if current_momentum > 0.001 and future_momentum > 0.001:
                labels.append(1)  # BUY
            elif current_momentum < -0.001 and future_momentum < -0.001:
                labels.append(0)  # SELL
            else:
                labels.append(-1)  # HOLD
        
        # Pad the end
        labels.extend([-1] * min(window, len(df)))
        
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        print(f"Momentum label distribution: BUY={label_dist.get(1, 0)}, SELL={label_dist.get(0, 0)}, HOLD={label_dist.get(-1, 0)}")
        
        return labels
    
    def calculate_momentum(self, prices):
        """Calculate momentum (price acceleration)"""
        if len(prices) < 2:
            return 0
        returns = prices.pct_change().dropna()
        if len(returns) < 2:
            return 0
        # Momentum as the change in returns
        return returns.iloc[-1] - returns.mean()
    
    def create_breakout_labels(self, df, window=10):
        """
        Create labels based on breakout patterns
        """
        print("Creating breakout labels...")
        
        labels = []
        
        for i in range(window, len(df) - 2):
            current_data = df['close'].iloc[i-window:i+1]
            resistance = current_data.max()
            support = current_data.min()
            current_price = df['close'].iloc[i]
            
            future_prices = df['close'].iloc[i+1:i+3]
            future_max = future_prices.max()
            future_min = future_prices.min()
            
            # Breakout above resistance
            if current_price >= resistance * 0.999 and future_max > resistance:
                labels.append(1)  # BUY - breakout up
            # Breakdown below support
            elif current_price <= support * 1.001 and future_min < support:
                labels.append(0)  # SELL - breakdown
            else:
                labels.append(-1)  # HOLD
        
        # Pad the beginning and end
        labels = [-1] * window + labels
        labels.extend([-1] * min(2, len(df)))
        
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        print(f"Breakout label distribution: BUY={label_dist.get(1, 0)}, SELL={label_dist.get(0, 0)}, HOLD={label_dist.get(-1, 0)}")
        
        return labels
    
    def create_binary_classification(self, df):
        """
        Simple binary classification: predict if next price will be higher
        """
        print("Creating binary classification labels...")
        
        labels = []
        
        for i in range(len(df) - 1):
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]
            
            # Simple binary: 1 if price goes up, 0 if down
            if next_price > current_price:
                labels.append(1)
            else:
                labels.append(0)
        
        # Pad the end
        labels.append(0)  # Last point
        
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        print(f"Binary label distribution: UP={label_dist.get(1, 0)}, DOWN={label_dist.get(0, 0)}")
        
        return labels
    
    def train_enhanced_models(self, df, indicator_engine):
        """Train enhanced ML models with adaptive labeling"""
        print("🤖 TRAINING ENHANCED ML MODELS...")
        print("=" * 50)
        
        # Prepare features
        X, df_clean = self.prepare_features(df, indicator_engine)
        if X is None:
            return None
        
        # Try different labeling strategies
        labeling_strategies = [
            ("Binary", self.create_binary_classification),
            ("Adaptive", self.create_adaptive_labels),
            ("Breakout", self.create_breakout_labels),
            ("Momentum", self.create_momentum_labels),
            ("Trend", self.create_trend_following_labels),
        ]
        
        best_X = None
        best_y = None
        best_strategy = None
        best_sample_count = 0
        
        for strategy_name, label_func in labeling_strategies:
            print(f"\nTrying {strategy_name} labeling...")
            try:
                y = label_func(df_clean)
                
                # For binary classification, we use all data
                if strategy_name == "Binary":
                    X_binary = X[:len(y)]
                    y_binary = np.array(y)
                    sample_count = len(X_binary)
                else:
                    # Filter out unclear movements (-1 labels)
                    binary_mask = np.array(y) != -1
                    X_binary = X[binary_mask]
                    y_binary = np.array(y)[binary_mask]
                    sample_count = len(X_binary)
                
                print(f"   Samples: {sample_count}")
                
                # Update best strategy if this one has more samples
                if sample_count > best_sample_count:
                    best_X = X_binary
                    best_y = y_binary
                    best_strategy = strategy_name
                    best_sample_count = sample_count
                    
                # If we have enough data, use this strategy
                if sample_count > 1000:
                    break
                    
            except Exception as e:
                print(f"   ❌ {strategy_name} failed: {e}")
                continue
        
        if best_X is None or len(best_X) < 50:
            print("❌ No labeling strategy produced sufficient training data")
            print("💡 Try collecting more data or using different timeframes")
            return None
        
        print(f"\n📊 Using {best_strategy} labeling: {len(best_X)} samples")
        print(f"   Class distribution: {np.unique(best_y, return_counts=True)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            best_X, best_y, test_size=0.2, random_state=42, shuffle=True, stratify=best_y
        )
        
        print(f"🔧 Training set: {len(X_train)} samples")
        print(f"🔧 Testing set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models based on data size
        if len(X_train) > 1000:
            # Full models for large datasets
            print("\nTraining full models...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1,
                random_state=42
            )
            rf_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            )
            gb_model = GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=6, 
                random_state=42
            )
        else:
            # Simpler models for smaller datasets
            print("\nTraining simplified models (small dataset)...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=50, 
                max_depth=4, 
                random_state=42
            )
            rf_model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=6, 
                random_state=42,
                n_jobs=-1
            )
            gb_model = GradientBoostingClassifier(
                n_estimators=50, 
                max_depth=4, 
                random_state=42
            )
        
        print("Training XGBoost...")
        xgb_model.fit(X_train_scaled, y_train)
        self.models['xgb'] = xgb_model
        
        print("Training Random Forest...")
        rf_model.fit(X_train_scaled, y_train)
        self.models['rf'] = rf_model
        
        print("Training Gradient Boosting...")
        gb_model.fit(X_train_scaled, y_train)
        self.models['gb'] = gb_model
        
        # Evaluate models
        self.evaluate_enhanced_models(X_test_scaled, y_test)
        
        # Save models
        self.save_models()
        
        print("💾 All ML models saved successfully")
        return self.models
    
    def evaluate_enhanced_models(self, X_test, y_test):
        """Enhanced model evaluation"""
        print("\n📊 MODEL EVALUATION:")
        print("=" * 50)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{name.upper()} Model:")
            print(f"  Accuracy: {accuracy:.4f}")
            
            if len(np.unique(y_test)) > 2:
                print(classification_report(y_test, y_pred, target_names=['SELL', 'BUY']))
            else:
                print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
    
    def save_models(self):
        """Save all models and scaler"""
        for name, model in self.models.items():
            joblib.dump(model, f'{name}_ml_model.pkl')
        joblib.dump(self.scaler, 'ml_scaler.pkl')
        print("💾 Models saved as: xgb_ml_model.pkl, rf_ml_model.pkl, gb_ml_model.pkl, ml_scaler.pkl")
    
    def load_ml_models(self):
        """Load trained ML models"""
        try:
            self.models['xgb'] = joblib.load('xgb_ml_model.pkl')
            self.models['rf'] = joblib.load('rf_ml_model.pkl')
            self.models['gb'] = joblib.load('gb_ml_model.pkl')
            self.scaler = joblib.load('ml_scaler.pkl')
            print("✅ ML models loaded successfully")
            return True
        except FileNotFoundError as e:
            print(f"❌ No trained ML models found: {e}")
            return False
    
    def predict_direction(self, df, indicator_engine):
        """Predict market direction using ensemble of ML models"""
        if not self.models:
            print("❌ No models loaded")
            return None
        
        # Prepare features for prediction
        X, _ = self.prepare_features(df, indicator_engine)
        
        # Check if we got valid features
        if X is None or X.empty or len(X) == 0:
            # print("❌ Not enough data for prediction")
            return None
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            print(f"❌ Error scaling features: {e}")
            return None
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[-1]
                proba = model.predict_proba(X_scaled)[-1]
                confidence = max(proba)
                
                # Handle binary vs multi-class
                if len(proba) == 2:  # Binary classification
                    predictions[name] = 'UP' if pred == 1 else 'DOWN'
                else:  # Multi-class
                    predictions[name] = 'UP' if pred == 1 else 'DOWN'
                
                confidences[name] = confidence
            except Exception as e:
                # print(f"❌ Prediction error with {name}: {e}")
                continue
        
        if not predictions:
            return None
        
        # Ensemble prediction
        up_votes = sum(1 for p in predictions.values() if p == 'UP')
        down_votes = sum(1 for p in predictions.values() if p == 'DOWN')
        
        final_prediction = 'UP' if up_votes > down_votes else 'DOWN'
        avg_confidence = np.mean(list(confidences.values()))
        
        return {
            'prediction': final_prediction,
            'confidence': avg_confidence,
            'model_predictions': predictions,
            'model_confidences': confidences
        }

    def get_model_info(self):
        """Get information about trained models"""
        if not self.models:
            print("❌ No models loaded")
            return None
        
        info = {}
        for name, model in self.models.items():
            info[name] = {
                'type': type(model).__name__,
                'features_used': getattr(model, 'n_features_in_', 'Unknown'),
                'parameters': model.get_params() if hasattr(model, 'get_params') else 'N/A'
            }
        
        return info

# For backward compatibility
MLStrategyTrainer.train_ml_models = MLStrategyTrainer.train_enhanced_models

# Simple test function
def test_ml_trainer():
    """Test the ML trainer"""
    from data_manager import DataManager
    from simple_indicators import SimpleMLIndicatorEngine
    
    print("🧪 TESTING ML TRAINER")
    print("=" * 50)
    
    # Load sample data
    dm = DataManager()
    data = dm.fetch_historical_data(limit=1000)
    
    if data is not None:
        indicator_engine = SimpleMLIndicatorEngine()
        trainer = MLStrategyTrainer()
        
        print("Testing feature preparation...")
        X, df_clean = trainer.prepare_features(data, indicator_engine)
        
        if X is not None:
            print(f"✅ Feature preparation successful: {X.shape}")
            
            print("\nTesting model training...")
            models = trainer.train_enhanced_models(data, indicator_engine)
            
            if models:
                print("✅ Model training successful!")
                
                print("\nTesting prediction...")
                prediction = trainer.predict_direction(data, indicator_engine)
                if prediction:
                    print(f"✅ Prediction successful: {prediction['prediction']} (confidence: {prediction['confidence']:.2f})")
                else:
                    print("❌ Prediction failed")
            else:
                print("❌ Model training failed")
        else:
            print("❌ Feature preparation failed")
    else:
        print("❌ Data loading failed")

if __name__ == "__main__":
    test_ml_trainer()
