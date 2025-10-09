#!/usr/bin/env python3
"""
Aggressive ML Trainer for More Trading Signals
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

class AggressiveMLTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}

    def prepare_features(self, df, indicator_engine):
        """Prepare features - more lenient on data cleaning"""
        try:
            df_with_indicators = indicator_engine.calculate_all_indicators(df.copy())
            feature_cols = indicator_engine.get_feature_columns()
            
            # Use forward fill to handle NaN values instead of dropping
            df_clean = df_with_indicators.ffill().dropna()
            
            if len(df_clean) < 50:  # Lower threshold
                raise ValueError(f"Not enough clean data: {len(df_clean)} samples")

            X = df_clean[feature_cols]
            print(f"✅ Feature preparation: {X.shape[0]} samples, {X.shape[1]} features")
            return X, df_clean

        except Exception as e:
            print(f"❌ Feature preparation failed: {e}")
            return None, None

    def create_aggressive_labels(self, df, future_bars=1, threshold=0.001):
        """
        More aggressive labeling for scalping
        """
        print("Creating aggressive labels (LONG/SHORT/HOLD)...")
        labels = []

        for i in range(len(df) - future_bars):
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + future_bars]
            
            price_change = (future_price - current_price) / current_price
            
            # More aggressive thresholds for scalping
            if price_change > threshold:
                labels.append(2)  # LONG
            elif price_change < -threshold:
                labels.append(0)  # SHORT
            else:
                labels.append(1)  # HOLD

        # Pad the end
        labels.extend([1] * min(future_bars, len(df)))

        unique, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
        distribution = dict(zip([label_names[x] for x in unique], counts))
        print(f"Aggressive label distribution: {distribution}")
        
        total = sum(distribution.values())
        active_trades = distribution.get('LONG', 0) + distribution.get('SHORT', 0)
        print(f"Active trades: {active_trades}/{total} ({active_trades/total*100:.1f}%)")
        
        return labels

    def train_aggressive_models(self, df, indicator_engine):
        """Train models with more aggressive labeling"""
        print("🤖 TRAINING AGGRESSIVE MODELS FOR SCALPING")
        print("=" * 50)

        # Prepare features
        X, df_clean = self.prepare_features(df, indicator_engine)
        if X is None:
            return None

        # Use aggressive labeling
        y = self.create_aggressive_labels(df_clean)
        X = X[:len(y)]  # Align shapes

        print(f"📊 Training data: {len(X)} total samples")
        print(f"   LONG: {sum(1 for label in y if label == 2)}")
        print(f"   SHORT: {sum(1 for label in y if label == 0)}")
        print(f"   HOLD: {sum(1 for label in y if label == 1)}")

        if len(X) < 100:
            print("❌ Not enough training samples")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"🔧 Training set: {len(X_train)} samples")
        print(f"🔧 Testing set: {len(X_test)} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest - tuned for more signals
        print("\nTraining Aggressive Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=50,  # Fewer trees for faster, more aggressive signals
            max_depth=8,      # Shallower trees
            min_samples_split=10,  # More generalizable
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['rf'] = rf_model

        # Train Gradient Boosting - tuned for scalping
        print("Training Aggressive Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=15,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['gb'] = gb_model

        # Evaluate models
        self.evaluate_models(X_test_scaled, y_test)

        # Save models
        self.save_models()

        print("💾 Aggressive models saved successfully")
        return self.models

    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 AGGRESSIVE MODEL EVALUATION:")
        print("=" * 50)

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"\n{name.upper()} Model:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred, target_names=['SHORT', 'HOLD', 'LONG']))

    def save_models(self):
        """Save models and scaler"""
        for name, model in self.models.items():
            joblib.dump(model, f'aggressive_{name}_model.pkl')
        joblib.dump(self.scaler, 'aggressive_scaler.pkl')

    def load_ml_models(self):
        """Load trained aggressive models"""
        try:
            self.models['rf'] = joblib.load('aggressive_rf_model.pkl')
            self.models['gb'] = joblib.load('aggressive_gb_model.pkl')
            self.scaler = joblib.load('aggressive_scaler.pkl')
            print("✅ Aggressive ML models loaded successfully")
            return True
        except FileNotFoundError as e:
            print(f"❌ No aggressive models found: {e}")
            return False

    def predict_direction(self, df, indicator_engine):
        """Predict with lower confidence threshold for more signals"""
        if not self.models:
            print("❌ No models loaded")
            return None

        # Prepare features for prediction
        X, _ = self.prepare_features(df, indicator_engine)

        if X is None or X.empty or len(X) == 0:
            return None

        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            print(f"❌ Error scaling features: {e}")
            return None

        # Get predictions
        predictions = {}
        confidences = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[-1]
                proba = model.predict_proba(X_scaled)[-1]
                confidence = max(proba)

                # Map predictions to trading actions
                if pred == 2:  # LONG
                    predictions[name] = 'LONG'
                elif pred == 0:  # SHORT
                    predictions[name] = 'SHORT'
                else:  # HOLD
                    predictions[name] = 'HOLD'

                confidences[name] = confidence
            except Exception as e:
                print(f"❌ Prediction error with {name}: {e}")
                continue

        if not predictions:
            return None

        # Count votes - more aggressive: require only 1 vote for action
        long_votes = sum(1 for p in predictions.values() if p == 'LONG')
        short_votes = sum(1 for p in predictions.values() if p == 'SHORT')

        # More aggressive signal generation
        if long_votes >= 1:  # Only need 1 vote for LONG
            final_prediction = 'LONG'
        elif short_votes >= 1:  # Only need 1 vote for SHORT
            final_prediction = 'SHORT'
        else:
            final_prediction = 'HOLD'

        avg_confidence = np.mean(list(confidences.values()))

        return {
            'prediction': final_prediction,
            'confidence': avg_confidence,
            'model_predictions': predictions,
            'model_confidences': confidences,
            'vote_count': {'LONG': long_votes, 'SHORT': short_votes, 'HOLD': 2 - (long_votes + short_votes)}
        }


def train_aggressive_models_main():
    """Main function to train aggressive models"""
    from data_manager_enhanced import EnhancedDataManager
    from simple_indicators import SimpleMLIndicatorEngine
    
    print("🚀 TRAINING AGGRESSIVE MODELS FOR SCALPING")
    print("=" * 50)
    
    dm = EnhancedDataManager()
    
    # Fetch data for training
    print("Fetching 5-minute training data...")
    data = dm.fetch_historical_data(limit=2000)  # Less data for faster training
    
    if data is None or len(data) < 500:
        print("❌ Not enough training data")
        return False
    
    print(f"✅ Loaded {len(data)} 5-minute candles")
    
    # Train models
    indicator_engine = SimpleMLIndicatorEngine()
    trainer = AggressiveMLTrainer()
    
    models = trainer.train_aggressive_models(data, indicator_engine)
    
    if models:
        print("\n🎉 Aggressive models trained successfully!")
        print("💾 Models saved: aggressive_rf_model.pkl, aggressive_gb_model.pkl")
        return True
    else:
        print("\n❌ Aggressive training failed")
        return False


if __name__ == "__main__":
    train_aggressive_models_main()
