#!/usr/bin/env python3
"""
Bidirectional ML Trainer - LONG & SHORT signals
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

class BidirectionalMLTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}

    def prepare_features(self, df, indicator_engine):
        """Prepare features for ML training"""
        try:
            df_with_indicators = indicator_engine.calculate_all_indicators(df.copy())
            feature_cols = indicator_engine.get_feature_columns()
            df_clean = df_with_indicators.dropna()

            if len(df_clean) < 100:
                raise ValueError(f"Not enough clean data: {len(df_clean)} samples")

            X = df_clean[feature_cols]
            print(f"✅ Feature preparation: {X.shape[0]} samples, {X.shape[1]} features")
            return X, df_clean

        except Exception as e:
            print(f"❌ Feature preparation failed: {e}")
            return None, None

    def create_3way_labels(self, df, future_bars=2, threshold=0.003):
        """
        Create 3-way labels: LONG, SHORT, HOLD
        Based on 2-bar future movement with 0.3% threshold
        """
        print("Creating 3-way labels (LONG/SHORT/HOLD)...")
        labels = []

        for i in range(len(df) - future_bars):
            current_price = df['close'].iloc[i]
            future_max = df['close'].iloc[i+1:i+future_bars+1].max()
            future_min = df['close'].iloc[i+1:i+future_bars+1].min()

            upside = (future_max - current_price) / current_price
            downside = (current_price - future_min) / current_price

            # 3-way classification for scalping
            if upside > threshold and upside > downside:
                labels.append(2)  # LONG (price will go up)
            elif downside > threshold and downside > upside:
                labels.append(0)  # SHORT (price will go down)
            else:
                labels.append(1)  # HOLD (no clear movement)

        # Pad the end
        labels.extend([1] * min(future_bars, len(df)))

        unique, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
        print(f"Label distribution: {dict(zip([label_names[x] for x in unique], counts))}")

        return labels

    def train_bidirectional_models(self, df, indicator_engine):
        """Train models for LONG/SHORT trading"""
        print("🤖 TRAINING BIDIRECTIONAL MODELS (LONG & SHORT)")
        print("=" * 50)

        # Prepare features
        X, df_clean = self.prepare_features(df, indicator_engine)
        if X is None:
            return None

        # Create 3-way labels
        y = self.create_3way_labels(df_clean)
        X = X[:len(y)]  # Align shapes

        # Filter out HOLD labels for active trading
        active_mask = np.array(y) != 1
        X_active = X[active_mask]
        y_active = np.array(y)[active_mask]

        print(f"📊 Training data: {len(X_active)} active trades")
        print(f"   LONG: {sum(y_active == 2)}, SHORT: {sum(y_active == 0)}")

        if len(X_active) < 100:
            print("❌ Not enough active trading samples")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_active, y_active, test_size=0.2, random_state=42, stratify=y_active
        )

        print(f"🔧 Training set: {len(X_train)} samples")
        print(f"🔧 Testing set: {len(X_test)} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['rf'] = rf_model

        # Train Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['gb'] = gb_model

        # Evaluate models
        self.evaluate_bidirectional_models(X_test_scaled, y_test)

        # Save models
        self.save_models()

        print("💾 Bidirectional models saved successfully")
        return self.models

    def evaluate_bidirectional_models(self, X_test, y_test):
        """Evaluate 3-way classification performance"""
        print("\n📊 BIDIRECTIONAL MODEL EVALUATION:")
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
            joblib.dump(model, f'bidirectional_{name}_model.pkl')
        joblib.dump(self.scaler, 'bidirectional_scaler.pkl')

    def load_ml_models(self):
        """Load trained bidirectional models"""
        try:
            self.models['rf'] = joblib.load('bidirectional_rf_model.pkl')
            self.models['gb'] = joblib.load('bidirectional_gb_model.pkl')
            self.scaler = joblib.load('bidirectional_scaler.pkl')
            print("✅ Bidirectional ML models loaded successfully")
            return True
        except FileNotFoundError as e:
            print(f"❌ No bidirectional models found: {e}")
            return False

    def predict_direction(self, df, indicator_engine):
        """Predict market direction with 3-way classification"""
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
        probabilities = {}

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
                probabilities[name] = proba
            except Exception as e:
                print(f"❌ Prediction error with {name}: {e}")
                continue

        if not predictions:
            return None

        # Count votes
        long_votes = sum(1 for p in predictions.values() if p == 'LONG')
        short_votes = sum(1 for p in predictions.values() if p == 'SHORT')
        hold_votes = sum(1 for p in predictions.values() if p == 'HOLD')

        # Determine final prediction
        if long_votes > short_votes and long_votes > hold_votes:
            final_prediction = 'LONG'
        elif short_votes > long_votes and short_votes > hold_votes:
            final_prediction = 'SHORT'
        else:
            final_prediction = 'HOLD'

        avg_confidence = np.mean(list(confidences.values()))

        return {
            'prediction': final_prediction,
            'confidence': avg_confidence,
            'model_predictions': predictions,
            'model_confidences': confidences,
            'vote_count': {'LONG': long_votes, 'SHORT': short_votes, 'HOLD': hold_votes}
        }


def train_bidirectional_models_main():
    """Main function to train bidirectional models"""
    from data_manager_simple import DataManager
    from simple_indicators import SimpleMLIndicatorEngine
    
    print("🚀 TRAINING BIDIRECTIONAL MODELS")
    print("=" * 50)
    
    dm = DataManager()
    
    # Fetch 5-minute data for training
    print("Fetching 5-minute training data...")
    data = dm.fetch_historical_data(limit=3000)
    
    if data is None or len(data) < 500:
        print("❌ Not enough training data")
        return False
    
    print(f"✅ Loaded {len(data)} 5-minute candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Train models
    indicator_engine = SimpleMLIndicatorEngine()
    trainer = BidirectionalMLTrainer()
    
    models = trainer.train_bidirectional_models(data, indicator_engine)
    
    if models:
        print("\n🎉 Bidirectional models trained successfully!")
        print("💾 Models saved: bidirectional_rf_model.pkl, bidirectional_gb_model.pkl")
        return True
    else:
        print("\n❌ Bidirectional training failed")
        return False


if __name__ == "__main__":
    train_bidirectional_models_main()
