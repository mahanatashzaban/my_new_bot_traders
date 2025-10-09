#!/usr/bin/env python3
"""
Bidirectional ML Trainer - FIXED VERSION with better labeling
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

    def create_3way_labels(self, df, future_bars=3, threshold=0.0015):
        """
        Create 3-way labels: LONG, SHORT, HOLD
        Adjusted for 5-minute scalping with smaller threshold
        """
        print("Creating 3-way labels (LONG/SHORT/HOLD)...")
        labels = []

        for i in range(len(df) - future_bars):
            current_price = df['close'].iloc[i]
            future_max = df['close'].iloc[i+1:i+future_bars+1].max()
            future_min = df['close'].iloc[i+1:i+future_bars+1].min()

            upside = (future_max - current_price) / current_price
            downside = (current_price - future_min) / current_price

            # More aggressive labeling for scalping
            if upside > threshold and upside > downside * 1.2:  # Strong upside bias
                labels.append(2)  # LONG
            elif downside > threshold and downside > upside * 1.2:  # Strong downside bias
                labels.append(0)  # SHORT
            else:
                labels.append(1)  # HOLD

        # Pad the end
        labels.extend([1] * min(future_bars, len(df)))

        unique, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
        distribution = dict(zip([label_names[x] for x in unique], counts))
        print(f"Label distribution: {distribution}")
        
        total = sum(distribution.values())
        print(f"Active trades: {distribution.get('LONG', 0) + distribution.get('SHORT', 0)}/{total} "
              f"({(distribution.get('LONG', 0) + distribution.get('SHORT', 0)) / total * 100:.1f}%)")
        
        return labels

    def create_momentum_labels(self, df, lookahead_bars=2):
        """
        Alternative labeling based on momentum
        """
        print("Creating momentum-based labels...")
        labels = []
        
        for i in range(len(df) - lookahead_bars):
            current_close = df['close'].iloc[i]
            future_close = df['close'].iloc[i + lookahead_bars]
            
            # FIXED: Use current_close instead of current_price
            price_change = (future_close - current_close) / current_close
            
            # Classify based on momentum
            if price_change > 0.002:  # 0.2% up
                labels.append(2)  # LONG
            elif price_change < -0.002:  # 0.2% down
                labels.append(0)  # SHORT
            else:
                labels.append(1)  # HOLD
        
        # Pad the end
        labels.extend([1] * min(lookahead_bars, len(df)))
        
        unique, counts = np.unique(labels, return_counts=True)
        label_names = {0: 'SHORT', 1: 'HOLD', 2: 'LONG'}
        distribution = dict(zip([label_names[x] for x in unique], counts))
        print(f"Momentum label distribution: {distribution}")
        
        return labels

    def train_bidirectional_models(self, df, indicator_engine):
        """Train models for LONG/SHORT trading"""
        print("🤖 TRAINING BIDIRECTIONAL MODELS (LONG & SHORT)")
        print("=" * 50)

        # Prepare features
        X, df_clean = self.prepare_features(df, indicator_engine)
        if X is None:
            return None

        # Try different labeling strategies
        print("\n🔧 Testing labeling strategies...")
        
        # Strategy 1: Original
        y1 = self.create_3way_labels(df_clean)
        active1 = sum(1 for label in y1 if label != 1)
        
        # Strategy 2: Momentum-based
        y2 = self.create_momentum_labels(df_clean)
        active2 = sum(1 for label in y2 if label != 1)
        
        # Choose the strategy with more active samples
        if active2 > active1:
            print(f"🎯 Using momentum labeling (more active samples: {active2} vs {active1})")
            y = y2
        else:
            print(f"🎯 Using threshold labeling (active samples: {active1})")
            y = y1
        
        X = X[:len(y)]  # Align shapes

        # Filter out HOLD labels for active trading
        active_mask = np.array(y) != 1
        X_active = X[active_mask]
        y_active = np.array(y)[active_mask]

        print(f"📊 Final training data: {len(X_active)} active trades")
        print(f"   LONG: {sum(y_active == 2)}, SHORT: {sum(y_active == 0)}")

        if len(X_active) < 50:
            print("⚠️  Low number of active trades, consider gathering more data")
            if len(X_active) < 20:
                print("❌ Not enough active trading samples")
                return None

        # Use all data but weight the active trades more
        X_all = X
        y_all = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        print(f"🔧 Training set: {len(X_train)} samples")
        print(f"🔧 Testing set: {len(X_test)} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest with class weights
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
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
        if long_votes > short_votes and long_votes >= hold_votes:
            final_prediction = 'LONG'
        elif short_votes > long_votes and short_votes >= hold_votes:
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
    
    print("🚀 TRAINING BIDIRECTIONAL MODELS - FIXED VERSION")
    print("=" * 50)
    
    dm = DataManager()
    
    # Fetch more data for better training
    print("Fetching 5-minute training data...")
    data = dm.fetch_historical_data(limit=5000)  # More data
    
    if data is None or len(data) < 1000:
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
