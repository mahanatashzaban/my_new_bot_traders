#!/usr/bin/env python3
"""
Simple ML Trainer - No XGBoost Dependency
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleMLTrainer:
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
    
    def create_binary_labels(self, df):
        """Simple binary classification labels"""
        print("Creating binary labels...")
        labels = []
        
        for i in range(len(df) - 1):
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]
            
            if next_price > current_price:
                labels.append(1)  # UP
            else:
                labels.append(0)  # DOWN
        
        labels.append(0)  # Last point
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Label distribution: UP={counts[1]}, DOWN={counts[0]}")
        return labels
    
    def train_simple_models(self, df, indicator_engine):
        """Train simple models without XGBoost"""
        print("🤖 TRAINING SIMPLE MODELS (No XGBoost)")
        print("=" * 50)
        
        # Prepare features
        X, df_clean = self.prepare_features(df, indicator_engine)
        if X is None:
            return None
        
        # Create labels
        y = self.create_binary_labels(df_clean)
        X = X[:len(y)]  # Align shapes
        
        print(f"📊 Training data: {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"🔧 Training set: {len(X_train)} samples")
        print(f"🔧 Testing set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Train Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['rf'] = rf_model
        
        # 2. Train Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['gb'] = gb_model
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, y_test)
        
        # Save models
        self.save_models()
        
        print("💾 Simple models saved successfully")
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 MODEL EVALUATION:")
        print("=" * 50)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{name.upper()} Model:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
    
    def save_models(self):
        """Save models and scaler"""
        for name, model in self.models.items():
            joblib.dump(model, f'simple_{name}_model.pkl')
        joblib.dump(self.scaler, 'simple_scaler.pkl')
        print("💾 Models saved as: simple_rf_model.pkl, simple_gb_model.pkl, simple_scaler.pkl")
    
    def load_ml_models(self):
        """Load trained models"""
        try:
            self.models['rf'] = joblib.load('simple_rf_model.pkl')
            self.models['gb'] = joblib.load('simple_gb_model.pkl')
            self.scaler = joblib.load('simple_scaler.pkl')
            print("✅ Simple ML models loaded successfully")
            return True
        except FileNotFoundError as e:
            print(f"❌ No trained models found: {e}")
            return False
    
    def predict_direction(self, df, indicator_engine):
        """Predict market direction"""
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
                
                predictions[name] = 'UP' if pred == 1 else 'DOWN'
                confidences[name] = confidence
            except Exception as e:
                print(f"❌ Prediction error with {name}: {e}")
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

if __name__ == "__main__":
    print("🧪 Testing Simple ML Trainer")
    from data_manager_simple import DataManager
    from simple_indicators import SimpleMLIndicatorEngine
    
    dm = DataManager()
    data = dm.fetch_historical_data(limit=500)
    
    if data is not None:
        indicator_engine = SimpleMLIndicatorEngine()
        trainer = SimpleMLTrainer()
        models = trainer.train_simple_models(data, indicator_engine)
        
        if models:
            print("🎉 Simple training successful!")
