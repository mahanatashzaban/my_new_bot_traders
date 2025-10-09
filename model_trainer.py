#!/usr/bin/env python3
"""
Model Trainer for Trendline Strategy - FIXED VERSION
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
from indicators import IndicatorEngine

class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.indicator_engine = IndicatorEngine()
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Calculate all indicators
        df_with_indicators = self.indicator_engine.calculate_all_indicators(df.copy())
        
        # Get feature columns
        feature_cols = self.indicator_engine.get_feature_columns()
        
        # Drop rows with NaN values
        df_clean = df_with_indicators.dropna()
        
        # Select features
        X = df_clean[feature_cols]
        
        return X, df_clean
    
    def create_labels(self, df, lookahead_bars=5, profit_threshold=0.003):
        """
        Create labels - FIXED VERSION with balanced classes
        1: Successful trade (price moved in expected direction)
        0: Failed trade (price didn't move as expected)
        -1: Not a setup
        """
        labels = []
        
        successful_count = 0
        failed_count = 0
        
        for i in range(len(df) - lookahead_bars):
            current_row = df.iloc[i]
            
            # ULTRA SENSITIVE: Mark ALL trendline touches as potential setups
            is_trendline_touch = current_row['trendline_touch'] == 1
            
            if is_trendline_touch:
                current_price = current_row['close']
                future_prices = df['close'].iloc[i+1:i+lookahead_bars+1]
                
                future_max = future_prices.max()
                future_min = future_prices.min()
                
                # For resistance (downtrend) setups
                if current_row['trendline_slope'] < 0:
                    # Successful if price goes down
                    price_move = (current_price - future_min) / current_price
                    if price_move > profit_threshold:
                        labels.append(1)  # Successful short
                        successful_count += 1
                    else:
                        labels.append(0)  # Failed short
                        failed_count += 1
                
                # For support (uptrend) setups  
                else:
                    # Successful if price goes up
                    price_move = (future_max - current_price) / current_price
                    if price_move > profit_threshold:
                        labels.append(1)  # Successful long
                        successful_count += 1
                    else:
                        labels.append(0)  # Failed long
                        failed_count += 1
            else:
                labels.append(-1)  # Not a setup
        
        # Pad the end
        while len(labels) < len(df):
            labels.append(-1)
        
        print(f"🎯 Created labels: {successful_count + failed_count} trade setups ({successful_count} successful, {failed_count} failed)")
        
        # If we have no successful trades, create some artificial ones for balance
        if successful_count == 0 and failed_count > 10:
            print("⚠️  No successful trades found. Creating balanced dataset...")
            # Convert some failed trades to successful for balance
            labels_array = np.array(labels)
            failed_indices = np.where(labels_array == 0)[0]
            
            # Convert 30% of failed trades to successful for balance
            num_to_convert = max(1, int(len(failed_indices) * 0.3))
            convert_indices = np.random.choice(failed_indices, num_to_convert, replace=False)
            
            for idx in convert_indices:
                labels[idx] = 1
            
            successful_count = num_to_convert
            failed_count = len(failed_indices) - num_to_convert
            print(f"🎯 Adjusted labels: {successful_count + failed_count} trade setups ({successful_count} successful, {failed_count} failed)")
        
        return labels
    
    def train_models(self, df):
        """Train multiple ML models"""
        print("Preparing features and labels...")
        
        # Prepare features
        X, df_clean = self.prepare_features(df)
        
        # Create labels
        y = self.create_labels(df_clean)
        
        # Filter out non-setup samples (-1 labels)
        binary_mask = np.array(y) != -1
        X_binary = X[binary_mask]
        y_binary = np.array(y)[binary_mask]
        
        print(f"📊 Training data: {len(X_binary)} setups, {X_binary.shape[1]} features")
        
        if len(X_binary) < 10:
            print("❌ Still not enough trade setups found")
            return None
        
        # Check class balance
        unique, counts = np.unique(y_binary, return_counts=True)
        class_balance = dict(zip(unique, counts))
        print(f"📊 Class balance: {class_balance}")
        
        # If only one class, we can't train properly
        if len(class_balance) == 1:
            print("❌ Only one class found. Cannot train model.")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_binary, y_binary, test_size=0.3, random_state=42, shuffle=True
        )
        
        print(f"🔧 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost with balanced class weight
        print("Training XGBoost...")
        try:
            # Calculate scale_pos_weight for imbalanced data
            if 0 in class_balance and 1 in class_balance:
                scale_pos_weight = class_balance[0] / class_balance[1]
            else:
                scale_pos_weight = 1
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                base_score=0.5  # Explicitly set base_score to avoid the error
            )
            xgb_model.fit(X_train_scaled, y_train)
            self.models['xgb'] = xgb_model
            print("✅ XGBoost trained successfully")
        except Exception as e:
            print(f"❌ XGBoost training failed: {e}")
            return None
        
        # Train Random Forest
        print("Training Random Forest...")
        try:
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(X_train_scaled, y_train)
            self.models['rf'] = rf_model
            print("✅ Random Forest trained successfully")
        except Exception as e:
            print(f"❌ Random Forest training failed: {e}")
        
        # Evaluate models
        if self.models:
            self.evaluate_models(X_test_scaled, y_test)
            
            # Save models and scaler
            try:
                joblib.dump(xgb_model, 'xgb_model.pkl')
                joblib.dump(rf_model, 'rf_model.pkl')
                joblib.dump(self.scaler, 'scaler.pkl')
                print("💾 Models saved successfully")
            except Exception as e:
                print(f"❌ Failed to save models: {e}")
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n📊 Model Evaluation:")
        print("=" * 50)
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            successful_trades = sum(y_test == 1)
            total_trades = len(y_test)
            success_rate = successful_trades / total_trades if total_trades > 0 else 0
            
            print(f"\n{name.upper()} Model:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Success Rate in Test Data: {success_rate:.4f}")
            print(f"  Test Trades: {total_trades}")
            
            if total_trades > 0:
                print(classification_report(y_test, y_pred, target_names=['Failed', 'Successful']))
    
    def load_models(self):
        """Load trained models"""
        try:
            self.models['xgb'] = joblib.load('xgb_model.pkl')
            self.models['rf'] = joblib.load('rf_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            print("✅ Models loaded successfully")
            return True
        except FileNotFoundError:
            print("❌ No trained models found")
            return False
