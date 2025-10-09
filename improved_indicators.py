#!/usr/bin/env python3
"""
Improved Indicator Engine with Better Trendline Detection
"""

import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume

class ImprovedIndicatorEngine:
    def __init__(self):
        self.features = []
    
    def calculate_trendline_features(self, df):
        """IMPROVED trendline detection - Much more selective"""
        # Initialize with zeros
        df['distance_to_trendline'] = 1.0
        df['trendline_touch'] = 0
        df['trendline_slope'] = 0.0
        df['rejection_strength'] = 0.0
        
        print("🔍 Finding QUALITY trendline setups...")
        quality_setups = 0
        
        for i in range(100, len(df)):  # Start from 100 for better context
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]
            current_open = df['open'].iloc[i]
            
            # Look for STRONG support/resistance levels
            # Use swing highs/lows instead of simple rolling max/min
            lookback = 50
            if i < lookback:
                continue
            
            # Find significant swing points
            recent_highs = []
            recent_lows = []
            
            # Look for local maxima and minima
            for j in range(i-30, i):
                if j < 10 or j > len(df) - 10:
                    continue
                    
                # Check for swing high
                is_swing_high = True
                for k in range(1, 6):
                    if df['high'].iloc[j] <= df['high'].iloc[j-k] or df['high'].iloc[j] <= df['high'].iloc[j+k]:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    recent_highs.append(df['high'].iloc[j])
                
                # Check for swing low
                is_swing_low = True
                for k in range(1, 6):
                    if df['low'].iloc[j] >= df['low'].iloc[j-k] or df['low'].iloc[j] >= df['low'].iloc[j+k]:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    recent_lows.append(df['low'].iloc[j])
            
            if not recent_highs or not recent_lows:
                continue
            
            # Find closest significant levels
            closest_resistance = min(recent_highs, key=lambda x: abs(current_high - x))
            closest_support = min(recent_lows, key=lambda x: abs(current_low - x))
            
            dist_to_resistance = abs(current_high - closest_resistance) / closest_resistance
            dist_to_support = abs(current_low - closest_support) / closest_support
            
            # Use the smaller distance
            min_distance = min(dist_to_resistance, dist_to_support)
            
            # Determine direction
            if dist_to_resistance < dist_to_support:
                trend_direction = -1  # Near resistance
                touch_price = closest_resistance
            else:
                trend_direction = 1   # Near support
                touch_price = closest_support
            
            df.loc[df.index[i], 'distance_to_trendline'] = min_distance
            df.loc[df.index[i], 'trendline_slope'] = trend_direction
            
            # MUCH STRICTER: Only mark as touch if very close (0.1%) AND has volume
            volume_spike = df['volume'].iloc[i] > df['volume'].iloc[i-20:i].mean() * 1.2
            
            if min_distance < 0.001 and volume_spike:  # 0.1% distance + volume confirmation
                df.loc[df.index[i], 'trendline_touch'] = 1
                
                # Calculate rejection strength more accurately
                body_size = abs(current_close - current_open) / (current_open + 0.0001)
                upper_wick = (current_high - max(current_open, current_close)) / (current_high + 0.0001)
                lower_wick = (min(current_open, current_close) - current_low) / (current_low + 0.0001)
                
                if trend_direction < 0:  # Resistance
                    rejection_strength = upper_wick / (body_size + 0.001)
                else:  # Support
                    rejection_strength = lower_wick / (body_size + 0.001)
                
                df.loc[df.index[i], 'rejection_strength'] = rejection_strength
                
                # Only count as quality setup if rejection is strong
                if rejection_strength > 1.5:  # Higher threshold
                    quality_setups += 1
        
        print(f"✅ Found {quality_setups} QUALITY trendline setups (much more selective)")
        return df
    
    def calculate_all_indicators(self, df):
        """Calculate all indicators"""
        print("📊 Calculating technical indicators...")
        
        # Your improved strategy features
        df = self.calculate_trendline_features(df)
        
        # Trend indicators
        df['sma_20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['ema_12'] = trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = trend.EMAIndicator(df['close'], window=26).ema_indicator()
        df['macd'] = trend.MACD(df['close']).macd()
        df['macd_signal'] = trend.MACD(df['close']).macd_signal()
        df['adx'] = trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        # Momentum indicators
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['stoch_k'] = momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['stoch_d'] = momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
        df['williams_r'] = momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        df['cci_20'] = trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        # Volatility indicators
        df['bb_upper'] = volatility.BollingerBands(df['close']).bollinger_hband()
        df['bb_lower'] = volatility.BollingerBands(df['close']).bollinger_lband()
        df['atr_14'] = volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Volume indicators
        df['mfi_14'] = volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        df['obv'] = volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Custom features
        df['price_vs_high_20'] = (df['close'] - df['high'].rolling(20).max()) / df['high'].rolling(20).max()
        df['price_vs_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['low'].rolling(20).min()
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['high']
        df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['low']
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        print("✅ All indicators calculated successfully")
        return df
    
    def get_feature_columns(self):
        """Return all feature columns"""
        return [
            # Your strategy features
            'distance_to_trendline', 'trendline_touch', 'trendline_slope', 'rejection_strength',
            
            # Trend indicators
            'sma_20', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'adx',
            
            # Momentum indicators
            'rsi_14', 'stoch_k', 'stoch_d', 'williams_r', 'cci_20',
            
            # Volatility indicators
            'bb_upper', 'bb_lower', 'atr_14',
            
            # Volume indicators
            'mfi_14', 'obv',
            
            # Custom features
            'price_vs_high_20', 'price_vs_low_20', 'body_size', 
            'upper_wick_ratio', 'lower_wick_ratio', 'volume_ma_ratio'
        ]
