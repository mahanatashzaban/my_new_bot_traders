import pandas as pd
import numpy as np

class SimpleMLIndicatorEngine:
    def __init__(self):
        self.features = []
    
    def calculate_all_indicators(self, df):
        """Calculate basic indicators that definitely work"""
        print("📊 Calculating basic technical indicators...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # 1. BASIC TREND INDICATORS
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['wma_14'] = self.calculate_wma(df['close'], 14)
        
        # 2. BASIC MOMENTUM INDICATORS
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['stoch_k'] = self.calculate_stoch_k(df['high'], df['low'], df['close'])
        df['stoch_d'] = self.calculate_stoch_d(df['high'], df['low'], df['close'])
        df['williams_r'] = self.calculate_williams_r(df['high'], df['low'], df['close'])
        df['mom_10'] = df['close'].pct_change(10) * 100
        df['ao'] = self.calculate_ao(df['high'], df['low'])
        
        # 3. BASIC VOLATILITY INDICATORS
        df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['atr_14'] = self.calculate_atr(df['high'], df['low'], df['close'])
        df['kc_upper'] = self.calculate_kc_upper(df['high'], df['low'], df['close'])
        df['kc_lower'] = self.calculate_kc_lower(df['high'], df['low'], df['close'])
        
        # 4. BASIC VOLUME INDICATORS
        df['obv'] = self.calculate_obv(df['close'], df['volume'])
        df['cmf'] = self.calculate_cmf(df['high'], df['low'], df['close'], df['volume'])
        df['mfi'] = self.calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # 5. CUSTOM FEATURES
        df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 6. MACD
        df['macd'] = self.calculate_macd(df['close'])
        df['macd_signal'] = self.calculate_macd_signal(df['close'])
        df['macd_hist'] = self.calculate_macd_hist(df['close'])
        
        print(f"✅ Successfully calculated {len(self.get_feature_columns())} technical indicators")
        return df
    
    # Manual calculations
    def calculate_wma(self, prices, window):
        """Calculate Weighted Moving Average"""
        weights = np.arange(1, window + 1)
        return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_stoch_k(self, high, low, close, window=14):
        """Calculate Stochastic %K"""
        lowest_low = low.rolling(window).min()
        highest_high = high.rolling(window).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low)
    
    def calculate_stoch_d(self, high, low, close, window=14):
        """Calculate Stochastic %D"""
        stoch_k = self.calculate_stoch_k(high, low, close, window)
        return stoch_k.rolling(3).mean()
    
    def calculate_williams_r(self, high, low, close, lbp=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(lbp).max()
        lowest_low = low.rolling(lbp).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    def calculate_ao(self, high, low):
        """Calculate Awesome Oscillator"""
        median_price = (high + low) / 2
        return median_price.rolling(5).mean() - median_price.rolling(34).mean()
    
    def calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return tr.rolling(window).mean()
    
    def calculate_kc_upper(self, high, low, close, window=20):
        """Calculate Keltner Channel Upper Band"""
        typical_price = (high + low + close) / 3
        return typical_price.rolling(window).mean() + 2 * typical_price.rolling(window).std()
    
    def calculate_kc_lower(self, high, low, close, window=20):
        """Calculate Keltner Channel Lower Band"""
        typical_price = (high + low + close) / 3
        return typical_price.rolling(window).mean() - 2 * typical_price.rolling(window).std()
    
    def calculate_obv(self, close, volume):
        """Calculate On Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def calculate_cmf(self, high, low, close, volume, window=20):
        """Calculate Chaikin Money Flow (simplified)"""
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, 0.0001)
        money_flow_volume = money_flow_multiplier * volume
        return money_flow_volume.rolling(window).mean() / volume.rolling(window).mean().replace(0, 0.0001)
    
    def calculate_mfi(self, high, low, close, volume, window=14):
        """Calculate Money Flow Index (simplified)"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window).sum()
        negative_flow = money_flow.where(typical_price <= typical_price.shift(), 0).rolling(window).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 0.0001)))
        return mfi.replace([np.inf, -np.inf], 50)  # Replace infinities with 50
    
    def calculate_macd(self, close, fast=12, slow=26):
        """Calculate MACD"""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def calculate_macd_signal(self, close, signal=9):
        """Calculate MACD Signal"""
        macd = self.calculate_macd(close)
        return macd.ewm(span=signal).mean()
    
    def calculate_macd_hist(self, close):
        """Calculate MACD Histogram"""
        macd = self.calculate_macd(close)
        signal = self.calculate_macd_signal(close)
        return macd - signal
    
    def get_feature_columns(self):
        """Return all feature columns for ML"""
        return [
            # Trend indicators
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'wma_14',
            # Momentum indicators
            'rsi_14', 'stoch_k', 'stoch_d', 'williams_r', 'mom_10', 'ao',
            # Volatility indicators
            'bb_upper', 'bb_lower', 'bb_width', 'atr_14', 'kc_upper', 'kc_lower',
            # Volume indicators
            'obv', 'cmf', 'mfi',
            # Custom features
            'price_vs_sma_20', 'high_low_range', 'body_size', 'volume_ratio',
            # MACD
            'macd', 'macd_signal', 'macd_hist'
        ]
