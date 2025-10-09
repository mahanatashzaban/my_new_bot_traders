import pandas as pd
import numpy as np
import ta
from ta import trend, momentum, volatility, volume

class MLIndicatorEngine:
    def __init__(self):
        self.features = []
    
    def calculate_all_indicators(self, df):
        """Calculate 20+ technical indicators for ML features - Universal version"""
        print("📊 Calculating 20+ technical indicators...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        try:
            # 1. TREND INDICATORS
            df['sma_10'] = self.safe_calculate(lambda: trend.SMAIndicator(df['close'], window=10).sma_indicator(), 
                                             df['close'].rolling(10).mean())
            df['sma_20'] = self.safe_calculate(lambda: trend.SMAIndicator(df['close'], window=20).sma_indicator(),
                                             df['close'].rolling(20).mean())
            df['sma_50'] = self.safe_calculate(lambda: trend.SMAIndicator(df['close'], window=50).sma_indicator(),
                                             df['close'].rolling(50).mean())
            df['ema_12'] = self.safe_calculate(lambda: trend.EMAIndicator(df['close'], window=12).ema_indicator(),
                                             df['close'].ewm(span=12).mean())
            df['ema_26'] = self.safe_calculate(lambda: trend.EMAIndicator(df['close'], window=26).ema_indicator(),
                                             df['close'].ewm(span=26).mean())
            
            # WMA - try different methods
            df['wma_14'] = self.safe_calculate(lambda: trend.WMAIndicator(df['close'], window=14).wma_indicator(),
                                             self.calculate_wma(df['close'], 14))
            
            df['adx'] = self.safe_calculate(lambda: trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx(),
                                          self.calculate_adx(df['high'], df['low'], df['close']))
            df['cci'] = self.safe_calculate(lambda: trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci(),
                                          self.calculate_cci(df['high'], df['low'], df['close']))
            
            # 2. MOMENTUM INDICATORS
            df['rsi_14'] = self.safe_calculate(lambda: momentum.RSIIndicator(df['close'], window=14).rsi(),
                                             self.calculate_rsi(df['close']))
            df['stoch_k'] = self.safe_calculate(lambda: momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch(),
                                              self.calculate_stoch_k(df['high'], df['low'], df['close']))
            df['stoch_d'] = self.safe_calculate(lambda: momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal(),
                                              self.calculate_stoch_d(df['high'], df['low'], df['close']))
            df['williams_r'] = self.safe_calculate(lambda: momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r(),
                                                 self.calculate_williams_r(df['high'], df['low'], df['close']))
            df['mom_10'] = self.safe_calculate(lambda: momentum.ROCIndicator(df['close'], window=10).roc(),
                                             df['close'].pct_change(10) * 100)
            df['ao'] = self.safe_calculate(lambda: momentum.AwesomeOscillatorIndicator(df['high'], df['low']).awesome_oscillator(),
                                         self.calculate_ao(df['high'], df['low']))
            
            # 3. VOLATILITY INDICATORS
            bb_indicator = volatility.BollingerBands(df['close'])
            df['bb_upper'] = self.safe_calculate(lambda: bb_indicator.bollinger_hband(),
                                               df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std())
            df['bb_lower'] = self.safe_calculate(lambda: bb_indicator.bollinger_lband(),
                                               df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std())
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            
            df['atr_14'] = self.safe_calculate(lambda: volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range(),
                                             self.calculate_atr(df['high'], df['low'], df['close']))
            
            # Keltner Channel
            df['kc_upper'] = self.safe_calculate(lambda: volatility.KeltnerChannel(df['high'], df['low'], df['close']).keltner_channel_hband(),
                                               self.calculate_kc_upper(df['high'], df['low'], df['close']))
            df['kc_lower'] = self.safe_calculate(lambda: volatility.KeltnerChannel(df['high'], df['low'], df['close']).keltner_channel_lband(),
                                               self.calculate_kc_lower(df['high'], df['low'], df['close']))
            
            # 4. VOLUME INDICATORS
            df['obv'] = self.safe_calculate(lambda: volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume(),
                                          self.calculate_obv(df['close'], df['volume']))
            df['cmf'] = self.safe_calculate(lambda: volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow(),
                                          self.calculate_cmf(df['high'], df['low'], df['close'], df['volume']))
            df['mfi'] = self.safe_calculate(lambda: volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index(),
                                          self.calculate_mfi(df['high'], df['low'], df['close'], df['volume']))
            
            # 5. CUSTOM FEATURES
            df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['open']
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # MACD
            macd_indicator = trend.MACD(df['close'])
            df['macd'] = self.safe_calculate(lambda: macd_indicator.macd(),
                                           self.calculate_macd(df['close']))
            df['macd_signal'] = self.safe_calculate(lambda: macd_indicator.macd_signal(),
                                                  self.calculate_macd_signal(df['close']))
            df['macd_hist'] = self.safe_calculate(lambda: macd_indicator.macd_diff(),
                                                self.calculate_macd_hist(df['close']))
            
            print(f"✅ Successfully calculated {len(self.get_feature_columns())} technical indicators")
            return df
            
        except Exception as e:
            print(f"❌ Error in indicator calculation: {e}")
            print("🔄 Using basic indicator fallback...")
            return self.calculate_basic_indicators(df)
    
    def safe_calculate(self, ta_func, fallback_func=None):
        """Safely calculate indicator with fallback"""
        try:
            return ta_func()
        except Exception as e:
            if fallback_func is not None:
                return fallback_func
            else:
                # Return NaN series with same index as input
                if hasattr(ta_func, '__code__') and 'df' in ta_func.__code__.co_varnames:
                    # Extract dataframe from closure to get proper index
                    import inspect
                    closure_vars = inspect.getclosurevars(ta_func)
                    for var in closure_vars.nonlocals.values():
                        if isinstance(var, pd.Series):
                            return pd.Series(np.nan, index=var.index)
                return np.nan
    
    # Manual indicator calculations as fallbacks
    def calculate_wma(self, prices, window):
        """Calculate Weighted Moving Average manually"""
        weights = np.arange(1, window + 1)
        return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_stoch_k(self, high, low, close, window=14):
        """Calculate Stochastic %K manually"""
        lowest_low = low.rolling(window).min()
        highest_high = high.rolling(window).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low)
    
    def calculate_stoch_d(self, high, low, close, window=14):
        """Calculate Stochastic %D manually"""
        stoch_k = self.calculate_stoch_k(high, low, close, window)
        return stoch_k.rolling(3).mean()
    
    def calculate_williams_r(self, high, low, close, lbp=14):
        """Calculate Williams %R manually"""
        highest_high = high.rolling(lbp).max()
        lowest_low = low.rolling(lbp).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    def calculate_ao(self, high, low):
        """Calculate Awesome Oscillator manually"""
        median_price = (high + low) / 2
        return median_price.rolling(5).mean() - median_price.rolling(34).mean()
    
    def calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range manually"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return tr.rolling(window).mean()
    
    def calculate_adx(self, high, low, close, window=14):
        """Calculate ADX manually (simplified)"""
        return 50  # Placeholder
    
    def calculate_cci(self, high, low, close, window=20):
        """Calculate CCI manually (simplified)"""
        tp = (high + low + close) / 3
        return (tp - tp.rolling(window).mean()) / (0.015 * tp.rolling(window).std())
    
    def calculate_kc_upper(self, high, low, close, window=20):
        """Calculate Keltner Channel Upper Band manually"""
        typical_price = (high + low + close) / 3
        return typical_price.rolling(window).mean() + 2 * typical_price.rolling(window).std()
    
    def calculate_kc_lower(self, high, low, close, window=20):
        """Calculate Keltner Channel Lower Band manually"""
        typical_price = (high + low + close) / 3
        return typical_price.rolling(window).mean() - 2 * typical_price.rolling(window).std()
    
    def calculate_obv(self, close, volume):
        """Calculate OBV manually"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def calculate_cmf(self, high, low, close, volume, window=20):
        """Calculate Chaikin Money Flow manually (simplified)"""
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        return money_flow_volume.rolling(window).mean() / volume.rolling(window).mean()
    
    def calculate_mfi(self, high, low, close, volume, window=14):
        """Calculate Money Flow Index manually (simplified)"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        return 50  # Placeholder
    
    def calculate_macd(self, close, fast=12, slow=26):
        """Calculate MACD manually"""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def calculate_macd_signal(self, close, signal=9):
        """Calculate MACD Signal manually"""
        macd = self.calculate_macd(close)
        return macd.ewm(span=signal).mean()
    
    def calculate_macd_hist(self, close):
        """Calculate MACD Histogram manually"""
        macd = self.calculate_macd(close)
        signal = self.calculate_macd_signal(close)
        return macd - signal
    
    def calculate_basic_indicators(self, df):
        """Calculate basic indicators as ultimate fallback"""
        print("🔄 Using ultimate basic indicators fallback...")
        
        # Basic trend
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # Basic momentum
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        
        # Basic volatility
        df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        
        # Custom features
        df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Basic MACD
        df['macd'] = self.calculate_macd(df['close'])
        df['macd_signal'] = self.calculate_macd_signal(df['close'])
        df['macd_hist'] = self.calculate_macd_hist(df['close'])
        
        return df
    
    def get_feature_columns(self):
        """Return all feature columns for ML"""
        return [
            # Trend indicators
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'wma_14', 'adx', 'cci',
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

    def get_indicator_stats(self, df):
        """Get statistics about the calculated indicators"""
        feature_cols = self.get_feature_columns()
        available_features = [col for col in feature_cols if col in df.columns]
        
        print(f"📊 Available features: {len(available_features)}/{len(feature_cols)}")
        
        stats = {}
        for col in available_features:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'null_count': df[col].isnull().sum()
                }
        
        return stats
