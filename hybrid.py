#!/usr/bin/env python3
"""
ADVANCED HYBRID TRADING BOT - FIXED & OPTIMIZED
ML + Technical Analysis + Adaptive Risk Management
"""

import pandas as pd
import numpy as np
import time
import requests
import datetime
import warnings
from typing import Dict, Tuple, Optional
warnings.filterwarnings('ignore')

class AdvancedHybridBot:
    def __init__(self):
        self.symbol = "BTCUSDT"
        self.initial_balance = 100.0
        self.balance = self.initial_balance
        self.position = None

        # OPTIMIZED Risk Parameters
        self.base_leverage = 3
        self.base_position_size = 0.15
        self.tp_percent = 0.015  # 1.5% Take Profit
        self.sl_percent = 0.010  # 1.0% Stop Loss

        # Trading Statistics
        self.equity_curve = [self.initial_balance]
        self.trade_log = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0.0

        # Market State
        self.market_regime = "NEUTRAL"
        self.volatility_regime = "NORMAL"
        self.trend_strength = 0.0

        # Signal Tracking
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.daily_trades = 0
        self.last_reset_date = datetime.datetime.now().date()

        print("🚀 ADVANCED HYBRID BOT INITIALIZED")
        print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        print("🎯 ML + Technical Analysis + Adaptive Risk Management")

    def reset_daily_counters(self):
        """Reset daily trade counters"""
        today = datetime.datetime.now().date()
        if self.last_reset_date != today:
            self.daily_trades = 0
            self.last_reset_date = today
            print("🔄 Daily counters reset")

    def detect_market_regime(self, df: pd.DataFrame) -> Tuple[str, str, float]:
        """Advanced market regime detection"""
        if len(df) < 50:
            return "NEUTRAL", "NORMAL", 0.0

        prices = df['close'].tail(50)
        
        # Volatility (ATR)
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        atr_percent = (atr / prices.iloc[-1]) * 100

        # Trend detection with multiple timeframes
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        ema_12 = prices.ewm(span=12).mean().iloc[-1]
        ema_26 = prices.ewm(span=26).mean().iloc[-1]
        
        current_price = prices.iloc[-1]
        
        # Trend strength calculation
        price_vs_sma20 = abs(current_price - sma_20) / sma_20 * 100
        sma20_vs_sma50 = abs(sma_20 - sma_50) / sma_50 * 100
        ema_diff = abs(ema_12 - ema_26) / ema_26 * 100
        
        self.trend_strength = (price_vs_sma20 + sma20_vs_sma50 + ema_diff) / 3

        # Volatility regime
        if atr_percent > 3.0:
            volatility = "HIGH"
        elif atr_percent < 0.8:
            volatility = "LOW"
        else:
            volatility = "NORMAL"

        # Market regime
        if self.trend_strength > 2.0 and price_vs_sma20 > 1.5:
            regime = "TRENDING"
        elif self.trend_strength < 0.8:
            regime = "RANGING"
        else:
            regime = "NEUTRAL"

        return regime, volatility, self.trend_strength

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive technical indicator suite"""
        df = df.copy()
        
        # Price-based indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI with different periods
        for period in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Support/Resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['price_vs_sr'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        # Volatility
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['volatility'] = (df['atr'] / df['close']) * 100
        
        return df

    def generate_advanced_signal(self, df: pd.DataFrame, regime: str, volatility: str) -> Dict:
        """Advanced signal generation with multiple confirmation"""
        if len(df) < 30:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        confidences = []
        reasons = []
        
        # 1. TREND FOLLOWING SIGNALS
        if regime == "TRENDING":
            # Strong trend following
            if current['sma_20'] > current['sma_50'] and current['macd_histogram'] > 0:
                signals.append('BUY')
                confidences.append(0.75)
                reasons.append("Strong uptrend + MACD bullish")
            elif current['sma_20'] < current['sma_50'] and current['macd_histogram'] < 0:
                signals.append('SELL')
                confidences.append(0.75)
                reasons.append("Strong downtrend + MACD bearish")
        
        # 2. MEAN REVERSION (Bollinger Bands)
        if current['bb_position'] < 0.1 and current['rsi_14'] < 35:
            signals.append('BUY')
            confidences.append(0.70)
            reasons.append("Oversold - BB lower + RSI low")
        elif current['bb_position'] > 0.9 and current['rsi_14'] > 65:
            signals.append('SELL')
            confidences.append(0.70)
            reasons.append("Overbought - BB upper + RSI high")
        
        # 3. BREAKOUT SIGNALS
        if current['close'] > current['resistance'] and current['volume_ratio'] > 1.5:
            signals.append('BUY')
            confidences.append(0.80)
            reasons.append("Resistance breakout with high volume")
        elif current['close'] < current['support'] and current['volume_ratio'] > 1.5:
            signals.append('SELL')
            confidences.append(0.80)
            reasons.append("Support breakdown with high volume")
        
        # 4. MOMENTUM CONFIRMATION
        if current['macd'] > current['macd_signal'] and current['macd_histogram'] > prev['macd_histogram']:
            if 'BUY' not in signals:
                signals.append('BUY')
                confidences.append(0.65)
                reasons.append("Momentum turning bullish")
        elif current['macd'] < current['macd_signal'] and current['macd_histogram'] < prev['macd_histogram']:
            if 'SELL' not in signals:
                signals.append('SELL')
                confidences.append(0.65)
                reasons.append("Momentum turning bearish")
        
        # Combine and filter signals
        if not signals:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'No clear signals'}
        
        # Take strongest signal
        best_idx = np.argmax(confidences)
        best_signal = signals[best_idx]
        best_confidence = confidences[best_idx]
        
        # Adjust confidence based on market conditions
        if volatility == "HIGH":
            best_confidence *= 0.9  # Reduce confidence in high volatility
        elif volatility == "LOW":
            best_confidence *= 1.1  # Increase confidence in low volatility
            
        # Minimum confidence threshold
        min_confidence = 0.65
        
        if best_confidence >= min_confidence:
            return {
                'signal': best_signal,
                'confidence': min(best_confidence, 0.95),
                'reason': ' | '.join(reasons),
                'regime': regime,
                'volatility': volatility
            }
        else:
            return {'signal': 'HOLD', 'confidence': best_confidence, 'reason': 'Below confidence threshold'}

    def adaptive_risk_management(self, signal: Dict) -> Tuple[float, float]:
        """Dynamic risk adjustment based on multiple factors"""
        base_size = self.base_position_size
        base_leverage = self.base_leverage
        
        # Adjust based on confidence
        confidence_boost = (signal['confidence'] - 0.65) / 0.3  # 0 to 1 scale
        size_multiplier = 0.8 + (confidence_boost * 0.4)  # 0.8x to 1.2x
        
        # Adjust based on market regime
        if signal['regime'] == "TRENDING":
            size_multiplier *= 1.2
        elif signal['volatility'] == "HIGH":
            size_multiplier *= 0.7
        
        # Adjust based on recent performance
        if self.consecutive_losses >= 2:
            size_multiplier *= 0.6  # Reduce size after consecutive losses
        
        # Position size limits
        final_size = min(base_size * size_multiplier, 0.25)  # Max 25%
        final_leverage = base_leverage
        
        return final_size, final_leverage

    def should_enter_trade(self, signal: Dict, current_price: float) -> Tuple[bool, str]:
        """Comprehensive trade validation"""
        if signal['signal'] == 'HOLD':
            return False, "No valid signal"
            
        if self.position:
            return False, "Already in position"
            
        # Minimum confidence check
        if signal['confidence'] < 0.65:
            return False, f"Low confidence: {signal['confidence']:.2f}"
            
        # Daily trade limit
        if self.daily_trades >= 20:
            return False, "Daily trade limit reached"
            
        # Cooldown after previous trade
        if self.last_trade_time:
            time_since_last = (datetime.datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < 120:  # 2 minute cooldown
                return False, f"Cooldown: {120 - int(time_since_last)}s remaining"
                
        # Balance protection
        if self.balance < self.initial_balance * 0.8:
            return False, "Balance below 80% of initial"
            
        return True, "All checks passed"

    def execute_trade(self, signal: Dict, current_price: float) -> bool:
        """Execute trade with proper risk management"""
        validation, reason = self.should_enter_trade(signal, current_price)
        if not validation:
            print(f"⏸️  Trade skipped: {reason}")
            return False
            
        # Get adaptive position size and leverage
        position_size, leverage = self.adaptive_risk_management(signal)
        
        # Calculate trade size
        trade_amount = self.balance * position_size * leverage / current_price
        
        self.position = {
            'type': signal['signal'],
            'entry_price': current_price,
            'size': trade_amount,
            'leverage': leverage,
            'position_size': position_size,
            'tp_price': current_price * (1 + self.tp_percent / leverage) if signal['signal'] == 'BUY' else current_price * (1 - self.tp_percent / leverage),
            'sl_price': current_price * (1 - self.sl_percent / leverage) if signal['signal'] == 'BUY' else current_price * (1 + self.sl_percent / leverage),
            'entry_time': datetime.datetime.now(),
            'confidence': signal['confidence'],
            'regime': signal['regime']
        }
        
        print(f"🎯 OPEN {signal['signal']} at ${current_price:.2f}")
        print(f"   Size: {trade_amount:.6f} BTC | Leverage: {leverage}x")
        print(f"   TP: ${self.position['tp_price']:.2f} | SL: ${self.position['sl_price']:.2f}")
        print(f"   Confidence: {signal['confidence']:.2f} | Position Size: {position_size*100:.1f}%")
        
        self.last_trade_time = datetime.datetime.now()
        self.daily_trades += 1
        
        return True

    def check_exit_conditions(self, current_price: float) -> bool:
        """Check TP/SL and other exit conditions"""
        if not self.position:
            return False
            
        entry = self.position['entry_price']
        position_type = self.position['type']
        
        # Fixed TP/SL based on price levels (FIXED VERSION)
        if position_type == 'BUY':
            # Long position
            if current_price >= self.position['tp_price']:
                self.close_position(current_price, "TP")
                return True
            elif current_price <= self.position['sl_price']:
                self.close_position(current_price, "SL")
                return True
        else:
            # Short position  
            if current_price <= self.position['tp_price']:
                self.close_position(current_price, "TP")
                return True
            elif current_price >= self.position['sl_price']:
                self.close_position(current_price, "SL")
                return True
                
        return False

    def close_position(self, current_price: float, reason: str):
        """Close position and update statistics"""
        if not self.position:
            return
            
        entry = self.position['entry_price']
        position_type = self.position['type']
        leverage = self.position['leverage']
        position_size = self.position['position_size']
        
        # Calculate PnL
        if position_type == 'BUY':
            pnl_pct = (current_price - entry) / entry * leverage
        else:
            pnl_pct = (entry - current_price) / entry * leverage
            
        pnl_amount = self.balance * position_size * pnl_pct
        
        # Update balance and statistics
        self.balance += pnl_amount
        self.total_pnl += pnl_amount
        self.total_trades += 1
        
        # Update win/loss counters
        if pnl_amount > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            
        # Log trade
        self.trade_log.append({
            'action': reason,
            'type': position_type,
            'entry': entry,
            'exit': current_price,
            'pnl': pnl_amount,
            'pnl_pct': pnl_pct,
            'timestamp': datetime.datetime.now(),
            'confidence': self.position['confidence']
        })
        
        print(f"🎯 {reason} {position_type}: {pnl_pct*100:+.2f}% (${pnl_amount:+.2f})")
        self.position = None

    def update_equity_curve(self, current_price: float):
        """Update equity curve with unrealized PnL"""
        equity = self.balance
        
        if self.position:
            entry = self.position['entry_price']
            position_type = self.position['type']
            leverage = self.position['leverage']
            position_size = self.position['position_size']
            
            if position_type == 'BUY':
                unrealized_pnl = (current_price - entry) / entry * leverage
            else:
                unrealized_pnl = (entry - current_price) / entry * leverage
                
            equity += self.balance * position_size * unrealized_pnl
            
        self.equity_curve.append(equity)
        
        # Update peak equity and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - equity) / self.peak_equity * 100
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def show_comprehensive_dashboard(self, current_price: float, signal: Dict, regime: str, volatility: str):
        """Detailed performance dashboard"""
        self.update_equity_curve(current_price)
        equity = self.equity_curve[-1]
        
        # Calculate metrics
        total_return = (equity - self.initial_balance) / self.initial_balance * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_win = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] > 0]) if self.winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trade_log if t['pnl'] < 0]) if self.losing_trades > 0 else 0
        profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if self.losing_trades > 0 else float('inf')
        
        print("\n" + "="*90)
        print("🚀 ADVANCED HYBRID TRADING BOT - REAL-TIME DASHBOARD")
        print("="*90)
        print(f"🕒 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Price: ${current_price:.2f} | Market: {regime} | Vol: {volatility} | Trend: {self.trend_strength:.2f}%")
        print("-"*90)
        
        print("💵 PERFORMANCE METRICS:")
        print(f"   Equity: ${equity:.2f} | Balance: ${self.balance:.2f} | Return: {total_return:+.2f}%")
        print(f"   Peak Equity: ${self.peak_equity:.2f} | Max Drawdown: {self.max_drawdown:.2f}%")
        print(f"   Total P&L: ${self.total_pnl:+.2f}")
        
        print("\n📊 TRADING STATISTICS:")
        print(f"   Trades: {self.total_trades} | Win Rate: {win_rate:.1f}%")
        print(f"   Wins: {self.winning_trades} | Losses: {self.losing_trades}")
        print(f"   Avg Win: ${avg_win:+.2f} | Avg Loss: ${avg_loss:+.2f}")
        print(f"   Profit Factor: {profit_factor:.2f} | Daily Trades: {self.daily_trades}/20")
        
        print("\n📈 CURRENT POSITION:")
        if self.position:
            entry = self.position['entry_price']
            position_type = self.position['type']
            leverage = self.position['leverage']
            
            if position_type == 'BUY':
                pnl_pct = (current_price - entry) / entry * leverage
            else:
                pnl_pct = (entry - current_price) / entry * leverage
                
            pnl_amount = self.balance * self.position['position_size'] * pnl_pct
            pnl_color = "🟢" if pnl_pct > 0 else "🔴"
            
            print(f"   {position_type} | Entry: ${entry:.2f} | Leverage: {leverage}x")
            print(f"   P&L: {pnl_color} {pnl_pct*100:+.2f}% (${pnl_amount:+.2f})")
            print(f"   To TP: {abs((self.position['tp_price'] - current_price) / current_price * 100):.2f}%")
            print(f"   To SL: {abs((self.position['sl_price'] - current_price) / current_price * 100):.2f}%")
        else:
            print("   No active position | Status: READY")
            
        print(f"\n🎯 TRADING SIGNAL:")
        print(f"   Signal: {signal['signal']} | Confidence: {signal['confidence']:.2f}")
        print(f"   Reason: {signal['reason']}")
        print("="*90)

    def fetch_market_data(self) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
        """Fetch quality market data with 5-minute candles"""
        try:
            # Use 5-minute candles for better signals (less noise)
            url = "https://api-pub.bitfinex.com/v2/candles/trade:5m:tBTCUSD/hist?limit=100&sort=-1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not data:
                return None, None
                
            df = pd.DataFrame(data, columns=["timestamp", "open", "close", "high", "low", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Convert to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
                
            # Get current price from ticker
            ticker_url = "https://api-pub.bitfinex.com/v2/ticker/tBTCUSD"
            ticker_response = requests.get(ticker_url, timeout=10)
            ticker_data = ticker_response.json()
            current_price = float(ticker_data[6])
            
            return df, current_price
            
        except Exception as e:
            print(f"❌ Data fetch error: {e}")
            return None, None

    def run_bot(self):
        """Main trading loop"""
        print("\n🤖 STARTING ADVANCED HYBRID BOT")
        print("✅ Fixed TP/SL | Better Signals | Advanced Risk Management")
        print("📊 Running with 5-minute data for cleaner signals")
        print("⏰ 60-second intervals | Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                self.reset_daily_counters()
                
                # Fetch market data
                df, current_price = self.fetch_market_data()
                if df is None or current_price is None:
                    print("❌ Failed to fetch market data")
                    time.sleep(60)
                    continue
                    
                # Detect market regime
                market_regime, volatility, trend_strength = self.detect_market_regime(df)
                
                # Calculate advanced indicators
                df_with_indicators = self.calculate_advanced_indicators(df)
                
                # Generate signal
                signal = self.generate_advanced_signal(df_with_indicators, market_regime, volatility)
                
                # Check exit conditions first
                if self.check_exit_conditions(current_price):
                    time.sleep(30)
                    continue
                    
                # Show comprehensive dashboard
                self.show_comprehensive_dashboard(current_price, signal, market_regime, volatility)
                
                # Execute new trade if conditions are met
                if not self.position and signal['signal'] != 'HOLD':
                    self.execute_trade(signal, current_price)
                    
                print(f"⏰ Next check in 60 seconds... (Iteration: {iteration})\n")
                time.sleep(60)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Bot stopped after {iteration} iterations")
            self.show_final_summary()

    def show_final_summary(self):
        """Display final performance summary"""
        if self.equity_curve:
            final_equity = self.equity_curve[-1]
            total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            print("\n🎯 FINAL PERFORMANCE SUMMARY")
            print("="*50)
            print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
            print(f"💰 Final Equity: ${final_equity:.2f}")
            print(f"📈 Total Return: {total_return:+.2f}%")
            print(f"📊 Total Trades: {self.total_trades}")
            print(f"🎯 Win Rate: {win_rate:.1f}%")
            print(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")
            print(f"💵 Total P&L: ${self.total_pnl:+.2f}")
            print(f"🔥 Winning Trades: {self.winning_trades}")
            print(f"💀 Losing Trades: {self.losing_trades}")
            print("="*50)

def main():
    bot = AdvancedHybridBot()
    bot.run_bot()

if __name__ == "__main__":
    main()
