#!/usr/bin/env python3
"""
AGGRESSIVE Bitcoin Scalping Bot - Lower Thresholds for More Trading
"""

import pandas as pd
import numpy as np
import time
import requests
import datetime
import hmac
import hashlib
import json
import os
from typing import Dict, List, Optional

class AggressiveBitcoinScalper:
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.symbol = "tBTCUST"
        self.timeframe = "1m"
        self.base_url = "https://api.bitfinex.com"
        
        # API credentials
        self.api_key = api_key or os.getenv('BITFINEX_API_KEY')
        self.api_secret = api_secret or os.getenv('BITFINEX_API_SECRET')
        
        # Trading parameters - MORE AGGRESSIVE
        self.initial_balance = 1000.0  # USDT
        self.balance = self.initial_balance
        self.position = None
        self.leverage = 8
        self.position_size = 0.15  # 15% of balance
        
        # Risk management - TIGHTER for scalping
        self.take_profit_pct = 0.006  # 0.6%
        self.stop_loss_pct = 0.003    # 0.3%
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_balance
        
        # Performance tracking
        self.trade_log = []
        self.equity_curve = []
        self.total_trades = 0
        self.winning_trades = 0
        
        # Technical indicators - OPTIMIZED for 1m scalping
        self.rsi_period = 10  # Shorter for scalping
        self.ema_fast = 5     # Very fast EMA
        self.ema_slow = 12    # Fast slow EMA
        self.bb_period = 15   # Shorter Bollinger Bands
        self.bb_std = 1.5     # Tighter bands
        
        print("🚀 AGGRESSIVE Bitcoin Scalping Bot Initialized")
        print(f"⚡ Timeframe: {self.timeframe}")
        print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        print(f"🎯 TP/SL: {self.take_profit_pct*100}%/{self.stop_loss_pct*100}%")
        print(f"🔥 Lower thresholds for more trading activity")

    def fetch_ohlcv(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Bitfinex"""
        try:
            url = f"https://api-pub.bitfinex.com/v2/candles/trade:1m:tBTCUSD/hist?limit={limit}&sort=-1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data, columns=["timestamp", "open", "close", "high", "low", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Convert to float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
                
            return df
        except Exception as e:
            print(f"❌ Error fetching OHLCV: {e}")
            return None

    def get_current_price(self) -> Optional[float]:
        """Get current Bitcoin price"""
        try:
            url = "https://api-pub.bitfinex.com/v2/ticker/tBTCUSD"
            response = requests.get(url, timeout=10)
            data = response.json()
            return float(data[6])  # Last price
        except Exception as e:
            print(f"❌ Error getting current price: {e}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate optimized indicators for 1m scalping"""
        df = df.copy()
        
        # RSI with shorter period
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Faster EMAs for scalping
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow).mean()
        df['ema_signal'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        
        # Tighter Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Price momentum for scalping
        df['momentum_3'] = df['close'].pct_change(3)  # 3-period momentum
        df['momentum_5'] = df['close'].pct_change(5)  # 5-period momentum
        
        # Volume spike detection
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price acceleration (rate of change of momentum)
        df['acceleration'] = df['momentum_3'].diff()
        
        return df

    def generate_trading_signal(self, df: pd.DataFrame) -> Dict:
        """Generate AGGRESSIVE trading signals with lower thresholds"""
        if len(df) < 20:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # AGGRESSIVE scoring - LOWER thresholds
        long_score = 0
        short_score = 0
        reasons = []
        
        # RSI signals - MORE SENSITIVE (25-75 range)
        if current['rsi'] < 35:  # Was 30
            long_score += 2
            reasons.append("RSI near oversold")
        elif current['rsi'] < 45:  # Was 40
            long_score += 1
        elif current['rsi'] > 65:  # Was 70
            short_score += 2
            reasons.append("RSI near overbought")
        elif current['rsi'] > 55:  # Was 60
            short_score += 1
        
        # EMA signals - FASTER CROSSOVERS
        if current['ema_signal'] == 1:
            long_score += 2
            reasons.append("EMA bullish")
        elif current['ema_signal'] == -1:
            short_score += 2
            reasons.append("EMA bearish")
        
        # Bollinger Bands - MORE SENSITIVE to band touches
        if current['bb_position'] < 0.3:  # Was 0.2
            long_score += 2
            reasons.append("Near BB lower")
        elif current['bb_position'] < 0.4:  # Was 0.3
            long_score += 1
        elif current['bb_position'] > 0.7:  # Was 0.8
            short_score += 2
            reasons.append("Near BB upper")
        elif current['bb_position'] > 0.6:  # Was 0.7
            short_score += 1
        
        # Momentum signals - MORE SENSITIVE
        if current['momentum_3'] > 0.0005:  # Smaller threshold
            long_score += 1
            reasons.append("Positive momentum")
        elif current['momentum_3'] < -0.0005:
            short_score += 1
            reasons.append("Negative momentum")
        
        # Volume spikes - ANY spike is good
        if current['volume_ratio'] > 1.5:
            if long_score > short_score:
                long_score += 2
                reasons.append("High volume bullish")
            elif short_score > long_score:
                short_score += 2
                reasons.append("High volume bearish")
        elif current['volume_ratio'] > 1.2:
            if long_score > short_score:
                long_score += 1
            elif short_score > long_score:
                short_score += 1
        
        # Acceleration (new for scalping)
        if current['acceleration'] > 0.0002:
            long_score += 1
            reasons.append("Accelerating up")
        elif current['acceleration'] < -0.0002:
            short_score += 1
            reasons.append("Accelerating down")
        
        print(f"🔍 Signal Analysis: LONG {long_score} | SHORT {short_score}")
        
        # AGGRESSIVE ENTRY - LOWER minimum score
        min_score_required = 4  # Was 6
        
        if long_score >= min_score_required:
            confidence = min(long_score / 8.0, 1.0)  # Adjusted max points
            return {'signal': 'BUY', 'confidence': confidence, 'reason': ', '.join(reasons)}
        elif short_score >= min_score_required:
            confidence = min(short_score / 8.0, 1.0)
            return {'signal': 'SELL', 'confidence': confidence, 'reason': ', '.join(reasons)}
        else:
            return {'signal': 'HOLD', 'confidence': max(long_score, short_score) / 8.0, 'reason': f'Score too low ({max(long_score, short_score)}/{min_score_required})'}

    def execute_order(self, side: str, amount: float, price: float) -> bool:
        """Execute trade order"""
        # For now, simulated execution
        print(f"🎯 EXECUTE {side}: {amount:.6f} BTC @ ${price:,.2f}")
        return True

    def calculate_position_size(self, price: float) -> float:
        """Calculate position size"""
        leveraged_amount = self.balance * self.position_size * self.leverage
        btc_amount = leveraged_amount / price
        return min(btc_amount, 0.02)  # Max 0.02 BTC

    def check_take_profit_stop_loss(self, current_price: float) -> bool:
        """Check if TP or SL is hit"""
        if not self.position:
            return False
        
        entry_price = self.position['entry_price']
        position_type = self.position['type']
        
        if position_type == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price * self.leverage
        else:  # SELL
            pnl_pct = (entry_price - current_price) / entry_price * self.leverage
        
        pnl_amount = self.balance * self.position_size * pnl_pct
        
        # Take Profit
        if pnl_pct >= self.take_profit_pct:
            self.balance += pnl_amount
            self.total_trades += 1
            if pnl_amount > 0:
                self.winning_trades += 1
            
            self.trade_log.append({
                'action': 'TP_' + position_type,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_percent': pnl_pct,
                'pnl_amount': pnl_amount,
                'timestamp': datetime.datetime.now()
            })
            
            print(f"🎯 TAKE PROFIT {position_type}: {pnl_pct*100:+.2f}% (${pnl_amount:+.2f})")
            self.position = None
            return True
        
        # Stop Loss
        elif pnl_pct <= -self.stop_loss_pct:
            self.balance += pnl_amount
            self.total_trades += 1
            if pnl_amount > 0:
                self.winning_trades += 1
            
            self.trade_log.append({
                'action': 'SL_' + position_type,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_percent': pnl_pct,
                'pnl_amount': pnl_amount,
                'timestamp': datetime.datetime.now()
            })
            
            print(f"🛑 STOP LOSS {position_type}: {pnl_pct*100:+.2f}% (${pnl_amount:+.2f})")
            self.position = None
            return True
        
        return False

    def update_equity_curve(self, current_price: float):
        """Update equity curve and calculate drawdown"""
        current_equity = self.balance
        
        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']
            
            if position_type == 'BUY':
                unrealized_pnl = (current_price - entry_price) / entry_price * self.leverage
            else:  # SELL
                unrealized_pnl = (entry_price - current_price) / entry_price * self.leverage
            
            current_equity += self.balance * self.position_size * unrealized_pnl
        
        self.equity_curve.append(current_equity)
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def show_detailed_analysis(self, df: pd.DataFrame, signal_data: Dict):
        """Show detailed technical analysis"""
        if len(df) < 2:
            return
            
        current = df.iloc[-1]
        
        print("\n🔍 TECHNICAL ANALYSIS DETAILS:")
        print(f"   RSI: {current['rsi']:.1f} (Oversold<35, Overbought>65)")
        print(f"   EMA Signal: {'BULLISH' if current['ema_signal'] == 1 else 'BEARISH'}")
        print(f"   BB Position: {current['bb_position']:.2f} (Lower<0.3, Upper>0.7)")
        print(f"   Momentum 3m: {current['momentum_3']*100:+.3f}%")
        print(f"   Volume Ratio: {current['volume_ratio']:.2f}x")
        print(f"   Acceleration: {current['acceleration']*100:+.3f}%")

    def show_performance_dashboard(self, current_price: float, signal_data: Dict, df: pd.DataFrame):
        """Show comprehensive performance dashboard"""
        current_equity = self.balance
        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']
            
            if position_type == 'BUY':
                unrealized_pnl = (current_price - entry_price) / entry_price * self.leverage
            else:
                unrealized_pnl = (entry_price - current_price) / entry_price * self.leverage
            
            current_equity += self.balance * self.position_size * unrealized_pnl
        
        total_return = (current_equity - self.initial_balance) / self.initial_balance * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print("\n" + "="*80)
        print("🔥 AGGRESSIVE SCALPING DASHBOARD")
        print("="*80)
        print(f"🕒 Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Price: ${current_price:,.2f}")
        print("-"*80)
        
        # Balance & Equity
        print("💵 BALANCE & EQUITY:")
        print(f"   Balance: ${self.balance:,.2f}")
        print(f"   Equity: ${current_equity:,.2f}")
        print(f"   Total Return: {total_return:+.2f}%")
        print(f"   Max Drawdown: {self.max_drawdown:.2f}%")
        
        # Trading Stats
        print("\n📈 TRADING STATISTICS:")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Winning Trades: {self.winning_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        
        # Current Position
        print("\n📊 CURRENT POSITION:")
        if self.position:
            entry_price = self.position['entry_price']
            position_type = self.position['type']
            
            if position_type == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price * self.leverage
            else:
                pnl_pct = (entry_price - current_price) / entry_price * self.leverage
            
            pnl_amount = self.balance * self.position_size * pnl_pct
            pnl_color = "🟢" if pnl_pct > 0 else "🔴"
            
            print(f"   Type: {position_type}")
            print(f"   Entry: ${entry_price:,.2f}")
            print(f"   Size: {self.position['size']:.6f} BTC")
            print(f"   P&L: {pnl_color} {pnl_pct*100:+.2f}% (${pnl_amount:+.2f})")
            print(f"   To TP: {(self.take_profit_pct - pnl_pct)*100:.2f}%")
            print(f"   To SL: {(self.stop_loss_pct + pnl_pct)*100:.2f}%")
        else:
            print("   No active position - READY TO TRADE")
        
        # Current Signal
        print(f"\n🎯 TRADING SIGNAL:")
        signal_color = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "⚪"
        print(f"   Signal: {signal_color} {signal_data['signal']}")
        print(f"   Confidence: {signal_data['confidence']:.2f}")
        print(f"   Reason: {signal_data['reason']}")
        
        # Show detailed analysis
        self.show_detailed_analysis(df, signal_data)
        
        print("="*80)

    def run_bot(self):
        """Main bot execution loop"""
        print("\n🤖 STARTING AGGRESSIVE SCALPING BOT")
        print("Lower thresholds for more trading activity")
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                # Fetch market data
                df = self.fetch_ohlcv(limit=50)  # Less data needed for scalping
                if df is None:
                    print("❌ Failed to fetch market data")
                    time.sleep(30)
                    continue
                
                current_price = self.get_current_price()
                if current_price is None:
                    print("❌ Failed to get current price")
                    time.sleep(30)
                    continue
                
                # Calculate indicators
                df_with_indicators = self.calculate_technical_indicators(df)
                
                # Generate trading signal
                signal_data = self.generate_trading_signal(df_with_indicators)
                
                # Update equity curve
                self.update_equity_curve(current_price)
                
                # Check TP/SL first
                if self.check_take_profit_stop_loss(current_price):
                    time.sleep(20)  # Shorter wait after position close
                    continue
                
                # Show performance dashboard
                self.show_performance_dashboard(current_price, signal_data, df_with_indicators)
                
                # EXECUTE TRADES - LOWER CONFIDENCE THRESHOLD
                min_confidence = 0.50  # Was 0.70
                
                if not self.position and signal_data['signal'] != 'HOLD' and signal_data['confidence'] >= min_confidence:
                    position_size = self.calculate_position_size(current_price)
                    
                    if self.execute_order(signal_data['signal'], position_size, current_price):
                        self.position = {
                            'type': signal_data['signal'],
                            'entry_price': current_price,
                            'size': position_size,
                            'timestamp': datetime.datetime.now(),
                            'signal_confidence': signal_data['confidence']
                        }
                        print(f"✅ POSITION OPENED: {signal_data['signal']} {position_size:.6f} BTC")
                        print(f"   Entry: ${current_price:,.2f}")
                        print(f"   Leverage: {self.leverage}x")
                        print(f"   TP: {self.take_profit_pct*100}% | SL: {self.stop_loss_pct*100}%")
                
                # Faster iterations for scalping
                print(f"⏰ Next update in 30 seconds...\n")
                time.sleep(30)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Bot stopped after {iteration} iterations")
            
            # Final performance summary
            if self.equity_curve:
                final_equity = self.equity_curve[-1]
                total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
                win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                
                print("\n🎯 FINAL PERFORMANCE SUMMARY:")
                print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
                print(f"💰 Final Equity: ${final_equity:.2f}")
                print(f"📈 Total Return: {total_return:+.2f}%")
                print(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")
                print(f"📊 Total Trades: {self.total_trades}")
                print(f"🎯 Win Rate: {win_rate:.1f}%")

def main():
    # Initialize bot
    bot = AggressiveBitcoinScalper()
    
    # Run the bot
    bot.run_bot()

if __name__ == "__main__":
    main()
