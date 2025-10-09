import time
import numpy as np
from data_manager import DataManager
from indicators import IndicatorEngine
from model_trainer import ModelTrainer
from config import TRADING_CONFIG

class TradingEngine:
    def __init__(self):
        self.data_manager = DataManager()
        self.indicator_engine = IndicatorEngine()
        self.model_trainer = ModelTrainer()
        self.position = None
        self.balance = TRADING_CONFIG['initial_balance']
        self.positions = []
        
    def initialize(self):
        """Initialize the trading bot"""
        print("Initializing Trading Bot...")
        
        # Load or train models
        if not self.model_trainer.load_models():
            print("Training new models...")
            historical_data = self.data_manager.fetch_historical_data(limit=2000)
            if historical_data is not None:
                self.model_trainer.train_models(historical_data)
            else:
                print("Failed to fetch historical data for training")
                return False
        
        print("Trading bot initialized successfully")
        return True
    
    def get_trading_signal(self, df):
        """Get trading signal from ML model"""
        try:
            # Prepare features for prediction
            X, _ = self.model_trainer.prepare_features(df.tail(50))  # Use last 50 candles
            
            if X.empty:
                return "HOLD", 0.0
            
            # Scale features
            X_scaled = self.model_trainer.scaler.transform(X)
            
            # Get prediction from XGBoost model
            model = self.model_trainer.models['xgb']
            prediction = model.predict(X_scaled)[-1]  # Get latest prediction
            probability = model.predict_proba(X_scaled)[-1]
            
            confidence = max(probability)
            
            if prediction == 1 and confidence > 0.6:  # BUY with 60%+ confidence
                return "BUY", confidence
            elif prediction == 0 and confidence > 0.6:  # SELL with 60%+ confidence
                return "SELL", confidence
            else:
                return "HOLD", confidence
                
        except Exception as e:
            print(f"Error getting trading signal: {e}")
            return "HOLD", 0.0
    
    def calculate_position_size(self, current_price, confidence):
        """Calculate position size based on risk management"""
        risk_amount = self.balance * TRADING_CONFIG['risk_per_trade']
        
        # Adjust position size based on confidence
        confidence_multiplier = min(confidence / 0.6, 1.5)  # Up to 1.5x for high confidence
        
        position_size = (risk_amount * confidence_multiplier) / current_price
        return position_size
    
    def execute_trade(self, signal, confidence, current_price):
        """Execute trade based on signal"""
        if signal == "HOLD":
            return
        
        position_size = self.calculate_position_size(current_price, confidence)
        
        if signal == "BUY" and not self.position:
            # Open long position
            self.position = {
                'type': 'LONG',
                'entry_price': current_price,
                'size': position_size,
                'entry_time': time.time(),
                'stop_loss': current_price * 0.99,  # 1% stop loss
                'take_profit': current_price * 1.02  # 2% take profit
            }
            print(f"🟢 OPEN LONG: {position_size:.6f} BTC @ ${current_price:.2f}")
            
        elif signal == "SELL" and not self.position:
            # Open short position
            self.position = {
                'type': 'SHORT', 
                'entry_price': current_price,
                'size': position_size,
                'entry_time': time.time(),
                'stop_loss': current_price * 1.01,  # 1% stop loss
                'take_profit': current_price * 0.98  # 2% take profit
            }
            print(f"🔴 OPEN SHORT: {position_size:.6f} BTC @ ${current_price:.2f}")
    
    def check_exit_conditions(self, current_price):
        """Check if we should exit current position"""
        if not self.position:
            return
        
        pnl = 0
        if self.position['type'] == 'LONG':
            pnl = (current_price - self.position['entry_price']) / self.position['entry_price']
            if current_price <= self.position['stop_loss'] or current_price >= self.position['take_profit']:
                self.close_position(current_price, "Stop Loss/Take Profit")
                
        elif self.position['type'] == 'SHORT':
            pnl = (self.position['entry_price'] - current_price) / self.position['entry_price'] 
            if current_price >= self.position['stop_loss'] or current_price <= self.position['take_profit']:
                self.close_position(current_price, "Stop Loss/Take Profit")
    
    def close_position(self, current_price, reason):
        """Close current position"""
        if not self.position:
            return
        
        # Calculate P&L
        if self.position['type'] == 'LONG':
            pnl_percent = (current_price - self.position['entry_price']) / self.position['entry_price']
        else:
            pnl_percent = (self.position['entry_price'] - current_price) / self.position['entry_price']
        
        pnl_amount = self.balance * pnl_percent
        self.balance += pnl_amount
        
        print(f"🟡 CLOSE {self.position['type']}: PnL {pnl_percent*100:.2f}% (${pnl_amount:.2f}) - {reason}")
        
        # Record trade
        trade_record = self.position.copy()
        trade_record.update({
            'exit_price': current_price,
            'exit_time': time.time(),
            'pnl_percent': pnl_percent,
            'pnl_amount': pnl_amount,
            'reason': reason
        })
        self.positions.append(trade_record)
        
        self.position = None
    
    def run_bot(self):
        """Main trading loop"""
        print("Starting trading bot...")
        
        if not self.initialize():
            print("Failed to initialize bot")
            return
        
        print("Bot is running. Press Ctrl+C to stop.")
        
        try:
            while True:
                # Fetch current data
                df = self.data_manager.fetch_historical_data(limit=100)
                current_price = self.data_manager.get_current_price()
                
                if df is not None and current_price is not None:
                    # Get trading signal
                    signal, confidence = self.get_trading_signal(df)
                    
                    print(f"Signal: {signal} (Confidence: {confidence:.2f}) | Price: ${current_price:.2f} | Balance: ${self.balance:.2f}")
                    
                    # Check exit conditions first
                    self.check_exit_conditions(current_price)
                    
                    # Execute new trade if no position
                    if not self.position:
                        self.execute_trade(signal, confidence, current_price)
                
                # Wait for next candle
                time.sleep(60)  # 1 minute
                
        except KeyboardInterrupt:
            print("\nBot stopped by user")
        except Exception as e:
            print(f"Bot error: {e}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print trading summary"""
        print("\n" + "="*50)
        print("TRADING SUMMARY")
        print("="*50)
        print(f"Initial Balance: ${TRADING_CONFIG['initial_balance']:.2f}")
        print(f"Final Balance: ${self.balance:.2f}")
        
        if self.positions:
            total_pnl = sum(trade['pnl_amount'] for trade in self.positions)
            winning_trades = [t for t in self.positions if t['pnl_amount'] > 0]
            win_rate = len(winning_trades) / len(self.positions) * 100
            
            print(f"Total Trades: {len(self.positions)}")
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Win Rate: {win_rate:.1f}%")
            print(f"Total P&L: ${total_pnl:.2f} ({total_pnl/TRADING_CONFIG['initial_balance']*100:.1f}%)")
