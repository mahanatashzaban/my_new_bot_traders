#!/usr/bin/env python3
"""
Live trading script - ONLY RUN AFTER SUCCESSFUL VALIDATION
"""

import time
import logging
from trading_engine import TradingEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

def live_trading():
    print("🚀 LIVE TRADING ACTIVATED")
    print("⚠️  WARNING: Real money is at risk!")
    print("=" * 50)
    
    confirmation = input("Are you sure you want to start LIVE TRADING? (yes/no): ")
    if confirmation.lower() != 'yes':
        print("Live trading cancelled.")
        return
    
    # Initialize and run bot
    bot = TradingEngine()
    
    # Force model loading
    if not bot.model_trainer.load_models():
        print("❌ No trained models found. Run training first.")
        return
    
    print("✅ Models loaded successfully")
    print("💰 Starting LIVE trading...")
    
    bot.run_bot()

if __name__ == "__main__":
    live_trading()
