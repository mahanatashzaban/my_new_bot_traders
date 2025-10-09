#!/usr/bin/env python3
"""
BTC/USDT Trading Bot - Main Entry Point
"""

import os
from trading_engine import TradingEngine

def main():
    print("🚀 Starting BTC/USDT Trading Bot")
    print("=" * 50)
    
    # Check for API keys
    if not os.getenv('BITFINEX_API_KEY') or not os.getenv('BITFINEX_SECRET'):
        print("⚠️  Warning: BITFINEX_API_KEY and BITFINEX_SECRET environment variables not set")
        print("The bot will run in simulation mode only")
    
    # Create and run trading bot
    bot = TradingEngine()
    bot.run_bot()

if __name__ == "__main__":
    main()
