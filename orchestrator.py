#!/usr/bin/env python3
"""
Main orchestrator - Controls the complete workflow
"""

import sys
from train_model import train_and_validate
from validate_model import main as validate_model
from live_trade import live_trading

def main():
    print("🎛️  TRADING BOT ORCHESTRATOR")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Train Model")
        print("2. Validate Model")
        print("3. Live Trade (ONLY AFTER VALIDATION)")
        print("4. Full Pipeline (Train → Validate → Trade)")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            print("\n" + "="*30)
            train_and_validate()
            
        elif choice == '2':
            print("\n" + "="*30)
            validate_model()
            
        elif choice == '3':
            print("\n" + "="*30)
            live_trading()
            
        elif choice == '4':
            print("\n" + "="*30)
            print("🚀 STARTING FULL PIPELINE")
            
            # Train
            if train_and_validate():
                # Validate
                if validate_model():
                    # Live trade
                    live_trading()
                else:
                    print("❌ Pipeline stopped: Validation failed")
            else:
                print("❌ Pipeline stopped: Training failed")
                
        elif choice == '5':
            print("👋 Exiting...")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
