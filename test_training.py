#!/usr/bin/env python3
"""
Test the complete training pipeline
"""

from train_ml_complete import train_complete_pipeline, verify_training

if __name__ == "__main__":
    print("🧪 TESTING COMPLETE TRAINING PIPELINE")
    
    # Run complete training
    success = train_complete_pipeline()
    
    if success:
        print("\n✅ PIPELINE TEST PASSED!")
        verify_training()
    else:
        print("\n❌ PIPELINE TEST FAILED!")
