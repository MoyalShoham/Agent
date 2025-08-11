#!/usr/bin/env python3
"""
Fix ML model feature mismatch - retrain with current feature set
"""

def fix_ml_models():
    """Retrain ML models with current feature set"""
    
    print("🔧 FIXING ML MODEL FEATURE MISMATCH")
    print("=" * 40)
    print("Problem: Models expect 270 features, system provides 280")
    print("Solution: Retrain models with current 280-feature dataset")
    
    print("\n🎯 RUNNING EMERGENCY ML RETRAINING:")
    
    import subprocess
    import sys
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 'train_meta_learner.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ML models retrained successfully!")
            print("✅ Feature mismatch should be resolved")
            print("\n🚀 Now you can restart trading:")
            print("   python main.py")
        else:
            print("❌ ML training failed:")
            print(result.stderr)
            print("\n🔄 Alternative: Delete old models and use fallback:")
            print("   del trading_bot\\ai_models\\*.pkl")
            print("   python main.py")
            
    except Exception as e:
        print(f"❌ Error during ML training: {e}")
        print("\n🔄 Manual fix:")
        print("1. Delete old model files:")
        print("   del trading_bot\\ai_models\\*.pkl") 
        print("2. Restart with fresh training:")
        print("   python main.py")

if __name__ == "__main__":
    fix_ml_models()
