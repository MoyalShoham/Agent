#!/usr/bin/env python3
"""
Training Control Center - Easy interface for all training options.
"""
import os
import sys

def show_menu():
    """Display training options menu"""
    print("🤖 ML TRAINING CONTROL CENTER")
    print("="*50)
    print()
    print("📚 TRAINING OPTIONS:")
    print()
    print("1. 🎯 Single Training")
    print("   - Run one training cycle")
    print("   - Quick and simple")
    print()
    print("2. ⚡ Quick Batch (3x)")
    print("   - 3 training cycles, 30s apart")
    print("   - Fast improvement")
    print()
    print("3. 🔄 Standard Batch (5x)")
    print("   - 5 training cycles, 10min apart")
    print("   - Balanced approach")
    print()
    print("4. 🧠 Adaptive Training")
    print("   - Uses your real trading data")
    print("   - Smart early stopping")
    print("   - Best for real improvement")
    print()
    print("5. 🚀 Intensive Training (10x)")
    print("   - 10 training cycles, 15min apart")
    print("   - Maximum optimization")
    print()
    print("6. 🔄 Continuous Training")
    print("   - Runs in background")
    print("   - Retrains automatically")
    print()
    print("7. 📊 Training Status")
    print("   - Check current model performance")
    print()
    print("0. Exit")
    print()

def run_option(choice):
    """Execute the selected training option"""
    if choice == '1':
        print("🎯 Running single training...")
        os.system("python train_ml_model.py")
        
    elif choice == '2':
        print("⚡ Starting quick batch training...")
        os.system("python batch_training.py")
        
    elif choice == '3':
        print("🔄 Starting standard batch training...")
        os.system("python -c \"from batch_training import run_training_batch; run_training_batch(5, 10)\"")
        
    elif choice == '4':
        print("🧠 Starting adaptive training...")
        os.system("python adaptive_training.py")
        
    elif choice == '5':
        print("🚀 Starting intensive training...")
        os.system("python -c \"from batch_training import run_training_batch; run_training_batch(10, 15)\"")
        
    elif choice == '6':
        print("🔄 Starting continuous training...")
        os.system("python continuous_training.py --mode continuous")
        
    elif choice == '7':
        show_training_status()
        
    elif choice == '0':
        print("👋 Goodbye!")
        return False
        
    else:
        print("❌ Invalid choice. Please try again.")
    
    return True

def show_training_status():
    """Show current training and model status"""
    print("📊 TRAINING STATUS")
    print("="*30)
    
    # Check for model files
    model_files = [
        'ml_model.joblib',
        'ml_model_best.joblib', 
        'ml_model_metadata.json'
    ]
    
    print("📁 Model Files:")
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size:,} bytes)")
        else:
            print(f"   ❌ {file} (missing)")
    
    # Check training data
    print("\n📊 Training Data:")
    data_files = [
        'futures_trades_log.csv',
        'trade_log.csv',
        'synthetic_meta_training.csv'
    ]
    
    total_trades = 0
    for file in data_files:
        if os.path.exists(file):
            try:
                import pandas as pd
                df = pd.read_csv(file)
                trades = len(df)
                total_trades += trades
                print(f"   ✅ {file}: {trades:,} trades")
            except:
                print(f"   ⚠️ {file}: Error reading")
        else:
            print(f"   ❌ {file}: Not found")
    
    print(f"\n🎯 Total Training Data: {total_trades:,} trades")
    
    # Performance estimate
    if total_trades > 100:
        print("✅ Sufficient data for quality training")
    elif total_trades > 50:
        print("⚠️ Limited data - consider more trading")
    else:
        print("❌ Insufficient data - need more trading history")
    
    print()

def main():
    """Main control center loop"""
    while True:
        try:
            show_menu()
            choice = input("Select option (0-7): ").strip()
            print()
            
            if not run_option(choice):
                break
                
            if choice != '0':
                input("\nPress Enter to continue...")
                print("\n" + "="*60 + "\n")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
