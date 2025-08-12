#!/usr/bin/env python3
"""
Simple Batch Training Loop - Train multiple times to improve model performance.
"""
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_training_batch(num_iterations=15, delay_minutes=0.1):
    """
    Run multiple training iterations to improve model performance.
    
    Args:
        num_iterations: Number of training cycles to run
        delay_minutes: Minutes to wait between training cycles
    """
    print("🚀 BATCH TRAINING SYSTEM")
    print("="*50)
    print(f"📊 Iterations: {num_iterations}")
    print(f"⏱️ Delay: {delay_minutes} minutes between iterations")
    print()
    
    results = []
    total_start = time.time()
    
    for i in range(num_iterations):
        print(f"🧠 TRAINING ITERATION {i+1}/{num_iterations}")
        print("-" * 40)
        
        iteration_start = time.time()
        
        try:
            # Run training
            os.system("python train_ml_model.py")
            
            iteration_time = time.time() - iteration_start
            
            results.append({
                'iteration': i+1,
                'success': True,
                'time_seconds': iteration_time,
                'timestamp': datetime.now()
            })
            
            print(f"✅ Iteration {i+1} completed in {iteration_time:.1f} seconds")
            
        except Exception as e:
            iteration_time = time.time() - iteration_start
            results.append({
                'iteration': i+1,
                'success': False,
                'error': str(e),
                'time_seconds': iteration_time,
                'timestamp': datetime.now()
            })
            print(f"❌ Iteration {i+1} failed: {e}")
        
        # Wait before next iteration (except for the last one)
        if i < num_iterations - 1:
            print(f"⏳ Waiting {delay_minutes} minutes before next iteration...")
            print()
            time.sleep(delay_minutes * 60)
    
    # Summary
    total_time = time.time() - total_start
    successful = sum(1 for r in results if r['success'])
    
    print("="*50)
    print("📊 BATCH TRAINING SUMMARY")
    print("="*50)
    print(f"✅ Successful iterations: {successful}/{num_iterations}")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"📈 Average time per iteration: {total_time/num_iterations/60:.1f} minutes")
    
    if successful > 0:
        avg_success_time = sum(r['time_seconds'] for r in results if r['success']) / successful / 60
        print(f"🎯 Average successful iteration time: {avg_success_time:.1f} minutes")
    
    print()
    print("💡 Expected improvements:")
    print("   - Better signal accuracy")
    print("   - Improved trade predictions")
    print("   - Enhanced model performance")
    print("   - More profitable signals")
    
    # 🔄 IMPORTANT: Restart recommendation
    print()
    print("🚀 NEXT STEPS:")
    print("   1. RESTART your trading agent: python main.py")
    print("   2. The new trained model will be automatically loaded")
    print("   3. Monitor improved signal quality")
    print()
    print("⚠️  IMPORTANT: You must restart main.py to use the new model!")
    
    return results

def quick_training_loop(iterations=3):
    """Quick training loop with shorter delays"""
    print("⚡ QUICK TRAINING LOOP")
    print("="*30)
    
    for i in range(iterations):
        print(f"🎯 Quick iteration {i+1}/{iterations}")
        
        start_time = time.time()
        os.system("python train_ml_model.py")
        duration = time.time() - start_time
        
        print(f"✅ Completed in {duration:.1f} seconds")
        
        # Short delay between iterations
        if i < iterations - 1:
            print("⏳ Brief pause...")
            time.sleep(30)  # 30 seconds
    
    print("🎉 Quick training complete!")

def main():
    """Main training options"""
    print("🤖 ML MODEL TRAINING LOOP OPTIONS")
    print("="*40)
    print("1. Quick Loop (3x, 30s delay)")
    print("2. Standard Loop (5x, 10min delay)")  
    print("3. Intensive Loop (10x, 15min delay)")
    print("4. Custom Loop")
    print("5. Single Training")
    print()
    
    try:
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            quick_training_loop(3)
            
        elif choice == '2':
            run_training_batch(5, 10)
            
        elif choice == '3':
            run_training_batch(10, 0.1)
            
        elif choice == '4':
            iterations = int(input("Number of iterations: "))
            delay = int(input("Delay in minutes: "))
            run_training_batch(iterations, delay)
            
        elif choice == '5':
            print("🎯 Running single training...")
            os.system("python train_ml_model.py")
            
        else:
            print("❌ Invalid choice. Running default quick loop...")
            quick_training_loop(3)
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
