#!/usr/bin/env python3
"""
Continuous ML Training Loop - Automatically retrain models as new data arrives.
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
import threading
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ContinuousTrainer:
    def __init__(self):
        self.is_running = False
        self.training_thread = None
        self.last_training_time = None
        self.min_new_trades = 50  # Minimum new trades before retraining
        self.training_interval_hours = 6  # Retrain every 6 hours
        self.max_training_time_minutes = 30  # Max training time
        
        # Performance tracking
        self.training_history = []
        self.model_versions = []
        
        logger.info("🔄 Continuous Trainer initialized")

    def should_retrain(self):
        """Check if model should be retrained based on new data"""
        try:
            # Check if enough time has passed
            if self.last_training_time:
                time_since_last = datetime.now() - self.last_training_time
                if time_since_last.total_seconds() < self.training_interval_hours * 3600:
                    return False, "Too soon since last training"
            
            # Check for new trade data
            trade_files = [
                'futures_trades_log.csv',
                'trade_log.csv',
                'trade_log_clean.csv'
            ]
            
            total_new_trades = 0
            for file in trade_files:
                if os.path.exists(file):
                    df = pd.read_csv(file)
                    
                    # Count trades since last training
                    if self.last_training_time and 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        new_trades = df[df['timestamp'] > self.last_training_time]
                        total_new_trades += len(new_trades)
                    else:
                        total_new_trades += len(df)
            
            if total_new_trades >= self.min_new_trades:
                return True, f"Found {total_new_trades} new trades"
            else:
                return False, f"Only {total_new_trades} new trades (need {self.min_new_trades})"
                
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
            return False, str(e)

    def train_model_iteration(self):
        """Single training iteration"""
        try:
            logger.info("🧠 Starting ML model training iteration...")
            start_time = time.time()
            
            # Import and run training
            from train_ml_model import main as train_main
            
            # Capture training results
            training_result = train_main()
            
            training_time = time.time() - start_time
            
            # Record training session
            training_record = {
                'timestamp': datetime.now(),
                'training_time_seconds': training_time,
                'success': training_result is not False,
                'version': len(self.training_history) + 1
            }
            
            self.training_history.append(training_record)
            self.last_training_time = datetime.now()
            
            logger.info(f"✅ Training iteration completed in {training_time:.1f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training iteration failed: {e}")
            return False

    def run_training_loop(self):
        """Main continuous training loop"""
        logger.info("🔄 Starting continuous training loop...")
        
        while self.is_running:
            try:
                # Check if we should retrain
                should_train, reason = self.should_retrain()
                
                if should_train:
                    logger.info(f"🎯 Retraining triggered: {reason}")
                    success = self.train_model_iteration()
                    
                    if success:
                        logger.info("✅ Model retrained successfully")
                    else:
                        logger.error("❌ Model retraining failed")
                else:
                    logger.debug(f"⏳ Skipping training: {reason}")
                
                # Wait before next check (check every hour)
                time.sleep(3600)  # 1 hour
                
            except KeyboardInterrupt:
                logger.info("🛑 Training loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in training loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def start_continuous_training(self):
        """Start the continuous training in background thread"""
        if self.is_running:
            logger.warning("Training loop already running")
            return
        
        self.is_running = True
        self.training_thread = threading.Thread(target=self.run_training_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("🚀 Continuous training started in background")

    def stop_continuous_training(self):
        """Stop the continuous training"""
        self.is_running = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
        
        logger.info("🛑 Continuous training stopped")

    def get_training_status(self):
        """Get current training status"""
        status = {
            'is_running': self.is_running,
            'last_training': self.last_training_time,
            'total_training_sessions': len(self.training_history),
            'next_check_in_hours': self.training_interval_hours
        }
        
        if self.training_history:
            recent_training = self.training_history[-1]
            status['last_training_duration'] = recent_training['training_time_seconds']
            status['last_training_success'] = recent_training['success']
        
        return status

    def force_training_now(self):
        """Force immediate training regardless of conditions"""
        logger.info("🔥 Forcing immediate training...")
        return self.train_model_iteration()

def run_batch_training_loop(iterations=10, delay_minutes=30):
    """Run a batch of training iterations with delays"""
    logger.info(f"🔄 Starting batch training: {iterations} iterations, {delay_minutes}min intervals")
    
    results = []
    
    for i in range(iterations):
        logger.info(f"📚 Training iteration {i+1}/{iterations}")
        
        try:
            # Import and run training
            from train_ml_model import main as train_main
            start_time = time.time()
            
            result = train_main()
            training_time = time.time() - start_time
            
            results.append({
                'iteration': i+1,
                'success': result is not False,
                'training_time': training_time,
                'timestamp': datetime.now()
            })
            
            logger.info(f"✅ Iteration {i+1} completed in {training_time:.1f}s")
            
            # Wait before next iteration (except last)
            if i < iterations - 1:
                logger.info(f"⏳ Waiting {delay_minutes} minutes before next iteration...")
                time.sleep(delay_minutes * 60)
                
        except Exception as e:
            logger.error(f"❌ Iteration {i+1} failed: {e}")
            results.append({
                'iteration': i+1,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            })
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"📊 Batch training complete: {successful}/{iterations} successful")
    
    return results

def schedule_training():
    """Set up scheduled training"""
    # Schedule training every 6 hours
    schedule.every(6).hours.do(lambda: ContinuousTrainer().train_model_iteration())
    
    # Schedule daily comprehensive training
    schedule.every().day.at("02:00").do(lambda: run_batch_training_loop(3, 60))
    
    logger.info("📅 Training schedule set up:")
    logger.info("   - Every 6 hours: Single training iteration")
    logger.info("   - Daily at 2 AM: 3 comprehensive iterations")

def main():
    """Main function with training options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous ML Training System")
    parser.add_argument('--mode', choices=['continuous', 'batch', 'once', 'schedule'], 
                       default='once', help='Training mode')
    parser.add_argument('--iterations', type=int, default=5, 
                       help='Number of training iterations for batch mode')
    parser.add_argument('--delay', type=int, default=30, 
                       help='Delay between iterations in minutes')
    
    args = parser.parse_args()
    
    logger.info("🚀 ML CONTINUOUS TRAINING SYSTEM")
    logger.info("="*50)
    
    if args.mode == 'once':
        logger.info("🎯 Running single training iteration...")
        from train_ml_model import main as train_main
        result = train_main()
        logger.info("✅ Single training completed" if result else "❌ Training failed")
        
    elif args.mode == 'batch':
        logger.info(f"🔄 Running batch training: {args.iterations} iterations")
        results = run_batch_training_loop(args.iterations, args.delay)
        successful = sum(1 for r in results if r.get('success', False))
        logger.info(f"📊 Batch complete: {successful}/{args.iterations} successful")
        
    elif args.mode == 'continuous':
        logger.info("🔄 Starting continuous training...")
        trainer = ContinuousTrainer()
        trainer.start_continuous_training()
        
        try:
            # Keep main thread alive
            while True:
                status = trainer.get_training_status()
                logger.info(f"📊 Status: {status['total_training_sessions']} sessions completed")
                time.sleep(300)  # Status update every 5 minutes
        except KeyboardInterrupt:
            logger.info("🛑 Stopping continuous training...")
            trainer.stop_continuous_training()
            
    elif args.mode == 'schedule':
        logger.info("📅 Setting up scheduled training...")
        schedule_training()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("🛑 Stopping scheduled training...")
    
    logger.info("🎉 Training system finished")

if __name__ == "__main__":
    main()
