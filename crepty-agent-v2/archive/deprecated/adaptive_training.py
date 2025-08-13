#!/usr/bin/env python3
"""
Adaptive Training Loop - Learns from actual trading performance to improve models.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime, timedelta
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class AdaptiveTrainer:
    def __init__(self):
        self.models = {}
        self.training_scores = []
        self.best_model = None
        self.best_score = 0.0
        
    def load_real_trading_data(self):
        """Load and process real trading data for training"""
        print("📊 Loading real trading data...")
        
        # Load futures trades
        if os.path.exists('futures_trades_log.csv'):
            df = pd.read_csv('futures_trades_log.csv')
            print(f"   - Loaded {len(df)} futures trades")
            
            # Create features from real trading data
            features = self.extract_features_from_trades(df)
            labels = self.create_labels_from_performance(df)
            
            return features, labels
        else:
            print("❌ No trading data found. Using synthetic data.")
            return self.generate_synthetic_data()
    
    def extract_features_from_trades(self, df):
        """Extract features from actual trading data"""
        features = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < 5:
                continue
                
            # Price features
            prices = symbol_data['price'].values
            atr_values = symbol_data['atr'].values
            
            # Technical indicators from real data
            price_change = np.diff(prices, prepend=prices[0])
            volatility = np.std(price_change)
            trend = np.polyfit(range(len(prices)), prices, 1)[0]
            
            # Position sizing patterns
            position_sizes = np.abs(symbol_data['new_pos'].values)
            avg_position_size = np.mean(position_sizes[position_sizes > 0])
            
            # Trading frequency
            trading_frequency = len(symbol_data)
            
            # Recent performance
            recent_pnl = symbol_data['realized_pnl'].tail(10).sum()
            
            feature_vector = [
                np.mean(prices),           # avg price
                volatility,                # volatility
                trend,                     # trend
                np.mean(atr_values),       # avg ATR
                avg_position_size,         # avg position
                trading_frequency,         # frequency
                recent_pnl,                # recent PnL
                np.std(prices),            # price volatility
                len(symbol_data),          # total trades
                symbol_data['realized_pnl'].sum()  # total PnL
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def create_labels_from_performance(self, df):
        """Create labels based on actual trading performance"""
        labels = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < 5:
                continue
            
            # Performance-based labeling
            total_pnl = symbol_data['realized_pnl'].sum()
            win_rate = (symbol_data['realized_pnl'] > 0).mean()
            recent_pnl = symbol_data['realized_pnl'].tail(5).sum()
            
            # Label: 0=bad, 1=good, 2=excellent
            if recent_pnl > 0 and win_rate > 0.6:
                label = 2  # excellent
            elif recent_pnl > 0 or win_rate > 0.4:
                label = 1  # good  
            else:
                label = 0  # bad
                
            labels.append(label)
        
        return np.array(labels)
    
    def generate_synthetic_data(self):
        """Generate synthetic training data as fallback"""
        print("🎲 Generating synthetic training data...")
        
        n_samples = 1000
        n_features = 10
        
        # Create synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Create realistic labels based on feature combinations
        y = []
        for i in range(n_samples):
            # Simulate trading logic
            score = (X[i, 0] * 0.3 +  # price trend
                    X[i, 1] * -0.2 +   # volatility (negative)
                    X[i, 2] * 0.4 +    # momentum
                    np.random.normal(0, 0.1))  # noise
            
            if score > 0.5:
                y.append(2)  # excellent
            elif score > 0:
                y.append(1)  # good
            else:
                y.append(0)  # bad
        
        return X, np.array(y)
    
    def train_multiple_models(self, X, y, iteration):
        """Train multiple models and select best"""
        print(f"🧠 Training iteration {iteration}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42+iteration)
        
        # Different model configurations
        models_to_try = {
            f'RandomForest_{iteration}': RandomForestClassifier(
                n_estimators=100 + iteration*20,
                max_depth=10 + iteration,
                random_state=42+iteration
            ),
            f'GradientBoosting_{iteration}': GradientBoostingClassifier(
                n_estimators=80 + iteration*15,
                max_depth=6 + iteration//2,
                learning_rate=0.1 + iteration*0.01,
                random_state=42+iteration
            )
        }
        
        best_iteration_model = None
        best_iteration_score = 0
        
        for name, model in models_to_try.items():
            print(f"   Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            print(f"   {name} accuracy: {score:.3f}")
            
            # Save model
            self.models[name] = {
                'model': model,
                'score': score,
                'iteration': iteration,
                'timestamp': datetime.now()
            }
            
            # Track best model overall
            if score > self.best_score:
                self.best_score = score
                self.best_model = name
                best_iteration_model = name
                
            if score > best_iteration_score:
                best_iteration_score = score
        
        self.training_scores.append({
            'iteration': iteration,
            'best_score': best_iteration_score,
            'timestamp': datetime.now()
        })
        
        print(f"   ✅ Best iteration score: {best_iteration_score:.3f}")
        print(f"   🏆 Overall best: {self.best_model} ({self.best_score:.3f})")
        
        return best_iteration_score
    
    def save_best_model(self):
        """Save the best performing model"""
        if self.best_model and self.best_model in self.models:
            model_data = self.models[self.best_model]
            
            # Save model
            model_path = 'ml_model_best.joblib'
            joblib.dump(model_data['model'], model_path)
            
            # Save metadata
            metadata = {
                'model_name': self.best_model,
                'accuracy': self.best_score,
                'training_date': datetime.now().isoformat(),
                'total_iterations': len(self.training_scores)
            }
            
            import json
            with open('ml_model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"💾 Saved best model: {self.best_model} (accuracy: {self.best_score:.3f})")
            return True
        
        return False
    
    def run_adaptive_training(self, max_iterations=10, min_improvement=0.01):
        """Run adaptive training loop with early stopping"""
        print("🚀 ADAPTIVE TRAINING LOOP")
        print("="*50)
        print(f"Max iterations: {max_iterations}")
        print(f"Min improvement: {min_improvement}")
        print()
        
        # Load data
        X, y = self.load_real_trading_data()
        print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Label distribution: {np.bincount(y)}")
        print()
        
        previous_best = 0
        no_improvement_count = 0
        
        for iteration in range(1, max_iterations + 1):
            print(f"🔄 ITERATION {iteration}/{max_iterations}")
            print("-" * 30)
            
            # Train models for this iteration
            current_score = self.train_multiple_models(X, y, iteration)
            
            # Check for improvement
            improvement = current_score - previous_best
            
            if improvement < min_improvement:
                no_improvement_count += 1
                print(f"⚠️ No significant improvement ({improvement:.3f} < {min_improvement})")
            else:
                no_improvement_count = 0
                print(f"📈 Improvement: {improvement:.3f}")
            
            previous_best = max(previous_best, current_score)
            
            # Early stopping
            if no_improvement_count >= 3:
                print(f"🛑 Early stopping: No improvement for 3 iterations")
                break
            
            print()
            time.sleep(5)  # Brief pause between iterations
        
        # Save best model
        self.save_best_model()
        
        # Summary
        print("="*50)
        print("📊 TRAINING SUMMARY")
        print("="*50)
        print(f"🏆 Best model: {self.best_model}")
        print(f"🎯 Best accuracy: {self.best_score:.3f}")
        print(f"🔄 Total iterations: {len(self.training_scores)}")
        print(f"📈 Score progression: {[s['best_score'] for s in self.training_scores]}")

def main():
    """Main function"""
    print("🤖 ADAPTIVE ML TRAINING SYSTEM")
    print("="*40)
    
    trainer = AdaptiveTrainer()
    
    try:
        trainer.run_adaptive_training(max_iterations=8, min_improvement=0.005)
        print("🎉 Adaptive training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        if trainer.best_model:
            trainer.save_best_model()
    except Exception as e:
        print(f"❌ Training error: {e}")

if __name__ == "__main__":
    main()
