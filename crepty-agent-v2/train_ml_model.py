#!/usr/bin/env python3
"""
ML Model Trainer - Train machine learning models for trading signals.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_ml_model_with_historical_data():
    """Train ML model using available historical data"""
    from trading_bot.utils.ml_signals import MLSignalGenerator
    
    # Look for historical data files
    data_files = ['BTCUSDT_1h.csv', 'trade_log.csv', 'trade_log_clean.csv']
    
    data_df = None
    for file in data_files:
        if os.path.exists(file):
            try:
                logger.info(f"Loading data from {file}")
                df = pd.read_csv(file)
                
                # Check if it has the required columns for price data
                required_columns = ['close']
                if all(col in df.columns for col in required_columns):
                    data_df = df
                    logger.info(f"Using {file} for training data")
                    break
                else:
                    logger.info(f"File {file} doesn't have required columns: {required_columns}")
                    
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
    
    if data_df is None:
        logger.warning("No suitable historical data found for training")
        logger.info("Creating synthetic data for demonstration...")
        
        # Create synthetic price data for demonstration
        np.random.seed(42)
        n_days = 1000
        
        # Generate synthetic price series with some trend and volatility
        returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
        prices = [100.0]  # Starting price
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Create DataFrame
        dates = pd.date_range(start='2023-01-01', periods=len(prices), freq='H')
        data_df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'open': prices,
            'volume': np.random.uniform(1000, 10000, len(prices))
        })
        
        logger.info("Created synthetic data with 1000 data points")
    
    # Prepare data
    if 'timestamp' in data_df.columns:
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df.set_index('timestamp', inplace=True)
    
    # Ensure we have numeric data
    for col in ['close', 'high', 'low', 'open', 'volume']:
        if col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    
    # Remove any rows with NaN values
    data_df = data_df.dropna()
    
    if len(data_df) < 100:
        logger.error("Insufficient data for training (need at least 100 data points)")
        return False
    
    # Initialize and train the ML model
    logger.info("Initializing ML signal generator...")
    ml_generator = MLSignalGenerator()
    
    # Create labels for training (simplified - predict if price will go up in next 5 periods)
    logger.info("Creating training labels...")
    data_df['future_return'] = data_df['close'].shift(-5) / data_df['close'] - 1
    data_df['label'] = 0  # Default hold
    data_df.loc[data_df['future_return'] > 0.01, 'label'] = 1   # Buy if >1% gain
    data_df.loc[data_df['future_return'] < -0.01, 'label'] = -1  # Sell if >1% loss
    
    # Remove rows with NaN labels
    training_data = data_df.dropna()
    
    if len(training_data) < 50:
        logger.error("Insufficient training data after label creation")
        return False
    
    logger.info(f"Training on {len(training_data)} data points")
    logger.info(f"Label distribution: {training_data['label'].value_counts().to_dict()}")
    
    # Train the model
    try:
        features = ml_generator.calculate_technical_indicators(training_data)
        
        if features.empty:
            logger.error("Failed to calculate technical indicators")
            return False
        
        # Align features with labels
        common_index = features.index.intersection(training_data.index)
        features_aligned = features.loc[common_index]
        labels_aligned = training_data.loc[common_index, 'label']
        
        # Remove any remaining NaN values
        mask = ~(features_aligned.isna().any(axis=1) | labels_aligned.isna())
        features_final = features_aligned[mask]
        labels_final = labels_aligned[mask]
        
        if len(features_final) < 30:
            logger.error("Insufficient clean data for training")
            return False
        
        logger.info(f"Final training set: {len(features_final)} samples")
        
        # Store feature names for later use
        ml_generator.feature_names = features_final.columns.tolist()
        
        # Fit scaler and model
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            features_final, labels_final, test_size=0.2, random_state=42
        )
        
        # Fit scaler
        X_train_scaled = ml_generator.scaler.fit_transform(X_train)
        X_test_scaled = ml_generator.scaler.transform(X_test)
        
        # Train model
        ml_generator.model.fit(X_train_scaled, y_train)
        ml_generator.is_trained = True
        
        # Evaluate model
        from sklearn.metrics import accuracy_score, classification_report
        
        y_pred = ml_generator.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training completed!")
        logger.info(f"Test accuracy: {accuracy:.3f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        import joblib
        model_data = {
            'model': ml_generator.model,
            'scaler': ml_generator.scaler,
            'feature_names': ml_generator.feature_names,
            'accuracy': accuracy,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, 'ml_model.pkl')
        logger.info("Model saved to ml_model.pkl")
        
        # Test signal generation
        logger.info("Testing signal generation...")
        test_signal = ml_generator.generate_ml_signal('TEST', training_data.tail(50))
        logger.info(f"Test signal: {test_signal}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function"""
    logger.info("🤖 Starting ML Model Training")
    
    success = train_ml_model_with_historical_data()
    
    if success:
        logger.info("✅ ML model training completed successfully!")
        print("\n" + "="*50)
        print("🎉 ML MODEL TRAINING COMPLETE!")
        print("="*50)
        print("\nYour ML model is now ready to generate trading signals.")
        print("The model has been saved as 'ml_model.pkl'")
        print("\nNext steps:")
        print("1. Test the model: python -c \"from trading_bot.utils.ml_signals import generate_ml_signal; print('ML Model Ready!')\"")
        print("2. Start your trading agent with enhanced ML signals")
    else:
        logger.error("❌ ML model training failed")
        print("\n" + "="*50)
        print("⚠️ ML MODEL TRAINING FAILED")
        print("="*50)
        print("\nPlease check the logs above for error details.")
        print("You may need to:")
        print("1. Ensure you have historical price data")
        print("2. Install required dependencies: pip install scikit-learn")

if __name__ == "__main__":
    main()
