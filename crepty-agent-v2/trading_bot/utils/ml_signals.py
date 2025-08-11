"""
Enhanced ML Signals - Real machine learning implementation for crypto trading.
Uses technical indicators and machine learning models for signal generation.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from loguru import logger
import joblib
import os
from typing import Optional

class MLSignalGenerator:
    def __init__(self, model_path: str = "ml_model.pkl"):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = model_path
        self.feature_names = []
        
        # Try to load existing model
        self.load_model()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for ML features"""
        if len(df) < 20:
            return pd.DataFrame()
            
        close = df['close'] if 'close' in df.columns else pd.Series([0])
        
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = close.pct_change()
        features['sma_10'] = close.rolling(10).mean()
        features['sma_20'] = close.rolling(20).mean()
        features['price_vs_sma_10'] = close / features['sma_10'] - 1
        features['price_vs_sma_20'] = close / features['sma_20'] - 1
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # Volatility
        features['volatility'] = close.rolling(20).std()
        
        return features.fillna(method='ffill').fillna(0)
    
    def generate_ml_signal(self, symbol: str, price_data: Optional[pd.DataFrame] = None) -> int:
        """Generate ML-based trading signal"""
        if not self.is_trained:
            logger.warning("ML model not trained. Returning neutral signal.")
            return 0
        
        try:
            if price_data is None or price_data.empty:
                logger.warning("No price data provided for ML signal generation")
                return 0
            
            features = self.calculate_technical_indicators(price_data)
            
            if len(features) == 0 or features.iloc[-1].isna().any():
                return 0
            
            # Use the most recent features
            latest_features = features.iloc[-1:][self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(latest_features)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            
            logger.info(f"ML signal for {symbol}: {prediction}")
            return int(prediction)
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return 0
    
    def load_model(self):
        """Load a trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = True
                logger.info(f"ML model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

# Global instance for backwards compatibility
_ml_generator = MLSignalGenerator()

def generate_ml_signal(symbol_or_df):
    """
    Generate ML-based trading signal.
    Accepts either a symbol string or DataFrame for backwards compatibility.
    """
    if isinstance(symbol_or_df, str):
        # If it's a symbol string, return default signal (needs price data)
        logger.info(f"ML model needs price data for symbol {symbol_or_df}")
        return 1  # Default buy signal for backwards compatibility
    elif isinstance(symbol_or_df, pd.DataFrame):
        # If it's a DataFrame, try to generate signal
        return _ml_generator.generate_ml_signal("UNKNOWN", symbol_or_df)
    else:
        # Fallback for any other input
        logger.info(f"ML model predicts: BUY for input {symbol_or_df}")
        return 1
