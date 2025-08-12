"""
Enhanced ML Signals - Real machine learning implementation for crypto trading.
Replaces the placeholder implementation with actual ML models.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger
import joblib
import os
from typing import Optional, Union

class MLSignalGenerator:
    def __init__(self, model_path: str = "ml_model.pkl"):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = model_path
        self.feature_names = []
        self.performance_metrics = {}
        
        # Try to load existing model
        self.load_model()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for ML features"""
        if len(df) < 50:
            return pd.DataFrame()
            
        close = df['close']
        high = df['high'] if 'high' in df.columns else close
        low = df['low'] if 'low' in df.columns else close
        volume = df['volume'] if 'volume' in df.columns else pd.Series([1] * len(df))
        
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        features['price_position'] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min())
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = close.rolling(period).mean()
            features[f'ema_{period}'] = close.ewm(span=period).mean()
            features[f'price_vs_sma_{period}'] = close / features[f'sma_{period}'] - 1
            features[f'price_vs_ema_{period}'] = close / features[f'ema_{period}'] - 1
        
        # Volatility features
        features['volatility_10'] = close.rolling(10).std()
        features['volatility_20'] = close.rolling(20).std()
        features['volatility_ratio'] = features['volatility_10'] / features['volatility_20']
        
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
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_squeeze'] = (features['bb_upper'] - features['bb_lower']) / bb_middle
        
        # Volume features
        features['volume_sma'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_sma']
        features['volume_price_trend'] = features['returns'] * features['volume_ratio']
        
        # Momentum features
        for period in [3, 7, 14]:
            features[f'momentum_{period}'] = close / close.shift(period) - 1
            features[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
        
        # Support/Resistance levels
        features['resistance_20'] = high.rolling(20).max()
        features['support_20'] = low.rolling(20).min()
        features['resistance_distance'] = (features['resistance_20'] - close) / close
        features['support_distance'] = (close - features['support_20']) / close
        
        # Trend strength
        features['trend_strength'] = abs(features['ema_10'] - features['ema_20']) / close
        
        # Market regime indicators
        features['volatility_regime'] = features['volatility_20'] > features['volatility_20'].rolling(50).quantile(0.7)
        features['trending_regime'] = abs(features['momentum_14']) > features['momentum_14'].rolling(50).std()
        
        return features.fillna(method='ffill').fillna(0)
    
    def create_labels(self, df: pd.DataFrame, lookforward: int = 5, threshold: float = 0.01) -> pd.Series:
        """Create trading labels based on future price movements"""
        close = df['close']
        future_returns = close.shift(-lookforward) / close - 1
        
        labels = pd.Series(0, index=df.index)  # Default: hold
        labels[future_returns > threshold] = 1    # Buy signal
        labels[future_returns < -threshold] = -1  # Sell signal
        
        return labels
    
    def prepare_training_data(self, price_data: pd.DataFrame) -> tuple:
        """Prepare features and labels for training"""
        features = self.calculate_technical_indicators(price_data)
        labels = self.create_labels(price_data)
        
        # Remove rows with insufficient data
        valid_idx = features.index.intersection(labels.index)
        features = features.loc[valid_idx]
        labels = labels.loc[valid_idx]
        
        # Remove rows with any NaN values
        mask = ~(features.isna().any(axis=1) | labels.isna())
        features = features[mask]
        labels = labels[mask]
        
        self.feature_names = features.columns.tolist()
        return features, labels
    
    def train_model(self, price_data: pd.DataFrame, test_size: float = 0.2) -> dict:
        """Train the ML model on historical price data"""
        logger.info("Training ML signal model...")
        
        features, labels = self.prepare_training_data(price_data)
        
        if len(features) < 100:
            logger.warning("Insufficient data for training ML model")
            return {"error": "Insufficient training data"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            
            logger.info(f"{name} accuracy: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = model
        
        self.model = best_model
        self.is_trained = True
        
        # Calculate performance metrics
        y_pred = self.model.predict(X_test_scaled)
        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'buy_signals': sum(y_pred == 1),
            'sell_signals': sum(y_pred == -1),
            'hold_signals': sum(y_pred == 0),
            'total_samples': len(y_pred)
        }
        
        # Save model
        self.save_model()
        
        logger.info(f"Model trained successfully. Accuracy: {self.performance_metrics['accuracy']:.3f}")
        return self.performance_metrics
    
    def generate_ml_signal(self, symbol: str, price_data: Optional[pd.DataFrame] = None) -> int:
        """Generate ML-based trading signal for a symbol"""
        if not self.is_trained:
            logger.warning("ML model not trained. Returning neutral signal.")
            return 0
        
        try:
            if price_data is None:
                # In practice, you would fetch recent price data for the symbol
                logger.warning("No price data provided for ML signal generation")
                return 0
            
            features = self.calculate_technical_indicators(price_data)
            
            if len(features) == 0:
                return 0
            
            # Use the most recent features
            latest_features = features.iloc[-1:][self.feature_names]
            
            if latest_features.isna().any().any():
                logger.warning("NaN values in features, returning neutral signal")
                return 0
            
            # Scale features
            X_scaled = self.scaler.transform(latest_features)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Add confidence threshold
            max_prob = max(probabilities)
            if max_prob < 0.6:  # Require 60% confidence
                prediction = 0  # Hold if not confident
            
            logger.info(f"ML signal for {symbol}: {prediction} (confidence: {max_prob:.3f})")
            return int(prediction)
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return 0
    
    def save_model(self):
        """Save the trained model to disk"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'performance_metrics': self.performance_metrics
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.performance_metrics = model_data.get('performance_metrics', {})
                self.is_trained = True
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")


# Global instance for backwards compatibility
_ml_generator = MLSignalGenerator()

def generate_ml_signal(symbol: str, price_data: Optional[pd.DataFrame] = None) -> int:
    """
    Generate ML-based trading signal.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        price_data: Historical price data DataFrame with OHLCV columns
        
    Returns:
        int: 1 for buy, 0 for hold, -1 for sell
    """
    return _ml_generator.generate_ml_signal(symbol, price_data)

def train_ml_model(price_data: pd.DataFrame) -> dict:
    """Train the ML model with historical price data"""
    return _ml_generator.train_model(price_data)

def get_ml_performance() -> dict:
    """Get ML model performance metrics"""
    return _ml_generator.performance_metrics
