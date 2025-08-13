"""
Enhanced ML Signals - Production-grade machine learning for crypto trading.
Features ensemble models, real-time feature engineering, and dynamic model selection.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from loguru import logger
import joblib
import os
import warnings
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
import ta

warnings.filterwarnings('ignore')

class EnhancedMLSignalGenerator:
    """
    Advanced ML signal generator with multiple models and ensemble voting.
    """
    
    def __init__(self, model_path: str = "enhanced_ml_model.pkl"):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_names = []
        self.feature_importance = {}
        self.model_weights = {}
        
        # Initialize models
        self._initialize_models()
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_history = []
        
        # Load existing model if available
        self.load_model()

    def _initialize_models(self):
        """Initialize ensemble of ML models"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss'
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
        
        # Default model weights (will be updated based on performance)
        self.model_weights = {
            'random_forest': 0.3,
            'gradient_boost': 0.25,
            'xgboost': 0.3,
            'logistic': 0.15
        }

    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators and features"""
        if len(df) < 50:
            return pd.DataFrame()
        try:
            features = pd.DataFrame(index=df.index)
            
            # Ensure we have OHLCV data
            if 'close' not in df.columns:
                logger.warning("Missing 'close' column in price data")
                return pd.DataFrame()
            
            close = df['close']
            high = df.get('high', close)
            low = df.get('low', close)
            volume = df.get('volume', pd.Series([1]*len(df), index=df.index))
            
            # Price-based features
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))
            features['price_momentum_5'] = close / close.shift(5) - 1
            features['price_momentum_10'] = close / close.shift(10) - 1
            features['price_momentum_20'] = close / close.shift(20) - 1
            
            # Moving averages and ratios
            features['sma_5'] = close.rolling(5).mean()
            features['sma_10'] = close.rolling(10).mean()
            features['sma_20'] = close.rolling(20).mean()
            features['sma_50'] = close.rolling(50).mean() if len(close) >= 50 else close.rolling(min(len(close), 20)).mean()
            
            features['ema_5'] = close.ewm(span=5).mean()
            features['ema_10'] = close.ewm(span=10).mean()
            features['ema_20'] = close.ewm(span=20).mean()
            
            # Price position relative to moving averages
            features['price_vs_sma_5'] = close / features['sma_5'] - 1
            features['price_vs_sma_10'] = close / features['sma_10'] - 1
            features['price_vs_sma_20'] = close / features['sma_20'] - 1
            features['price_vs_ema_5'] = close / features['ema_5'] - 1
            features['price_vs_ema_10'] = close / features['ema_10'] - 1
            
            # Moving average relationships
            features['sma_5_vs_20'] = features['sma_5'] / features['sma_20'] - 1
            features['sma_10_vs_20'] = features['sma_10'] / features['sma_20'] - 1
            features['ema_5_vs_20'] = features['ema_5'] / features['ema_20'] - 1
            
            # RSI variations
            features['rsi_14'] = self._calculate_rsi(close, 14)
            features['rsi_7'] = self._calculate_rsi(close, 7)
            features['rsi_21'] = self._calculate_rsi(close, 21)
            features['rsi_slope'] = features['rsi_14'].diff()
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9).mean()
            features['macd'] = macd_line
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_line - macd_signal
            features['macd_above_signal'] = (macd_line > macd_signal).astype(int)
            
            # Bollinger Bands
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_squeeze'] = features['bb_width'] < features['bb_width'].rolling(20).mean() * 0.8
            
            # Volatility features
            features['volatility_5'] = close.rolling(5).std()
            features['volatility_10'] = close.rolling(10).std()
            features['volatility_20'] = close.rolling(20).std()
            features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
            
            # ATR
            features['atr_14'] = self._calculate_atr(high, low, close, 14)
            features['atr_ratio'] = features['atr_14'] / close
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(high, low, close, 14)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
            features['stoch_position'] = stoch_k - stoch_d
            
            # Williams %R
            features['williams_r'] = ((high.rolling(14).max() - close) / 
                                    (high.rolling(14).max() - low.rolling(14).min())) * -100
            
            # CCI (Commodity Channel Index)
            features['cci'] = self._calculate_cci(high, low, close, 20)
            
            # Volume features (if available)
            if volume.sum() > 0:
                features['volume_sma_10'] = volume.rolling(10).mean()
                features['volume_ratio'] = volume / features['volume_sma_10']
                features['price_volume'] = close * volume
                features['vwap'] = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
                features['price_vs_vwap'] = close / features['vwap'] - 1
                
                # On-Balance Volume
                features['obv'] = (volume * ((close > close.shift(1)).astype(int) * 2 - 1)).cumsum()
                features['obv_slope'] = features['obv'].diff()
            else:
                # Default volume features
                features['volume_ratio'] = 1.0
                features['price_vs_vwap'] = 0.0
                features['obv_slope'] = 0.0
            
            # Support/Resistance levels
            features['high_20'] = high.rolling(20).max()
            features['low_20'] = low.rolling(20).min()
            features['price_vs_high_20'] = close / features['high_20'] - 1
            features['price_vs_low_20'] = close / features['low_20'] - 1
            
            # Trend strength
            features['adx'] = self._calculate_adx(high, low, close, 14)
            features['trend_strength'] = np.where(features['adx'] > 25, 1, 0)
            
            # Market structure
            features['higher_high'] = (high > high.shift(1)).rolling(5).sum()
            features['lower_low'] = (low < low.shift(1)).rolling(5).sum()
            features['market_structure'] = features['higher_high'] - features['lower_low']
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'rsi_lag_{lag}'] = features['rsi_14'].shift(lag)
                features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)
            
            # Cross-sectional features
            features['returns_rank'] = features['returns'].rolling(20).rank(pct=True)
            features['volume_rank'] = features['volume_ratio'].rolling(20).rank(pct=True)
            features['rsi_rank'] = features['rsi_14'].rolling(20).rank(pct=True)
            
            # Remove infinite and NaN values
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)
            # Only set feature_names here if model not yet trained to avoid overwriting trained schema (which may include microstructure columns)
            if not self.is_trained:
                self.feature_names = list(features.columns)
            logger.info(f"Calculated {len(features.columns)} advanced features")
            return features
        except Exception as e:
            logger.error(f"Error calculating advanced features: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _calculate_stochastic(self, high, low, close, window=14):
        """Calculate Stochastic oscillator"""
        lowest_low = low.rolling(window).min()
        highest_high = high.rolling(window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(3).mean()
        return k_percent, d_percent

    def _calculate_cci(self, high, low, close, window=20):
        """Calculate Commodity Channel Index with manual MAD (pandas Series.mad removed in newer versions).
        CCI = (TP - SMA(TP)) / (0.015 * MAD), where TP=(H+L+C)/3 and MAD is mean absolute deviation.
        """
        try:
            tp = (high + low + close) / 3.0
            sma = tp.rolling(window).mean()
            # Manual mean absolute deviation over rolling window
            mad = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
            # Prevent division by zero
            mad = mad.replace(0, np.nan)
            cci = (tp - sma) / (0.015 * mad)
            return cci
        except Exception as e:
            logger.error(f"CCI calculation failed: {e}")
            return pd.Series(index=close.index, dtype=float)

    def _calculate_adx(self, high, low, close, window=14):
        """Calculate Average Directional Index"""
        # Simplified ADX calculation
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window).mean()
        
        return adx

    def _load_latest_microstructure(self, symbol: str, ref_ts: pd.Timestamp) -> dict:
        """Load latest microstructure features for symbol (<= ref_ts). Returns dict with ob_ prefixed keys.
        Falls back to empty dict if file or row missing."""
        try:
            data_dir = os.environ.get('OB_DATA_DIR', 'data/orderbook')
            primary_symbol = os.environ.get('FUTURES_SYMBOLS', symbol).split(',')[0].strip().upper()
            path = os.path.join(data_dir, f"{primary_symbol}_5m_features.csv")
            if not os.path.exists(path):
                return {}
            # Read only needed columns (performance: read entire small file assumed). Parse timestamp.
            ob = pd.read_csv(path)
            if 'timestamp' not in ob.columns:
                return {}
            ob['timestamp'] = pd.to_datetime(ob['timestamp'], utc=True, errors='coerce')
            ob = ob.dropna(subset=['timestamp']).sort_values('timestamp')
            # Select last row up to ref_ts (ensure ref_ts is UTC)
            if ref_ts.tz is None:
                ref_ts = ref_ts.tz_localize('UTC')
            row = ob[ob['timestamp'] <= ref_ts].tail(1)
            if row.empty:
                # maybe ahead of microstructure generation; take last available
                row = ob.tail(1)
            row = row.drop(columns=['timestamp'])
            # Prefix columns to match training merge convention
            row_prefixed = row.add_prefix('ob_')
            return row_prefixed.iloc[0].to_dict()
        except Exception as e:
            logger.debug(f"Live microstructure load failed: {e}")
            return {}

    def generate_enhanced_signal(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate enhanced ML signal with confidence and analysis. Auto-train if needed."""
        import subprocess
        try:
            if not self.is_trained:
                logger.warning("Models not trained. Attempting auto-training...")
                try:
                    # Attempt to train the model by running the training script
                    result = subprocess.run([
                        'python', 'train_ml_model.py'
                    ], capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        logger.info("Auto-training complete. Reloading model...")
                        self.load_model()
                    else:
                        logger.error(f"Auto-training failed: {result.stderr}")
                except Exception as train_exc:
                    logger.error(f"Auto-training exception: {train_exc}")
                if not self.is_trained:
                    logger.error("Model still not trained after auto-training. Returning neutral signal.")
                    return {
                        'enhanced_signal': 0,
                        'confidence': 0.3,
                        'probabilities': {'buy': 0.33, 'hold': 0.34, 'sell': 0.33},
                        'reasoning': 'Models not trained - using neutral signal',
                        'feature_analysis': {},
                        'model_votes': {},
                        'risk_score': 0.5,
                        'market_regime': 'unknown'
                    }
            # Calculate features
            features = self.calculate_advanced_features(price_data)
            if features.empty:
                logger.warning("Feature calculation failed")
                return self._default_signal("Feature calculation failed")
            expected_cols = self.feature_names if self.feature_names else list(features.columns)
            latest = features.iloc[-1:].copy()
            # Inject live microstructure values if expected ob_ columns present
            if any(col.startswith('ob_') for col in expected_cols):
                live_micro = self._load_latest_microstructure(symbol, latest.index[-1])
                if live_micro:
                    for k, v in live_micro.items():
                        latest[k] = v
                    logger.debug(f"Injected {len(live_micro)} live microstructure fields")
                else:
                    logger.debug("No live microstructure data available (using zeros later if needed)")
            # Add missing expected columns with 0.0
            missing = [c for c in expected_cols if c not in latest.columns]
            if missing:
                for c in missing:
                    latest[c] = 0.0
                logger.debug(f"Added {len(missing)} missing feature columns (filled with 0)")
            # Drop unexpected extra columns
            extra = [c for c in latest.columns if c not in expected_cols]
            if extra:
                latest.drop(columns=extra, inplace=True)
                logger.debug(f"Dropped {len(extra)} unexpected extra feature columns")
            latest = latest[expected_cols]
            if latest.isna().any().any():
                latest = latest.fillna(0)
            latest_features = latest
            # Get predictions from all models
            model_predictions = {}
            model_probabilities = {}
            for model_name, model in self.models.items():
                try:
                    X_scaled = self.scalers[model_name].transform(latest_features.values)
                    # Get prediction
                    prediction = model.predict(X_scaled)[0]
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_scaled)[0]
                        if len(proba) == 3:  # Three classes: sell(-1), hold(0), buy(1)
                            model_probabilities[model_name] = {
                                'sell': proba[0],
                                'hold': proba[1],
                                'buy': proba[2]
                            }
                        else:
                            # Binary classification
                            model_probabilities[model_name] = {
                                'sell': proba[0] if prediction == -1 else 1-proba[1],
                                'hold': 0.1,
                                'buy': proba[1] if prediction == 1 else 1-proba[0]
                            }
                    else:
                        # For models without predict_proba
                        confidence = 0.6  # Default confidence
                        if prediction == 1:
                            model_probabilities[model_name] = {'buy': confidence, 'hold': (1-confidence)/2, 'sell': (1-confidence)/2}
                        elif prediction == -1:
                            model_probabilities[model_name] = {'sell': confidence, 'hold': (1-confidence)/2, 'buy': (1-confidence)/2}
                        else:
                            model_probabilities[model_name] = {'hold': confidence, 'buy': (1-confidence)/2, 'sell': (1-confidence)/2}
                    model_predictions[model_name] = prediction
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
                    model_predictions[model_name] = 0
                    model_probabilities[model_name] = {'buy': 0.33, 'hold': 0.34, 'sell': 0.33}
            # Ensemble prediction using weighted voting
            ensemble_probabilities = {'buy': 0.0, 'hold': 0.0, 'sell': 0.0}
            total_weight = 0.0
            for model_name, proba in model_probabilities.items():
                weight = self.model_weights.get(model_name, 0.25)
                total_weight += weight
                for action, prob in proba.items():
                    ensemble_probabilities[action] += prob * weight
            # Normalize probabilities
            if total_weight > 0:
                for action in ensemble_probabilities:
                    ensemble_probabilities[action] /= total_weight
            # Determine final signal
            max_prob_action = max(ensemble_probabilities, key=ensemble_probabilities.get)
            max_prob = ensemble_probabilities[max_prob_action]
            # Convert to numeric signal
            if max_prob_action == 'buy':
                signal = 1
            elif max_prob_action == 'sell':
                signal = -1
            else:
                signal = 0
            
            # Calculate confidence
            confidence = max_prob
            
            # Enhance confidence calculation
            prob_spread = max(ensemble_probabilities.values()) - min(ensemble_probabilities.values())
            adjusted_confidence = min(confidence + (prob_spread * 0.2), 1.0)
            
            # Feature analysis for reasoning
            feature_analysis = self._analyze_key_features(latest_features.iloc[0])
            
            # Generate reasoning
            reasoning = self._generate_reasoning(signal, adjusted_confidence, feature_analysis, model_predictions)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(latest_features.iloc[0], ensemble_probabilities)
            
            # Determine market regime
            market_regime = self._determine_market_regime(latest_features.iloc[0])
            
            result = {
                'enhanced_signal': signal,
                'confidence': adjusted_confidence,
                'probabilities': ensemble_probabilities,
                'reasoning': reasoning,
                'feature_analysis': feature_analysis,
                'model_votes': model_predictions,
                'risk_score': risk_score,
                'market_regime': market_regime,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'risk_factors': self._identify_risk_factors(latest_features.iloc[0])
            }
            
            logger.info(f"Enhanced ML signal for {symbol}: {signal} (confidence: {adjusted_confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced signal generation failed: {e}")
            return self._default_signal(f"Error: {str(e)}")

    def _default_signal(self, reason: str) -> Dict[str, Any]:
        """Return default neutral signal"""
        return {
            'enhanced_signal': 0,
            'confidence': 0.3,
            'probabilities': {'buy': 0.33, 'hold': 0.34, 'sell': 0.33},
            'reasoning': reason,
            'feature_analysis': {},
            'model_votes': {},
            'risk_score': 0.5,
            'market_regime': 'unknown',
            'risk_factors': []
        }

    def _analyze_key_features(self, features: pd.Series) -> Dict[str, Any]:
        """Analyze key features for reasoning"""
        analysis = {}
        
        try:
            # RSI analysis
            if 'rsi_14' in features:
                rsi = features['rsi_14']
                if rsi > 70:
                    analysis['rsi'] = 'Overbought (>70)'
                elif rsi < 30:
                    analysis['rsi'] = 'Oversold (<30)'
                else:
                    analysis['rsi'] = f'Neutral ({rsi:.1f})'
            
            # Trend analysis
            if 'price_vs_sma_20' in features:
                trend = features['price_vs_sma_20']
                if trend > 0.02:
                    analysis['trend'] = 'Strong uptrend'
                elif trend > 0:
                    analysis['trend'] = 'Weak uptrend'
                elif trend < -0.02:
                    analysis['trend'] = 'Strong downtrend'
                else:
                    analysis['trend'] = 'Sideways'
            
            # Volatility analysis
            if 'volatility_ratio' in features:
                vol_ratio = features['volatility_ratio']
                if vol_ratio > 1.5:
                    analysis['volatility'] = 'High volatility'
                elif vol_ratio < 0.7:
                    analysis['volatility'] = 'Low volatility'
                else:
                    analysis['volatility'] = 'Normal volatility'
            
        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
        
        return analysis

    def _generate_reasoning(self, signal: int, confidence: float, 
                          feature_analysis: Dict, model_votes: Dict) -> str:
        """Generate human-readable reasoning for the signal"""
        try:
            signal_text = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}[signal]
            
            reasoning_parts = [f"Signal: {signal_text} (confidence: {confidence:.1%})"]
            
            # Model consensus
            votes = list(model_votes.values())
            if votes:
                buy_votes = sum(1 for v in votes if v == 1)
                sell_votes = sum(1 for v in votes if v == -1)
                hold_votes = sum(1 for v in votes if v == 0)
                
                reasoning_parts.append(f"Model votes - Buy: {buy_votes}, Hold: {hold_votes}, Sell: {sell_votes}")
            
            # Key features
            if feature_analysis:
                key_factors = []
                for factor, description in feature_analysis.items():
                    key_factors.append(f"{factor}: {description}")
                if key_factors:
                    reasoning_parts.append("Key factors: " + "; ".join(key_factors[:3]))  # Top 3
            
            return ". ".join(reasoning_parts)
            
        except Exception as e:
            return f"Signal: {signal} with {confidence:.1%} confidence"

    def _calculate_risk_score(self, features: pd.Series, probabilities: Dict) -> float:
        """Calculate risk score for the position"""
        try:
            risk_factors = []
            
            # Volatility risk
            if 'volatility_ratio' in features:
                vol_risk = min(features['volatility_ratio'] / 2.0, 1.0)
                risk_factors.append(vol_risk)
            
            # RSI extreme risk
            if 'rsi_14' in features:
                rsi = features['rsi_14']
                if rsi > 80 or rsi < 20:
                    risk_factors.append(0.8)
                elif rsi > 70 or rsi < 30:
                    risk_factors.append(0.6)
                else:
                    risk_factors.append(0.3)
            
            # Probability uncertainty
            prob_uncertainty = 1 - max(probabilities.values())
            risk_factors.append(prob_uncertainty)
            
            # Average risk score
            if risk_factors:
                return sum(risk_factors) / len(risk_factors)
            else:
                return 0.5
                
        except Exception:
            return 0.5

    def _determine_market_regime(self, features: pd.Series) -> str:
        """Determine current market regime"""
        try:
            # Trend strength
            trend_score = 0
            if 'price_vs_sma_20' in features:
                trend_score += 1 if features['price_vs_sma_20'] > 0.02 else -1 if features['price_vs_sma_20'] < -0.02 else 0
            
            # Volatility
            vol_score = 0
            if 'volatility_ratio' in features:
                vol_score = 1 if features['volatility_ratio'] > 1.5 else 0
            
            # Momentum
            momentum_score = 0
            if 'price_momentum_10' in features:
                momentum_score = 1 if features['price_momentum_10'] > 0.05 else -1 if features['price_momentum_10'] < -0.05 else 0
            
            # Determine regime
            if trend_score > 0 and momentum_score > 0:
                return 'bull_trending'
            elif trend_score < 0 and momentum_score < 0:
                return 'bear_trending'
            elif vol_score > 0:
                return 'high_volatility'
            else:
                return 'sideways'
                
        except Exception:
            return 'unknown'

    def _identify_risk_factors(self, features: pd.Series) -> List[str]:
        """Identify current risk factors"""
        risk_factors = []
        
        try:
            # High volatility
            if 'volatility_ratio' in features and features['volatility_ratio'] > 2:
                risk_factors.append('High volatility detected')
            
            # Extreme RSI
            if 'rsi_14' in features:
                rsi = features['rsi_14']
                if rsi > 80:
                    risk_factors.append('Extremely overbought (RSI > 80)')
                elif rsi < 20:
                    risk_factors.append('Extremely oversold (RSI < 20)')
            
            # Low volume
            if 'volume_ratio' in features and features['volume_ratio'] < 0.5:
                risk_factors.append('Low trading volume')
            
            # High ATR
            if 'atr_ratio' in features and features['atr_ratio'] > 0.05:
                risk_factors.append('High ATR indicates increased risk')
            
        except Exception as e:
            logger.error(f"Risk factor identification failed: {e}")
        
        return risk_factors

    def load_model(self):
        """Load a trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                
                # Handle both old and new model formats
                if 'models' in model_data:
                    self.models = model_data['models']
                    self.scalers = model_data.get('scalers', {})
                    self.feature_names = model_data.get('feature_names', [])
                    self.model_weights = model_data.get('model_weights', {})
                    self.feature_importance = model_data.get('feature_importance', {})
                    self.is_trained = model_data.get('is_trained', False)
                else:
                    # Old format compatibility
                    self.models['random_forest'] = model_data.get('model', RandomForestClassifier())
                    self.scalers['random_forest'] = model_data.get('scaler', StandardScaler())
                    self.feature_names = model_data.get('feature_names', [])
                    self.is_trained = True
                
                logger.info(f"ML model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

    def generate_ml_signal(self, symbol: str, price_data: Optional[pd.DataFrame] = None) -> int:
        """Generate ML-based trading signal (backwards compatibility)"""
        if price_data is None or price_data.empty:
            return 1  # Default buy signal for backwards compatibility
        
        result = self.generate_enhanced_signal(symbol, price_data)
        return result.get('enhanced_signal', 1)


# Global instance for backwards compatibility
_ml_generator = EnhancedMLSignalGenerator()

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

def generate_enhanced_ml_signal(symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate enhanced ML signal with comprehensive analysis.
    """
    return _ml_generator.generate_enhanced_signal(symbol, price_data)
