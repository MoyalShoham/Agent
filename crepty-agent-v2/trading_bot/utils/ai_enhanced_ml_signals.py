"""
AI-Enhanced ML Signals - Combining OpenAI intelligence with ML models for superior trading signals.
This module enhances the existing ML signal generator with AI-powered analysis and validation.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from loguru import logger
import joblib
import os
from typing import Optional, Dict, Any
import asyncio
from datetime import datetime

try:
    from .openai_client import OpenAIClient
except ImportError:
    from trading_bot.utils.openai_client import OpenAIClient


class AIEnhancedMLSignalGenerator:
    def __init__(self, model_path: str = "data/ml_model.pkl"):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = model_path
        self.feature_names = []
        self.openai = OpenAIClient()
        
        # Try to load existing model
        self.load_model()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for ML features - ALIGNED WITH TRAINED MODEL"""
        if len(df) < 20:
            return pd.DataFrame()
            
        close = df['close'] if 'close' in df.columns else pd.Series([0])
        
        features = pd.DataFrame(index=df.index)
        
        # ONLY the features the model was trained on (9 features total)
        # Price-based features
        features['returns'] = close.pct_change()
        
        # Moving averages  
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
        
        # Fill NaN values and return EXACTLY 9 features to match trained model
        return features.ffill().fillna(0)
    
    def load_model(self):
        """Load the ML model and scaler from the specified path.
        Supports legacy tuple (model, scaler), dict {'model':..., 'scaler':...}, or single estimator/pipeline.
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file {self.model_path} not found.")
                return
            logger.info(f"Loading model from {self.model_path}")
            obj = joblib.load(self.model_path)
            model = None
            scaler = None
            feature_names = None
            if isinstance(obj, tuple) and len(obj) == 2:
                model, scaler = obj
            elif isinstance(obj, dict):
                model = obj.get('model') or obj.get('estimator')
                scaler = obj.get('scaler') or obj.get('preprocessor')
                feature_names = obj.get('feature_names')
            else:
                # Single object (e.g., pipeline or estimator). Try to extract scaler if pipeline.
                model = obj
                try:
                    from sklearn.pipeline import Pipeline
                    if isinstance(model, Pipeline):
                        # Heuristic: locate StandardScaler step
                        for name, step in model.steps:
                            from sklearn.preprocessing import StandardScaler as _SS
                            if isinstance(step, _SS):
                                scaler = step
                                break
                except Exception:
                    pass
            if model is None:
                raise ValueError("Loaded artifact missing model")
            if scaler is None:
                logger.warning("Scaler missing in artifact; using un-fitted default (confidence may degrade)")
            self.model = model
            if scaler is not None:
                self.scaler = scaler
            if feature_names:
                self.feature_names = feature_names
            self.is_trained = True
            logger.success("Model loaded successfully (flex format).")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def save_model(self):
        """Save the ML model and scaler to the specified path."""
        try:
            with open(self.model_path, "wb") as f:
                joblib.dump((self.model, self.scaler), f)
            logger.success(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the ML model with the given features and target."""
        try:
            logger.info("Starting model training...")
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logger.success("Model trained successfully.")
        except Exception as e:
            logger.error(f"Error during training: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained ML model."""
        if not self.is_trained:
            logger.warning("Model is not trained. Please train the model before prediction.")
            return np.array([])
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return np.array([])
    
    async def get_ai_analysis(self, base_ml_signal: int, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI analysis to enhance ML signal.
        NOTE: openai_client methods are synchronous; offload to executor to avoid blocking.
        """
        try:
            prompt = (
                f"Enhance ML trading signal for {market_context['symbol']} with advanced market analysis.\n\n"
                f"Base ML Signal: {base_ml_signal} (-1=sell, 0=hold, 1=buy)\n\n"
                "Market Data:\n"
                f"- Current Price: ${market_context['current_price']:.4f}\n"
                f"- RSI: {market_context['technical_indicators']['rsi']:.2f}\n"
                f"- MACD: {market_context['technical_indicators']['macd']:.4f}\n"
                f"- MACD Signal: {market_context['technical_indicators']['macd_signal']:.4f}\n"
                f"- MACD Histogram: {market_context['technical_indicators']['macd_histogram']:.4f}\n\n"
                "Price Trends (% vs SMA):\n"
                f"- vs SMA10: {market_context['price_trends']['price_vs_sma_10']:.2f}%\n"
                f"- vs SMA20: {market_context['price_trends']['price_vs_sma_20']:.2f}%\n"
                f"- vs SMA50: {market_context['price_trends']['price_vs_sma_50']:.2f}%\n\n"
                "Volatility / Volume:\n"
                f"- Volatility ratio: {market_context['volatility']['ratio']:.2f}x\n"
                f"- Volume ratio: {market_context['volume']['ratio']:.2f}x\n\n"
                "Support / Resistance distances (%):\n"
                f"- To Support: {market_context['support_resistance']['distance_to_support']:.2f}%\n"
                f"- To Resistance: {market_context['support_resistance']['distance_to_resistance']:.2f}%\n\n"
                "Analyze and adjust the base signal considering: market regime, signal quality, risk factors, confidence calibration, timeframe bias.\n"
                "Return ONLY a valid minified JSON object matching this schema (no commentary, no markdown).\n"
                "Schema example (values illustrative):\n"
                "{{\n"
                "  \"signal_adjustment\": 0,\n"
                "  \"confidence\": 0.55,\n"
                "  \"ml_signal_quality\": \"medium\",\n"
                "  \"market_regime\": \"sideways_range\",\n"
                "  \"reasoning\": \"Concise reasoning...\",\n"
                "  \"risk_factors\": [\"low_volume\"],\n"
                "  \"timeframe_bias\": \"medium\",\n"
                "  \"key_observations\": [\"Observation 1\", \"Observation 2\"]\n"
                "}}\n"
                "Field rules: signal_adjustment in [-1,0,1]; confidence 0-1 float."
            )
            loop = asyncio.get_running_loop()
            ai_response = await loop.run_in_executor(None, self.openai.ask_json, prompt)
            if not ai_response or not isinstance(ai_response, dict):
                return {"signal_adjustment": 0, "confidence": 0.5, "ml_signal_quality": "medium", "market_regime": "unknown", "reasoning": "AI analysis failed", "risk_factors": ["AI unavailable"], "timeframe_bias": "medium"}
            return ai_response
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {"signal_adjustment": 0, "confidence": 0.3, "ml_signal_quality": "low", "market_regime": "unknown", "reasoning": f"AI analysis error: {str(e)}", "risk_factors": ["AI error"], "timeframe_bias": "medium"}

# --- Global generator instance and async helper ---
_signal_generator: Optional[AIEnhancedMLSignalGenerator] = None

async def generate_ai_enhanced_ml_signal(symbol: str, price_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
    """Async helper used by ManagerAgent and others to produce AI-enhanced ML signal.
    Accepts price_data (preferred) but also tolerates legacy param names (ohlcv_data, df) via **kwargs.
    Returns dict with keys: enhanced_signal (-1/0/1), confidence (0-1), base_ml_signal, ai_adjustment,
    market_regime, risk_factors, reasoning.
    Added passthrough: if model untrained and caller supplies fallback_signal in kwargs, use it.
    """
    # Passthrough support: external consensus / fallback signal
    fallback_signal = int(kwargs.get('fallback_signal', 0)) if 'fallback_signal' in kwargs else 0

    global _signal_generator
    if _signal_generator is None:
        try:
            _signal_generator = AIEnhancedMLSignalGenerator(model_path=os.getenv('AI_ML_MODEL_PATH', 'data/ml_model.pkl'))
        except Exception as e:
            logger.error(f"[AI_ENHANCED_INIT] Failed to init generator: {e}")
            return {"enhanced_signal": 0, "confidence": 0.3, "reasoning": "init failure", "risk_factors": ["init_error"]}

    # Accept legacy aliases
    if price_data is None:
        price_data = kwargs.get('ohlcv_data') or kwargs.get('df') or kwargs.get('data')

    try:
        if price_data is None or price_data.empty:
            return {"enhanced_signal": 0, "confidence": 0.3, "reasoning": "no data", "risk_factors": ["no_price_data"]}

        # Ensure DataFrame has needed columns
        for col in ('close',):
            if col not in price_data.columns:
                raise ValueError(f"price_data missing required column: {col}")

        # Build features
        features_df = _signal_generator.calculate_technical_indicators(price_data)
        if features_df.empty:
            return {"enhanced_signal": fallback_signal, "confidence": 0.3, "reasoning": "insufficient bars", "risk_factors": ["short_window"]}

        latest_features = features_df.iloc[[-1]].copy()
        base_ml_signal = 0
        ml_confidence = 0.3
        try:
            if _signal_generator.is_trained:
                # Attempt probability-based confidence
                if hasattr(_signal_generator.model, 'predict_proba'):
                    proba = _signal_generator.model.predict_proba(_signal_generator.scaler.transform(latest_features))[0]
                    pred_class = _signal_generator.model.predict(_signal_generator.scaler.transform(latest_features))[0]
                    base_ml_signal = int(pred_class) if isinstance(pred_class, (int, np.integer)) else 0
                    ml_confidence = float(np.max(proba)) if isinstance(proba, (list, np.ndarray)) else 0.5
                else:
                    pred_class = _signal_generator.model.predict(_signal_generator.scaler.transform(latest_features))[0]
                    base_ml_signal = int(pred_class) if isinstance(pred_class, (int, np.integer)) else 0
                    ml_confidence = 0.5
            else:
                # Untrained: passthrough external fallback if provided
                base_ml_signal = fallback_signal
                ml_confidence = 0.35 if base_ml_signal == 0 else 0.45
                logger.debug("[AI_ENHANCED] Model not trained; passthrough fallback signal=%s", base_ml_signal)
        except Exception as mle:
            logger.error(f"[AI_ENHANCED] ML prediction error: {mle}")

        # Construct market context for AI layer
        try:
            row = latest_features.iloc[0]
            market_context = {
                'symbol': symbol,
                'current_price': float(price_data['close'].iloc[-1]),
                'technical_indicators': {
                    'rsi': float(row.get('rsi', 50) or 50),
                    'macd': float(row.get('macd', 0) or 0),
                    'macd_signal': float(row.get('macd_signal', 0) or 0),
                    'macd_histogram': float(row.get('macd_histogram', 0) or 0),
                },
                'price_trends': {
                    'price_vs_sma_10': float(row.get('price_vs_sma_10', 0) * 100),
                    'price_vs_sma_20': float(row.get('price_vs_sma_20', 0) * 100),
                    'price_vs_sma_50': float(row.get('price_vs_sma_50', 0) * 100),
                },
                'volatility': {
                    'ratio': float(row.get('volatility_ratio', 1) or 1),
                },
                'volume': {
                    'ratio': float(row.get('volume_ratio', 1) or 1),
                },
                'support_resistance': {
                    'distance_to_support': float(row.get('close_position', 0) * 100),
                    'distance_to_resistance': float((1 - row.get('close_position', 0)) * 100),
                }
            }
        except Exception as ctx_e:
            logger.error(f"[AI_ENHANCED] Context build error: {ctx_e}")
            market_context = {'symbol': symbol, 'current_price': float(price_data['close'].iloc[-1])}

        # AI analysis
        ai_adjustment = 0
        ai_confidence = ml_confidence
        reasoning = ''
        risk_factors = []
        try:
            ai_resp = await _signal_generator.get_ai_analysis(base_ml_signal, market_context)
            ai_adjustment = int(ai_resp.get('signal_adjustment', 0) or 0)
            ai_confidence = float(ai_resp.get('confidence', ml_confidence) or ml_confidence)
            reasoning = ai_resp.get('reasoning', '')
            risk_factors = ai_resp.get('risk_factors', []) or []
        except Exception as ai_e:
            logger.error(f"[AI_ENHANCED] AI layer error: {ai_e}")

        # Combine signals (simple additive then clamp)
        enhanced_signal = base_ml_signal + ai_adjustment
        if enhanced_signal > 1:
            enhanced_signal = 1
        if enhanced_signal < -1:
            enhanced_signal = -1
        if not _signal_generator.is_trained and ai_adjustment == 0 and base_ml_signal == fallback_signal:
            # Slightly reduce confidence if pure passthrough
            ml_confidence = min(ml_confidence, 0.4)
        return {
            'symbol': symbol,
            'enhanced_signal': int(enhanced_signal),
            'base_ml_signal': int(base_ml_signal),
            'ai_adjustment': int(ai_adjustment),
            'confidence': float(max(0.0, min(1.0, ai_confidence if _signal_generator.is_trained else ml_confidence))),
            'reasoning': reasoning or ('passthrough (untrained)' if not _signal_generator.is_trained else 'combined ml + ai'),
            'risk_factors': risk_factors if _signal_generator.is_trained else (risk_factors + ['untrained_model']),
            'market_regime': market_context.get('market_regime', 'unknown')
        }
    except Exception as e:
        logger.error(f"[AI_ENHANCED] Fatal error: {e}")
        return {"enhanced_signal": fallback_signal, "confidence": 0.3, "reasoning": f"fatal error: {e}", "risk_factors": ["fatal"]}
