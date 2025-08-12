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
    def __init__(self, model_path: str = "ml_model.pkl"):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = model_path
        self.feature_names = []
        self.openai = OpenAIClient()
        
        # Try to load existing model
        self.load_model()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for ML features"""
        if len(df) < 20:
            return pd.DataFrame()
            
        close = df['close'] if 'close' in df.columns else pd.Series([0])
        high = df['high'] if 'high' in df.columns else close
        low = df['low'] if 'low' in df.columns else close
        volume = df['volume'] if 'volume' in df.columns else pd.Series([1] * len(close))
        
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = close.pct_change()
        features['returns_2d'] = close.pct_change(2)
        features['returns_5d'] = close.pct_change(5)
        
        # Moving averages
        features['sma_10'] = close.rolling(10).mean()
        features['sma_20'] = close.rolling(20).mean()
        features['sma_50'] = close.rolling(50, min_periods=20).mean()
        features['price_vs_sma_10'] = close / features['sma_10'] - 1
        features['price_vs_sma_20'] = close / features['sma_20'] - 1
        features['price_vs_sma_50'] = close / features['sma_50'] - 1
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        
        # MACD
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_bullish'] = (features['macd'] > features['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_upper'] = bb_sma + (bb_std * 2)
        features['bb_lower'] = bb_sma - (bb_std * 2)
        features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_squeeze'] = (features['bb_upper'] - features['bb_lower']) / bb_sma
        
        # Volatility
        features['volatility'] = close.rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50, min_periods=20).mean()
        
        # Volume features
        features['volume_sma'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_sma']
        features['price_volume'] = close * volume
        
        # High/Low features
        features['high_low_ratio'] = high / low
        features['close_position'] = (close - low) / (high - low)
        
        return features.fillna(method='ffill').fillna(0)
    
    def calculate_market_context(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate market context for AI analysis"""
        if len(df) < 20:
            return {}
        
        close = df['close'] if 'close' in df.columns else pd.Series([0])
        volume = df['volume'] if 'volume' in df.columns else pd.Series([1] * len(close))
        
        current_price = float(close.iloc[-1])
        
        # Price trends
        sma_10 = close.rolling(10).mean().iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50, min_periods=20).mean().iloc[-1]
        
        # Volatility
        volatility_20d = close.rolling(20).std().iloc[-1]
        avg_volatility = close.rolling(50, min_periods=20).std().mean()
        
        # Volume analysis
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = float(volume.iloc[-1])
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # MACD
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9).mean()
        macd_current = float(macd.iloc[-1])
        macd_signal_current = float(macd_signal.iloc[-1])
        
        # Support/Resistance levels
        recent_highs = close.rolling(20).max()
        recent_lows = close.rolling(20).min()
        resistance = float(recent_highs.iloc[-1])
        support = float(recent_lows.iloc[-1])
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "price_trends": {
                "sma_10": float(sma_10),
                "sma_20": float(sma_20),
                "sma_50": float(sma_50),
                "price_vs_sma_10": (current_price / sma_10 - 1) * 100,
                "price_vs_sma_20": (current_price / sma_20 - 1) * 100,
                "price_vs_sma_50": (current_price / sma_50 - 1) * 100 if not pd.isna(sma_50) else 0
            },
            "technical_indicators": {
                "rsi": float(rsi),
                "macd": macd_current,
                "macd_signal": macd_signal_current,
                "macd_histogram": macd_current - macd_signal_current
            },
            "volatility": {
                "current": float(volatility_20d),
                "average": float(avg_volatility),
                "ratio": float(volatility_20d / avg_volatility) if avg_volatility > 0 else 1.0
            },
            "volume": {
                "current": current_volume,
                "average": float(avg_volume),
                "ratio": current_volume / avg_volume if avg_volume > 0 else 1.0
            },
            "support_resistance": {
                "support": support,
                "resistance": resistance,
                "distance_to_support": (current_price - support) / support * 100,
                "distance_to_resistance": (resistance - current_price) / current_price * 100
            }
        }
    
    async def generate_enhanced_signal(self, symbol: str, price_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate AI-enhanced trading signal combining ML and OpenAI analysis"""
        try:
            if price_data is None or price_data.empty:
                logger.warning(f"No price data provided for AI-enhanced signal generation for {symbol}")
                return {
                    "enhanced_signal": 0,
                    "confidence": 0.0,
                    "ml_signal": 0,
                    "ai_enhancement": 0,
                    "reasoning": "No price data available",
                    "risk_factors": ["No data"],
                    "market_regime": "unknown"
                }
            
            # Get base ML signal
            base_ml_signal = self.generate_base_ml_signal(symbol, price_data)
            
            # Calculate market context
            market_context = self.calculate_market_context(price_data, symbol)
            
            if not market_context:
                return {
                    "enhanced_signal": base_ml_signal,
                    "confidence": 0.5,
                    "ml_signal": base_ml_signal,
                    "ai_enhancement": 0,
                    "reasoning": "Insufficient data for AI analysis",
                    "risk_factors": ["Limited data"],
                    "market_regime": "unknown"
                }
            
            # Get AI enhancement
            ai_analysis = await self.get_ai_analysis(base_ml_signal, market_context)
            
            # Combine signals
            enhanced_signal = self.combine_signals(base_ml_signal, ai_analysis)
            
            result = {
                "enhanced_signal": enhanced_signal["signal"],
                "confidence": enhanced_signal["confidence"],
                "ml_signal": base_ml_signal,
                "ai_enhancement": ai_analysis.get("signal_adjustment", 0),
                "reasoning": ai_analysis.get("reasoning", "AI analysis completed"),
                "risk_factors": ai_analysis.get("risk_factors", []),
                "market_regime": ai_analysis.get("market_regime", "unknown"),
                "timeframe_bias": ai_analysis.get("timeframe_bias", "medium"),
                "ai_confidence": ai_analysis.get("confidence", 0.5)
            }
            
            logger.info(f"AI-Enhanced signal for {symbol}: {enhanced_signal['signal']} (confidence: {enhanced_signal['confidence']:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error generating AI-enhanced signal for {symbol}: {e}")
            return {
                "enhanced_signal": 0,
                "confidence": 0.0,
                "ml_signal": 0,
                "ai_enhancement": 0,
                "reasoning": f"Error in analysis: {str(e)}",
                "risk_factors": ["Analysis error"],
                "market_regime": "unknown"
            }
    
    def generate_base_ml_signal(self, symbol: str, price_data: pd.DataFrame) -> int:
        """Generate base ML signal using the existing RandomForest model"""
        if not self.is_trained:
            logger.warning("ML model not trained. Returning neutral signal.")
            return 0
        
        try:
            features = self.calculate_technical_indicators(price_data)
            
            if len(features) == 0 or features.iloc[-1].isna().any():
                return 0
            
            # Use the most recent features
            latest_features = features.iloc[-1:][self.feature_names] if self.feature_names else features.iloc[-1:]
            
            # Scale features
            X_scaled = self.scaler.transform(latest_features)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            
            return int(prediction)
            
        except Exception as e:
            logger.error(f"Error generating base ML signal: {e}")
            return 0
    
    async def get_ai_analysis(self, base_ml_signal: int, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI analysis to enhance ML signal"""
        try:
            prompt = f"""
            Enhance ML trading signal for {market_context['symbol']} with advanced market analysis:
            
            Base ML Signal: {base_ml_signal} (-1=sell, 0=hold, 1=buy)
            
            Market Data:
            - Current Price: ${market_context['current_price']:.4f}
            - RSI: {market_context['technical_indicators']['rsi']:.2f}
            - MACD: {market_context['technical_indicators']['macd']:.4f}
            - MACD Signal: {market_context['technical_indicators']['macd_signal']:.4f}
            - MACD Histogram: {market_context['technical_indicators']['macd_histogram']:.4f}
            
            Price Trends:
            - vs SMA10: {market_context['price_trends']['price_vs_sma_10']:.2f}%
            - vs SMA20: {market_context['price_trends']['price_vs_sma_20']:.2f}%
            - vs SMA50: {market_context['price_trends']['price_vs_sma_50']:.2f}%
            
            Volatility:
            - Current vs Average: {market_context['volatility']['ratio']:.2f}x
            
            Volume:
            - Current vs Average: {market_context['volume']['ratio']:.2f}x
            
            Support/Resistance:
            - Distance to Support: {market_context['support_resistance']['distance_to_support']:.2f}%
            - Distance to Resistance: {market_context['support_resistance']['distance_to_resistance']:.2f}%
            
            Provide enhanced analysis considering:
            1. Market regime (bull/bear/sideways/volatile)
            2. Signal quality assessment
            3. Risk factors identification
            4. Confidence calibration
            5. Timeframe considerations
            
            Return JSON format:
            {{
                "signal_adjustment": -1|0|1,
                "confidence": 0.0-1.0,
                "ml_signal_quality": "high|medium|low",
                "market_regime": "bull_trending|bear_trending|sideways_range|high_volatility|low_volatility",
                "reasoning": "detailed explanation of analysis",
                "risk_factors": ["factor1", "factor2", "factor3"],
                "timeframe_bias": "short|medium|long",
                "key_observations": ["observation1", "observation2"]
            }}
            """
            
            ai_response = await self.openai.ask_json(prompt)
            
            if not ai_response:
                return {
                    "signal_adjustment": 0,
                    "confidence": 0.5,
                    "ml_signal_quality": "medium",
                    "market_regime": "unknown",
                    "reasoning": "AI analysis failed",
                    "risk_factors": ["AI unavailable"],
                    "timeframe_bias": "medium"
                }
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {
                "signal_adjustment": 0,
                "confidence": 0.3,
                "ml_signal_quality": "low",
                "market_regime": "unknown",
                "reasoning": f"AI analysis error: {str(e)}",
                "risk_factors": ["AI error"],
                "timeframe_bias": "medium"
            }
    
    def combine_signals(self, base_ml_signal: int, ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine ML signal with AI analysis for final enhanced signal"""
        try:
            ai_adjustment = ai_analysis.get("signal_adjustment", 0)
            ai_confidence = ai_analysis.get("confidence", 0.5)
            ml_quality = ai_analysis.get("ml_signal_quality", "medium")
            
            # Quality weights
            quality_weights = {
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            }
            
            ml_weight = quality_weights.get(ml_quality, 0.6)
            ai_weight = min(ai_confidence, 0.8)  # Cap AI weight at 0.8
            
            # Calculate weighted signal
            if base_ml_signal == 0 and ai_adjustment == 0:
                final_signal = 0
                final_confidence = max(ai_confidence, 0.3)
            elif base_ml_signal == ai_adjustment:
                # Signals agree - high confidence
                final_signal = base_ml_signal
                final_confidence = min((ml_weight + ai_weight) / 2 + 0.2, 1.0)
            elif abs(base_ml_signal - ai_adjustment) == 2:
                # Signals strongly disagree (buy vs sell)
                final_signal = 0  # Hold when conflicting
                final_confidence = 0.3
            else:
                # One signal is neutral, use the stronger one
                if base_ml_signal == 0:
                    final_signal = ai_adjustment
                    final_confidence = ai_confidence * 0.8
                elif ai_adjustment == 0:
                    final_signal = base_ml_signal
                    final_confidence = ml_weight * 0.8
                else:
                    # Partial agreement (e.g., 1 and 0, or -1 and 0)
                    final_signal = base_ml_signal if ml_weight > ai_weight else ai_adjustment
                    final_confidence = max(ml_weight, ai_weight) * 0.9
            
            return {
                "signal": int(final_signal),
                "confidence": float(final_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return {
                "signal": base_ml_signal,
                "confidence": 0.4
            }
    
    def load_model(self):
        """Load a trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = True
                logger.info(f"AI-Enhanced ML model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")


# Global instance for integration with existing system
_ai_enhanced_generator = AIEnhancedMLSignalGenerator()

async def generate_ai_enhanced_ml_signal(symbol: str, price_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Generate AI-enhanced ML trading signal.
    Returns comprehensive signal analysis with confidence scoring.
    """
    return await _ai_enhanced_generator.generate_enhanced_signal(symbol, price_data)

def generate_enhanced_ml_signal_sync(symbol: str, price_data: Optional[pd.DataFrame] = None) -> int:
    """
    Synchronous wrapper for backwards compatibility.
    Returns simple signal: -1, 0, or 1
    """
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_ai_enhanced_generator.generate_enhanced_signal(symbol, price_data))
        loop.close()
        
        return result.get("enhanced_signal", 0)
    except Exception as e:
        logger.error(f"Error in sync enhanced ML signal: {e}")
        return 0
