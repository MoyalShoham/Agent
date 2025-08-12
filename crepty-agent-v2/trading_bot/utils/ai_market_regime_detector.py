"""
AI Market Regime Detector - Enhanced market state analysis for adaptive trading
Detects bull/bear/sideways/volatile regimes and adjusts strategy accordingly
"""
from typing import Dict, Any, Optional, List
from loguru import logger
import pandas as pd
import numpy as np
import os
import asyncio

try:
    from .openai_client import OpenAIClient
except ImportError:
    from trading_bot.utils.openai_client import OpenAIClient


class AIMarketRegimeDetector:
    def __init__(self):
        self.openai = OpenAIClient()
        self.regime_history = []
        self.confidence_threshold = float(os.getenv('AI_CONFIDENCE_THRESHOLD', '0.75'))
    
    async def detect_market_regime(
        self,
        price_data: pd.DataFrame,
        symbol: str = "MARKET",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect current market regime using AI analysis
        
        Returns:
        - regime: bull_trending|bear_trending|sideways_range|high_volatility|low_volatility
        - confidence: 0.0-1.0
        - regime_strength: 0.0-1.0
        - duration_estimate: days
        - trading_strategy: recommended approach
        - position_sizing_factor: 0.3-1.5 multiplier
        """
        
        try:
            if price_data.empty or len(price_data) < 20:
                return self._get_default_regime()
            
            # Calculate technical regime indicators
            regime_indicators = self._calculate_regime_indicators(price_data)
            
            # Get AI regime analysis
            ai_analysis = await self._get_ai_regime_analysis(regime_indicators, symbol, additional_context)
            
            # Combine with rule-based detection
            final_regime = self._combine_regime_analysis(regime_indicators, ai_analysis)
            
            # Update regime history
            self._update_regime_history(final_regime)
            
            # Add trading recommendations
            final_regime['trading_recommendations'] = self._get_trading_recommendations(final_regime)
            
            logger.info(f"Market Regime Detection for {symbol}: {final_regime['regime']} (confidence: {final_regime['confidence']:.2f})")
            
            return final_regime
            
        except Exception as e:
            logger.error(f"Error in market regime detection: {e}")
            return self._get_default_regime()
    
    def _calculate_regime_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for regime detection"""
        
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        
        # Trend indicators
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50, min_periods=20).mean()
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        
        # Volatility indicators
        returns = close.pct_change()
        volatility_20 = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        atr = ((high - low).rolling(14).mean()) / close.rolling(14).mean()
        
        # Momentum indicators
        rsi = self._calculate_rsi(close, 14)
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        # Volume indicators
        volume_sma = volume.rolling(20).mean()
        volume_ratio = volume / volume_sma
        
        # Support/Resistance
        resistance_20 = high.rolling(20).max()
        support_20 = low.rolling(20).min()
        price_position = (close - support_20) / (resistance_20 - support_20)
        
        # Current values
        current_price = float(close.iloc[-1])
        current_sma_20 = float(sma_20.iloc[-1])
        current_sma_50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else current_sma_20
        
        return {
            'price_trend': {
                'current_price': current_price,
                'sma_20': current_sma_20,
                'sma_50': current_sma_50,
                'price_vs_sma_20': (current_price / current_sma_20 - 1) * 100,
                'price_vs_sma_50': (current_price / current_sma_50 - 1) * 100,
                'sma_20_vs_50': (current_sma_20 / current_sma_50 - 1) * 100,
                'trend_direction': 'up' if current_price > current_sma_20 > current_sma_50 else 
                                 'down' if current_price < current_sma_20 < current_sma_50 else 'sideways'
            },
            'volatility': {
                'current_volatility': float(volatility_20.iloc[-1]),
                'avg_volatility': float(volatility_20.mean()),
                'volatility_percentile': float(volatility_20.rank(pct=True).iloc[-1] * 100),
                'atr_current': float(atr.iloc[-1]),
                'atr_avg': float(atr.mean()),
                'volatility_regime': 'high' if volatility_20.iloc[-1] > volatility_20.mean() * 1.5 else
                                   'low' if volatility_20.iloc[-1] < volatility_20.mean() * 0.7 else 'normal'
            },
            'momentum': {
                'rsi': float(rsi.iloc[-1]),
                'macd': float(macd.iloc[-1]),
                'macd_signal': float(macd_signal.iloc[-1]),
                'macd_histogram': float(macd.iloc[-1] - macd_signal.iloc[-1]),
                'momentum_direction': 'bullish' if rsi.iloc[-1] > 50 and macd.iloc[-1] > macd_signal.iloc[-1] else
                                    'bearish' if rsi.iloc[-1] < 50 and macd.iloc[-1] < macd_signal.iloc[-1] else 'neutral'
            },
            'volume': {
                'volume_trend': 'increasing' if volume_ratio.iloc[-5:].mean() > 1.2 else
                              'decreasing' if volume_ratio.iloc[-5:].mean() < 0.8 else 'normal',
                'volume_strength': float(volume_ratio.iloc[-5:].mean())
            },
            'range_analysis': {
                'price_position_in_range': float(price_position.iloc[-1]),
                'range_width': float((resistance_20.iloc[-1] - support_20.iloc[-1]) / close.iloc[-1] * 100),
                'breakout_potential': 'high' if price_position.iloc[-1] > 0.8 or price_position.iloc[-1] < 0.2 else 'low'
            }
        }
    
    async def _get_ai_regime_analysis(
        self,
        indicators: Dict[str, Any],
        symbol: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get AI analysis of market regime"""
        
        context_str = ""
        if additional_context:
            context_str = f"\nAdditional Context: {additional_context}"
        
        prompt = f"""
        Analyze the current market regime for {symbol} based on technical indicators:
        
        TREND ANALYSIS:
        - Price vs SMA20: {indicators['price_trend']['price_vs_sma_20']:.2f}%
        - Price vs SMA50: {indicators['price_trend']['price_vs_sma_50']:.2f}%
        - SMA20 vs SMA50: {indicators['price_trend']['sma_20_vs_50']:.2f}%
        - Trend Direction: {indicators['price_trend']['trend_direction']}
        
        VOLATILITY ANALYSIS:
        - Current Volatility: {indicators['volatility']['current_volatility']:.2f}%
        - Volatility Percentile: {indicators['volatility']['volatility_percentile']:.1f}%
        - Volatility Regime: {indicators['volatility']['volatility_regime']}
        - ATR vs Average: {indicators['volatility']['atr_current']:.4f} vs {indicators['volatility']['atr_avg']:.4f}
        
        MOMENTUM ANALYSIS:
        - RSI: {indicators['momentum']['rsi']:.2f}
        - MACD: {indicators['momentum']['macd']:.4f}
        - MACD Signal: {indicators['momentum']['macd_signal']:.4f}
        - Momentum Direction: {indicators['momentum']['momentum_direction']}
        
        VOLUME ANALYSIS:
        - Volume Trend: {indicators['volume']['volume_trend']}
        - Volume Strength: {indicators['volume']['volume_strength']:.2f}x
        
        RANGE ANALYSIS:
        - Price Position in Range: {indicators['range_analysis']['price_position_in_range']:.2f}
        - Range Width: {indicators['range_analysis']['range_width']:.2f}%
        - Breakout Potential: {indicators['range_analysis']['breakout_potential']}
        {context_str}
        
        Determine the market regime and provide trading guidance:
        
        Return JSON:
        {{
            "regime": "bull_trending|bear_trending|sideways_range|high_volatility|low_volatility|consolidation",
            "confidence": 0.0-1.0,
            "regime_strength": 0.0-1.0,
            "duration_estimate_days": 1-30,
            "key_characteristics": ["characteristic1", "characteristic2"],
            "regime_reasoning": "detailed explanation",
            "breakout_probability": 0.0-1.0,
            "recommended_approach": "trend_following|mean_reversion|breakout|defensive|opportunistic",
            "position_sizing_factor": 0.3-1.5,
            "stop_loss_adjustment": 0.5-2.0,
            "take_profit_adjustment": 0.5-2.0
        }}
        """
        
        try:
            # Offload synchronous OpenAI call to executor to prevent await of non-awaitable
            loop = asyncio.get_running_loop()
            ai_response = await loop.run_in_executor(None, self.openai.ask_json, prompt)
            
            if not ai_response:
                return self._get_default_ai_response()
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error in AI regime analysis: {e}")
            return self._get_default_ai_response()
    
    def _combine_regime_analysis(
        self,
        indicators: Dict[str, Any],
        ai_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine rule-based and AI analysis"""
        
        # Rule-based regime detection
        rule_based_regime = self._rule_based_regime_detection(indicators)
        
        # Get AI regime
        ai_regime = ai_analysis.get('regime', 'sideways_range')
        ai_confidence = ai_analysis.get('confidence', 0.5)
        
        # Combine regimes
        if rule_based_regime == ai_regime:
            final_confidence = min(ai_confidence + 0.2, 1.0)  # Boost confidence when they agree
            final_regime = ai_regime
        else:
            # Use the one with higher confidence, but reduce overall confidence
            if ai_confidence > 0.7:
                final_regime = ai_regime
                final_confidence = ai_confidence * 0.8
            else:
                final_regime = rule_based_regime
                final_confidence = 0.6
        
        return {
            'regime': final_regime,
            'confidence': final_confidence,
            'regime_strength': ai_analysis.get('regime_strength', 0.5),
            'duration_estimate_days': ai_analysis.get('duration_estimate_days', 7),
            'key_characteristics': ai_analysis.get('key_characteristics', []),
            'regime_reasoning': ai_analysis.get('regime_reasoning', 'Combined analysis'),
            'breakout_probability': ai_analysis.get('breakout_probability', 0.3),
            'recommended_approach': ai_analysis.get('recommended_approach', 'defensive'),
            'position_sizing_factor': ai_analysis.get('position_sizing_factor', 0.7),
            'stop_loss_adjustment': ai_analysis.get('stop_loss_adjustment', 1.0),
            'take_profit_adjustment': ai_analysis.get('take_profit_adjustment', 1.0),
            'rule_based_regime': rule_based_regime,
            'ai_regime': ai_regime,
            'ai_confidence': ai_confidence
        }
    
    def _rule_based_regime_detection(self, indicators: Dict[str, Any]) -> str:
        """Simple rule-based regime detection"""
        
        trend = indicators['price_trend']
        volatility = indicators['volatility']
        momentum = indicators['momentum']
        
        # High volatility regime
        if volatility['volatility_regime'] == 'high':
            return 'high_volatility'
        
        # Low volatility regime
        if volatility['volatility_regime'] == 'low':
            return 'low_volatility'
        
        # Strong trend regimes
        if abs(trend['price_vs_sma_20']) > 5 and abs(trend['sma_20_vs_50']) > 3:
            if trend['price_vs_sma_20'] > 0 and trend['sma_20_vs_50'] > 0:
                return 'bull_trending'
            elif trend['price_vs_sma_20'] < 0 and trend['sma_20_vs_50'] < 0:
                return 'bear_trending'
        
        # Default to sideways
        return 'sideways_range'
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _get_trading_recommendations(self, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get specific trading recommendations based on regime"""
        
        regime = regime_analysis['regime']
        confidence = regime_analysis['confidence']
        
        if regime == 'sideways_range':
            return {
                'strategy': 'mean_reversion',
                'position_size_multiplier': 0.5,  # Reduce position sizes
                'trade_frequency': 'reduced',     # Trade less frequently
                'stop_loss_factor': 1.2,         # Slightly wider stops
                'take_profit_factor': 0.8,       # Quicker profits
                'confidence_requirement': 0.8,   # Higher confidence required
                'notes': 'Focus on range trading, avoid trend following'
            }
        elif regime == 'bull_trending':
            return {
                'strategy': 'trend_following',
                'position_size_multiplier': 1.2,
                'trade_frequency': 'normal',
                'stop_loss_factor': 0.8,
                'take_profit_factor': 1.5,
                'confidence_requirement': 0.6,
                'notes': 'Follow the trend, use momentum strategies'
            }
        elif regime == 'bear_trending':
            return {
                'strategy': 'short_bias',
                'position_size_multiplier': 1.0,
                'trade_frequency': 'normal',
                'stop_loss_factor': 0.8,
                'take_profit_factor': 1.2,
                'confidence_requirement': 0.7,
                'notes': 'Short bias, quick profits, tight stops'
            }
        elif regime == 'high_volatility':
            return {
                'strategy': 'defensive',
                'position_size_multiplier': 0.3,
                'trade_frequency': 'minimal',
                'stop_loss_factor': 1.5,
                'take_profit_factor': 0.6,
                'confidence_requirement': 0.9,
                'notes': 'Defensive positioning, avoid large positions'
            }
        else:
            return {
                'strategy': 'balanced',
                'position_size_multiplier': 0.7,
                'trade_frequency': 'reduced',
                'stop_loss_factor': 1.0,
                'take_profit_factor': 1.0,
                'confidence_requirement': 0.75,
                'notes': 'Balanced approach with reduced risk'
            }
    
    def _update_regime_history(self, regime_analysis: Dict[str, Any]):
        """Update regime history for trend analysis"""
        self.regime_history.append({
            'timestamp': pd.Timestamp.now(),
            'regime': regime_analysis['regime'],
            'confidence': regime_analysis['confidence']
        })
        
        # Keep only last 100 entries
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
    
    def _get_default_regime(self) -> Dict[str, Any]:
        """Default regime when analysis fails"""
        return {
            'regime': 'sideways_range',
            'confidence': 0.5,
            'regime_strength': 0.5,
            'duration_estimate_days': 7,
            'key_characteristics': ['analysis_unavailable'],
            'regime_reasoning': 'Default regime due to analysis failure',
            'breakout_probability': 0.3,
            'recommended_approach': 'defensive',
            'position_sizing_factor': 0.5,
            'stop_loss_adjustment': 1.2,
            'take_profit_adjustment': 0.8,
            'trading_recommendations': {
                'strategy': 'defensive',
                'position_size_multiplier': 0.5,
                'trade_frequency': 'minimal',
                'confidence_requirement': 0.8,
                'notes': 'Conservative approach due to analysis failure'
            }
        }
    
    def _get_default_ai_response(self) -> Dict[str, Any]:
        """Default AI response when AI analysis fails"""
        return {
            'regime': 'sideways_range',
            'confidence': 0.4,
            'regime_strength': 0.4,
            'duration_estimate_days': 5,
            'key_characteristics': ['ai_analysis_failed'],
            'regime_reasoning': 'AI analysis unavailable',
            'breakout_probability': 0.3,
            'recommended_approach': 'defensive',
            'position_sizing_factor': 0.5,
            'stop_loss_adjustment': 1.1,
            'take_profit_adjustment': 0.9
        }


# Global instance
_regime_detector = AIMarketRegimeDetector()

async def detect_market_regime(
    price_data: pd.DataFrame,
    symbol: str = "MARKET",
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Global function for AI market regime detection
    """
    return await _regime_detector.detect_market_regime(price_data, symbol, additional_context)
