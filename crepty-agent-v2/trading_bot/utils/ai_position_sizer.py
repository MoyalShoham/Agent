"""
AI-Powered Position Sizing - Intelligent position management using OpenAI analysis.
Optimizes position sizes based on signal strength, risk factors, and market conditions.
"""
from typing import Dict, Any, Optional
from loguru import logger
import os
import asyncio

try:
    from .openai_client import OpenAIClient
except ImportError:
    from trading_bot.utils.openai_client import OpenAIClient


class AIPositionSizer:
    def __init__(self):
        self.openai = OpenAIClient()
        self.default_risk_per_trade = float(os.getenv('DEFAULT_RISK_PER_TRADE', '0.02'))  # 2%
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.15'))  # 15%
        self.min_position_size = float(os.getenv('MIN_POSITION_SIZE', '0.01'))  # 1%
    
    async def calculate_optimal_position(
        self,
        symbol: str,
        signal_strength: float,
        signal_confidence: float,
        current_price: float,
        available_capital: float,
        portfolio_state: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using AI analysis
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (-1 to 1)
            signal_confidence: Confidence in signal (0 to 1)
            current_price: Current market price
            available_capital: Available trading capital
            portfolio_state: Current portfolio information
            market_context: Additional market context
        
        Returns:
            Dict with position sizing recommendations
        """
        try:
            # Prepare context for AI analysis
            analysis_context = self._prepare_analysis_context(
                symbol, signal_strength, signal_confidence, current_price,
                available_capital, portfolio_state, market_context
            )
            
            # Get AI position sizing recommendation
            ai_recommendation = await self._get_ai_position_analysis(analysis_context)
            
            # Apply safety constraints
            final_position = self._apply_safety_constraints(ai_recommendation, available_capital, current_price)
            
            logger.info(f"AI Position Sizing for {symbol}: ${final_position['position_size_usd']:.2f} ({final_position['position_size_pct']:.1%})")
            
            return final_position
            
        except Exception as e:
            logger.error(f"Error in AI position sizing for {symbol}: {e}")
            return self._get_fallback_position(signal_confidence, available_capital, current_price)
    
    def _prepare_analysis_context(
        self,
        symbol: str,
        signal_strength: float,
        signal_confidence: float,
        current_price: float,
        available_capital: float,
        portfolio_state: Dict[str, Any],
        market_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare comprehensive context for AI analysis"""
        
        # Portfolio risk metrics
        current_positions = portfolio_state.get('positions', {})
        total_exposure = sum(pos.get('notional_value', 0) for pos in current_positions.values())
        position_count = len([pos for pos in current_positions.values() if pos.get('quantity', 0) != 0])
        
        # Calculate portfolio concentration
        portfolio_concentration = total_exposure / available_capital if available_capital > 0 else 0
        
        # Market context defaults
        if market_context is None:
            market_context = {}
        
        # Apply regime-based position sizing factors
        regime = market_context.get('market_regime', 'unknown')
        regime_factor = self._get_regime_position_factor(regime)
        
        # Apply drawdown protection
        drawdown_protection = os.getenv('DRAWDOWN_PROTECTION_MODE', 'false').lower() == 'true'
        drawdown_factor = 0.5 if drawdown_protection else 1.0
        
        context = {
            "symbol": symbol,
            "current_price": current_price,
            "signal_metrics": {
                "strength": signal_strength,
                "confidence": signal_confidence,
                "signal_quality": "high" if signal_confidence > 0.7 else "medium" if signal_confidence > 0.4 else "low"
            },
            "portfolio_state": {
                "available_capital": available_capital,
                "total_exposure": total_exposure,
                "position_count": position_count,
                "concentration": portfolio_concentration,
                "max_positions": portfolio_state.get('max_positions', 8),
                "current_drawdown": portfolio_state.get('current_drawdown', 0.0),
                "drawdown_protection": drawdown_protection
            },
            "risk_parameters": {
                "default_risk_per_trade": self.default_risk_per_trade * drawdown_factor,
                "max_position_size": self.max_position_size * drawdown_factor,
                "min_position_size": self.min_position_size,
                "regime_factor": regime_factor,
                "drawdown_factor": drawdown_factor
            },
            "market_context": {
                "volatility": market_context.get('volatility', 'medium'),
                "regime": regime,
                "correlation_risk": market_context.get('correlation_risk', 'medium'),
                "liquidity": market_context.get('liquidity', 'good'),
                "regime_strength": market_context.get('regime_strength', 0.5),
                "breakout_probability": market_context.get('breakout_probability', 0.3)
            }
        }
        
        return context
    
    def _get_regime_position_factor(self, regime: str) -> float:
        """Get position sizing factor based on market regime"""
        regime_factors = {
            'bull_trending': 1.2,
            'bear_trending': 1.0,
            'sideways_range': float(os.getenv('SIDEWAYS_MARKET_FACTOR', '0.5')),
            'high_volatility': float(os.getenv('HIGH_VOLATILITY_FACTOR', '0.7')),
            'low_volatility': 1.1,
            'consolidation': 0.8,
            'unknown': 0.7
        }
        return regime_factors.get(regime, 0.7)
        
        return context
    
    async def _get_ai_position_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered position sizing analysis"""
        
        prompt = f"""
        Calculate optimal position size for {context['symbol']} trade with DRAWDOWN PROTECTION:
        
        SIGNAL ANALYSIS:
        - Signal Strength: {context['signal_metrics']['strength']:.2f} (-1 to 1)
        - Signal Confidence: {context['signal_metrics']['confidence']:.2f} (0 to 1)
        - Signal Quality: {context['signal_metrics']['signal_quality']}
        
        PORTFOLIO STATE:
        - Available Capital: ${context['portfolio_state']['available_capital']:,.2f}
        - Current Exposure: ${context['portfolio_state']['total_exposure']:,.2f}
        - Position Count: {context['portfolio_state']['position_count']}/{context['portfolio_state']['max_positions']}
        - Portfolio Concentration: {context['portfolio_state']['concentration']:.1%}
        - Current Drawdown: {context['portfolio_state']['current_drawdown']:.1%}
        - Drawdown Protection Mode: {context['portfolio_state']['drawdown_protection']}
        
        MARKET CONDITIONS:
        - Market Regime: {context['market_context']['regime']}
        - Regime Strength: {context['market_context']['regime_strength']:.2f}
        - Volatility: {context['market_context']['volatility']}
        - Correlation Risk: {context['market_context']['correlation_risk']}
        - Breakout Probability: {context['market_context']['breakout_probability']:.2f}
        
        RISK PARAMETERS (ADJUSTED):
        - Base Risk per Trade: {context['risk_parameters']['default_risk_per_trade']:.1%}
        - Max Position Size: {context['risk_parameters']['max_position_size']:.1%}
        - Regime Factor: {context['risk_parameters']['regime_factor']:.2f}
        - Drawdown Factor: {context['risk_parameters']['drawdown_factor']:.2f}
        
        SPECIAL CONSIDERATIONS FOR CURRENT SITUATION:
        - System is in drawdown protection mode
        - Market regime appears to be sideways/ranging
        - Reduce position sizes significantly
        - Require higher confidence levels
        - Focus on capital preservation
        
        Calculate conservative position considering:
        1. CAPITAL PRESERVATION is priority #1
        2. Significantly reduced sizing during drawdown
        3. Market regime specific adjustments
        4. Higher confidence requirements
        5. Tighter risk controls
        
        Return JSON:
        {{
            "position_size_pct": 0.005-0.08,
            "risk_per_trade_pct": 0.003-0.01,
            "stop_loss_pct": 0.03-0.08,
            "take_profit_pct": 0.02-0.15,
            "confidence_multiplier": 0.3-1.5,
            "risk_score": 0.0-1.0,
            "reasoning": "detailed conservative explanation",
            "risk_factors": ["factor1", "factor2"],
            "position_rationale": "why this conservative size is optimal",
            "drawdown_adjustment": "how drawdown protection affects sizing",
            "regime_impact": "how market regime influences position"
        }}
        """
        
        try:
            # Offload synchronous OpenAI client call to executor to avoid improper await errors
            loop = asyncio.get_running_loop()
            ai_response = await loop.run_in_executor(None, self.openai.ask_json, prompt)
            
            if not ai_response:
                return self._get_default_ai_response()
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error in AI position analysis: {e}")
            return self._get_default_ai_response()
    
    def _apply_safety_constraints(
        self,
        ai_recommendation: Dict[str, Any],
        available_capital: float,
        current_price: float
    ) -> Dict[str, Any]:
        """Apply safety constraints to AI recommendations"""
        
        # Extract AI recommendations with defaults
        ai_position_pct = ai_recommendation.get('position_size_pct', self.default_risk_per_trade)
        ai_risk_pct = ai_recommendation.get('risk_per_trade_pct', self.default_risk_per_trade)
        ai_stop_loss_pct = ai_recommendation.get('stop_loss_pct', 0.05)
        ai_take_profit_pct = ai_recommendation.get('take_profit_pct', 0.10)
        confidence_multiplier = ai_recommendation.get('confidence_multiplier', 1.0)
        
        # Apply hard limits
        position_size_pct = max(
            self.min_position_size,
            min(self.max_position_size, ai_position_pct)
        )
        
        risk_per_trade_pct = max(0.005, min(0.05, ai_risk_pct))  # 0.5% to 5%
        stop_loss_pct = max(0.02, min(0.10, ai_stop_loss_pct))   # 2% to 10%
        take_profit_pct = max(0.03, min(0.30, ai_take_profit_pct)) # 3% to 30%
        
        # Calculate position size in USD
        position_size_usd = available_capital * position_size_pct
        
        # Calculate quantity
        quantity = position_size_usd / current_price if current_price > 0 else 0
        
        return {
            "position_size_usd": position_size_usd,
            "position_size_pct": position_size_pct,
            "quantity": quantity,
            "risk_per_trade_pct": risk_per_trade_pct,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "stop_loss_price": current_price * (1 - stop_loss_pct),
            "take_profit_price": current_price * (1 + take_profit_pct),
            "confidence_multiplier": confidence_multiplier,
            "risk_score": ai_recommendation.get('risk_score', 0.5),
            "reasoning": ai_recommendation.get('reasoning', 'AI-optimized position sizing'),
            "risk_factors": ai_recommendation.get('risk_factors', []),
            "position_rationale": ai_recommendation.get('position_rationale', 'Standard position sizing applied')
        }
    
    def _get_fallback_position(
        self,
        signal_confidence: float,
        available_capital: float,
        current_price: float
    ) -> Dict[str, Any]:
        """Get fallback position sizing when AI analysis fails"""
        
        # Conservative fallback based on confidence
        base_size = self.default_risk_per_trade
        confidence_adjusted_size = base_size * max(0.5, signal_confidence)
        
        position_size_pct = max(self.min_position_size, min(self.max_position_size, confidence_adjusted_size))
        position_size_usd = available_capital * position_size_pct
        quantity = position_size_usd / current_price if current_price > 0 else 0
        
        return {
            "position_size_usd": position_size_usd,
            "position_size_pct": position_size_pct,
            "quantity": quantity,
            "risk_per_trade_pct": self.default_risk_per_trade,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "stop_loss_price": current_price * 0.95,
            "take_profit_price": current_price * 1.10,
            "confidence_multiplier": signal_confidence,
            "risk_score": 0.5,
            "reasoning": "Fallback position sizing due to AI analysis failure",
            "risk_factors": ["AI analysis unavailable"],
            "position_rationale": "Conservative fallback sizing applied"
        }
    
    def _get_default_ai_response(self) -> Dict[str, Any]:
        """Get default AI response when analysis fails"""
        return {
            "position_size_pct": self.default_risk_per_trade,
            "risk_per_trade_pct": self.default_risk_per_trade,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.10,
            "confidence_multiplier": 1.0,
            "risk_score": 0.5,
            "reasoning": "Default position sizing applied",
            "risk_factors": ["AI analysis failed"],
            "position_rationale": "Standard risk management applied"
        }


# Global instance
_ai_position_sizer = AIPositionSizer()

async def calculate_ai_position_size(
    symbol: str,
    signal_strength: float,
    signal_confidence: float,
    current_price: float,
    available_capital: float,
    portfolio_state: Dict[str, Any],
    market_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Global function for AI-powered position sizing
    """
    return await _ai_position_sizer.calculate_optimal_position(
        symbol, signal_strength, signal_confidence, current_price,
        available_capital, portfolio_state, market_context
    )
