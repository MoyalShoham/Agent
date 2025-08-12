"""
AI Model Integration and Optimization Framework
Handles AI agent coordination, model optimization, and intelligent decision making.
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import openai
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import logging
from loguru import logger
import os
from collections import defaultdict, deque

@dataclass
class AIModelConfig:
    """AI model configuration"""
    openai_api_key: str = ""
    model_name: str = "gpt-4"
    max_tokens: int = 1500
    temperature: float = 0.3
    confidence_threshold: float = 0.7
    ensemble_voting_threshold: float = 0.6
    model_retrain_interval_hours: int = 24
    feature_importance_threshold: float = 0.05
    cross_validation_folds: int = 5

@dataclass
class AIDecision:
    """AI decision structure"""
    decision_type: str  # 'trade', 'risk_adjustment', 'portfolio_optimization'
    action: str
    confidence: float
    reasoning: str
    supporting_data: Dict[str, Any]
    timestamp: datetime
    model_used: str
    risk_score: float = 0.5
    expected_return: float = 0.0
    time_horizon: str = "short"  # short, medium, long

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: datetime
    predictions_count: int
    correct_predictions: int

class AIModelIntegrator:
    """
    Advanced AI model integration system for trading decisions.
    Combines OpenAI API, local ML models, and ensemble methods.
    """
    
    def __init__(self, config: AIModelConfig = None):
        self.config = config or AIModelConfig()
        
        # AI Models
        self.openai_client = None
        self.local_models: Dict[str, Any] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.scaler = StandardScaler()
        
        # Decision tracking
        self.decision_history: deque = deque(maxlen=1000)
        self.model_predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Feature engineering
        self.feature_columns: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
        # Performance tracking
        self.ensemble_performance = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'cumulative_return': 0.0,
            'win_rate': 0.0
        }
        
        # Model files
        self.model_dir = 'ai_models'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_openai()
        self._load_local_models()
        
        logger.info("AI Model Integrator initialized")

    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            if self.config.openai_api_key:
                openai.api_key = self.config.openai_api_key
                self.openai_client = openai
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not provided - GPT features disabled")
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")

    def _load_local_models(self):
        """Load or initialize local ML models"""
        try:
            # Random Forest for trend prediction
            rf_path = os.path.join(self.model_dir, 'random_forest_trend.joblib')
            if os.path.exists(rf_path):
                self.local_models['random_forest'] = joblib.load(rf_path)
                logger.info("Random Forest model loaded")
            else:
                self.local_models['random_forest'] = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )
                logger.info("New Random Forest model initialized")
            
            # Gradient Boosting for volatility prediction
            gb_path = os.path.join(self.model_dir, 'gradient_boosting_volatility.joblib')
            if os.path.exists(gb_path):
                self.local_models['gradient_boosting'] = joblib.load(gb_path)
                logger.info("Gradient Boosting model loaded")
            else:
                self.local_models['gradient_boosting'] = GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=42
                )
                logger.info("New Gradient Boosting model initialized")
            
            # Load feature scaler
            scaler_path = os.path.join(self.model_dir, 'feature_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded")
            
        except Exception as e:
            logger.error(f"Error loading local models: {e}")

    async def get_ai_trading_decision(self, market_data: Dict, portfolio_data: Dict, 
                                     risk_metrics: Dict) -> AIDecision:
        """Get comprehensive AI trading decision using multiple models"""
        try:
            # Prepare context for AI
            context = self._prepare_ai_context(market_data, portfolio_data, risk_metrics)
            
            # Get decisions from different models
            decisions = {}
            
            # OpenAI decision
            if self.openai_client:
                gpt_decision = await self._get_openai_decision(context)
                decisions['openai'] = gpt_decision
            
            # Local ML model decisions
            local_decision = self._get_local_model_decision(context)
            decisions['local_ml'] = local_decision
            
            # Technical analysis decision
            ta_decision = self._get_technical_analysis_decision(market_data)
            decisions['technical'] = ta_decision
            
            # Ensemble decision
            final_decision = self._create_ensemble_decision(decisions, context)
            
            # Record decision
            self.decision_history.append(final_decision)
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Error getting AI trading decision: {e}")
            return AIDecision(
                decision_type='trade',
                action='hold',
                confidence=0.0,
                reasoning=f"Error in AI decision: {str(e)}",
                supporting_data={},
                timestamp=datetime.now(),
                model_used='error_fallback',
                risk_score=1.0
            )

    def _prepare_ai_context(self, market_data: Dict, portfolio_data: Dict, 
                           risk_metrics: Dict) -> Dict:
        """Prepare comprehensive context for AI decision making"""
        return {
            'market_data': {
                'symbol': market_data.get('symbol', 'UNKNOWN'),
                'price': market_data.get('price', 0),
                'volume': market_data.get('volume', 0),
                'price_change_24h': market_data.get('price_change_24h', 0),
                'volatility': market_data.get('volatility', 0),
                'trend': market_data.get('trend', 'neutral'),
                'support_level': market_data.get('support_level', 0),
                'resistance_level': market_data.get('resistance_level', 0),
                'rsi': market_data.get('rsi', 50),
                'macd': market_data.get('macd', 0),
                'bollinger_position': market_data.get('bollinger_position', 0.5)
            },
            'portfolio_data': {
                'total_value': portfolio_data.get('total_value', 0),
                'available_balance': portfolio_data.get('available_balance', 0),
                'current_positions': portfolio_data.get('current_positions', {}),
                'daily_pnl': portfolio_data.get('daily_pnl', 0),
                'win_rate': portfolio_data.get('win_rate', 0),
                'sharpe_ratio': portfolio_data.get('sharpe_ratio', 0)
            },
            'risk_metrics': {
                'portfolio_risk': risk_metrics.get('portfolio_risk', 0.5),
                'max_drawdown': risk_metrics.get('max_drawdown', 0),
                'var_95': risk_metrics.get('var_95', 0),
                'correlation_risk': risk_metrics.get('correlation_risk', 0),
                'leverage_ratio': risk_metrics.get('leverage_ratio', 1),
                'emergency_mode': risk_metrics.get('emergency_mode', False)
            },
            'market_sentiment': self._analyze_market_sentiment(market_data),
            'recent_performance': self._get_recent_performance_summary()
        }

    async def _get_openai_decision(self, context: Dict) -> Dict:
        """Get trading decision from OpenAI GPT model"""
        try:
            if not self.openai_client:
                return self._create_fallback_decision("OpenAI not available")
            
            # Create prompt
            prompt = self._create_openai_prompt(context)
            
            # Get response
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert crypto trading AI assistant. Analyze the provided market data and give clear, actionable trading recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse response
            ai_response = response.choices[0].message.content
            decision = self._parse_openai_response(ai_response)
            
            logger.info(f"OpenAI decision: {decision['action']} (confidence: {decision['confidence']:.1%})")
            return decision
            
        except Exception as e:
            logger.error(f"Error getting OpenAI decision: {e}")
            return self._create_fallback_decision(f"OpenAI error: {str(e)}")

    def _create_openai_prompt(self, context: Dict) -> str:
        """Create comprehensive prompt for OpenAI"""
        market = context['market_data']
        portfolio = context['portfolio_data']
        risk = context['risk_metrics']
        
        prompt = f"""
        CRYPTO TRADING ANALYSIS REQUEST
        
        MARKET DATA:
        - Symbol: {market['symbol']}
        - Current Price: ${market['price']:.4f}
        - 24h Change: {market['price_change_24h']:.2%}
        - Volume: {market['volume']:,.0f}
        - Volatility: {market['volatility']:.1%}
        - Trend: {market['trend']}
        - RSI: {market['rsi']:.1f}
        - MACD: {market['macd']:.4f}
        - Support: ${market['support_level']:.4f}
        - Resistance: ${market['resistance_level']:.4f}
        
        PORTFOLIO STATUS:
        - Total Value: ${portfolio['total_value']:,.2f}
        - Available Balance: ${portfolio['available_balance']:,.2f}
        - Daily PnL: ${portfolio['daily_pnl']:,.2f}
        - Win Rate: {portfolio['win_rate']:.1%}
        - Active Positions: {len(portfolio['current_positions'])}
        
        RISK METRICS:
        - Portfolio Risk: {risk['portfolio_risk']:.1%}
        - Max Drawdown: {risk['max_drawdown']:.1%}
        - VaR (95%): ${risk['var_95']:,.2f}
        - Emergency Mode: {risk['emergency_mode']}
        
        MARKET SENTIMENT: {context['market_sentiment']}
        
        RECENT PERFORMANCE: {context['recent_performance']}
        
        Please provide a trading recommendation with:
        1. Action (buy/sell/hold)
        2. Confidence level (0-1)
        3. Reasoning (2-3 sentences)
        4. Risk assessment (low/medium/high)
        5. Expected return potential
        6. Time horizon (short/medium/long)
        
        Format your response as JSON:
        {{
            "action": "buy/sell/hold",
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation",
            "risk_level": "low/medium/high",
            "expected_return": 0.0-1.0,
            "time_horizon": "short/medium/long",
            "position_size_recommendation": 0.0-1.0
        }}
        """
        
        return prompt

    def _parse_openai_response(self, response: str) -> Dict:
        """Parse OpenAI JSON response"""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                return {
                    'action': parsed.get('action', 'hold'),
                    'confidence': float(parsed.get('confidence', 0.5)),
                    'reasoning': parsed.get('reasoning', 'No reasoning provided'),
                    'risk_level': parsed.get('risk_level', 'medium'),
                    'expected_return': float(parsed.get('expected_return', 0.0)),
                    'time_horizon': parsed.get('time_horizon', 'short'),
                    'position_size_rec': float(parsed.get('position_size_recommendation', 0.5))
                }
            else:
                # Fallback parsing
                return self._parse_text_response(response)
                
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            return self._create_fallback_decision("Response parsing error")

    def _parse_text_response(self, response: str) -> Dict:
        """Fallback text parsing for OpenAI response"""
        response_lower = response.lower()
        
        # Determine action
        if 'buy' in response_lower:
            action = 'buy'
        elif 'sell' in response_lower:
            action = 'sell'
        else:
            action = 'hold'
        
        # Estimate confidence
        confidence = 0.5
        if 'high confidence' in response_lower or 'very confident' in response_lower:
            confidence = 0.8
        elif 'low confidence' in response_lower or 'uncertain' in response_lower:
            confidence = 0.3
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': response[:200] + "..." if len(response) > 200 else response,
            'risk_level': 'medium',
            'expected_return': 0.0,
            'time_horizon': 'short',
            'position_size_rec': 0.5
        }

    def _get_local_model_decision(self, context: Dict) -> Dict:
        """Get decision from local ML models"""
        try:
            # Prepare features
            features = self._extract_ml_features(context)
            
            if len(features) == 0:
                return self._create_fallback_decision("No features available")
            
            # Get predictions from each model
            predictions = {}
            
            for model_name, model in self.local_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        # Scale features
                        features_scaled = self.scaler.transform([features])
                        
                        # Get probability predictions
                        probabilities = model.predict_proba(features_scaled)[0]
                        prediction = model.predict(features_scaled)[0]
                        
                        confidence = max(probabilities) if len(probabilities) > 0 else 0.5
                        
                        predictions[model_name] = {
                            'prediction': prediction,
                            'confidence': confidence,
                            'probabilities': probabilities.tolist()
                        }
                        
                except Exception as e:
                    logger.error(f"Error with model {model_name}: {e}")
                    continue
            
            # Aggregate predictions
            if predictions:
                return self._aggregate_local_predictions(predictions)
            else:
                return self._create_fallback_decision("All local models failed")
                
        except Exception as e:
            logger.error(f"Error in local model decision: {e}")
            return self._create_fallback_decision(f"Local model error: {str(e)}")

    def _extract_ml_features(self, context: Dict) -> List[float]:
        """Extract numerical features for ML models"""
        try:
            market = context['market_data']
            portfolio = context['portfolio_data']
            risk = context['risk_metrics']
            
            features = [
                market.get('price', 0),
                market.get('volume', 0) / 1000000,  # Scale volume
                market.get('price_change_24h', 0),
                market.get('volatility', 0),
                market.get('rsi', 50) / 100,  # Normalize RSI
                market.get('macd', 0),
                market.get('bollinger_position', 0.5),
                portfolio.get('daily_pnl', 0) / 1000,  # Scale PnL
                portfolio.get('win_rate', 0),
                portfolio.get('sharpe_ratio', 0),
                risk.get('portfolio_risk', 0.5),
                risk.get('max_drawdown', 0),
                risk.get('leverage_ratio', 1),
                len(portfolio.get('current_positions', {})),
                1 if risk.get('emergency_mode', False) else 0
            ]
            
            # Remove any NaN or infinite values
            features = [f if np.isfinite(f) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting ML features: {e}")
            return []

    def _aggregate_local_predictions(self, predictions: Dict) -> Dict:
        """Aggregate predictions from multiple local models"""
        try:
            # Convert predictions to actions
            actions = []
            confidences = []
            
            for model_name, pred_data in predictions.items():
                prediction = pred_data['prediction']
                confidence = pred_data['confidence']
                
                # Map prediction to action (assuming binary classification: 0=sell/hold, 1=buy)
                if prediction == 1:
                    actions.append('buy')
                else:
                    actions.append('hold')  # Conservative approach
                
                confidences.append(confidence)
            
            # Determine final action by majority vote
            action_counts = {'buy': 0, 'sell': 0, 'hold': 0}
            for action in actions:
                action_counts[action] += 1
            
            final_action = max(action_counts, key=action_counts.get)
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            return {
                'action': final_action,
                'confidence': avg_confidence,
                'reasoning': f"Local ML models consensus: {action_counts}",
                'risk_level': 'medium',
                'expected_return': avg_confidence * 0.1,  # Conservative estimate
                'time_horizon': 'short',
                'position_size_rec': avg_confidence * 0.5
            }
            
        except Exception as e:
            logger.error(f"Error aggregating local predictions: {e}")
            return self._create_fallback_decision("Aggregation error")

    def _get_technical_analysis_decision(self, market_data: Dict) -> Dict:
        """Get decision based on technical analysis"""
        try:
            rsi = market_data.get('rsi', 50)
            macd = market_data.get('macd', 0)
            trend = market_data.get('trend', 'neutral')
            bollinger_pos = market_data.get('bollinger_position', 0.5)
            
            # RSI signals
            rsi_signal = 'neutral'
            if rsi < 30:
                rsi_signal = 'buy'  # Oversold
            elif rsi > 70:
                rsi_signal = 'sell'  # Overbought
            
            # MACD signals
            macd_signal = 'buy' if macd > 0 else 'sell'
            
            # Trend signals
            trend_signal = 'buy' if trend == 'uptrend' else 'sell' if trend == 'downtrend' else 'hold'
            
            # Bollinger Bands signals
            bb_signal = 'buy' if bollinger_pos < 0.2 else 'sell' if bollinger_pos > 0.8 else 'neutral'
            
            # Aggregate signals
            signals = [rsi_signal, macd_signal, trend_signal, bb_signal]
            signal_counts = {'buy': 0, 'sell': 0, 'hold': 0, 'neutral': 0}
            
            for signal in signals:
                signal_counts[signal] += 1
            
            # Determine final action
            if signal_counts['buy'] > signal_counts['sell'] and signal_counts['buy'] > signal_counts['hold']:
                action = 'buy'
                confidence = signal_counts['buy'] / len(signals)
            elif signal_counts['sell'] > signal_counts['buy'] and signal_counts['sell'] > signal_counts['hold']:
                action = 'sell'
                confidence = signal_counts['sell'] / len(signals)
            else:
                action = 'hold'
                confidence = max(signal_counts['hold'], signal_counts['neutral']) / len(signals)
            
            reasoning = f"Technical analysis: RSI={rsi:.1f}, MACD={macd:.4f}, Trend={trend}, BB_pos={bollinger_pos:.2f}"
            
            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'risk_level': 'low',
                'expected_return': confidence * 0.05,
                'time_horizon': 'short',
                'position_size_rec': confidence * 0.3
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis decision: {e}")
            return self._create_fallback_decision("Technical analysis error")

    def _create_ensemble_decision(self, decisions: Dict, context: Dict) -> AIDecision:
        """Create final decision by combining multiple model outputs"""
        try:
            # Weight models based on historical performance
            model_weights = {
                'openai': 0.4,
                'local_ml': 0.35,
                'technical': 0.25
            }
            
            # Collect all decisions
            actions = []
            confidences = []
            weighted_confidences = []
            
            reasoning_parts = []
            
            for model_name, decision in decisions.items():
                weight = model_weights.get(model_name, 0.2)
                
                actions.append(decision['action'])
                conf = decision['confidence']
                confidences.append(conf)
                weighted_confidences.append(conf * weight)
                
                reasoning_parts.append(f"{model_name}: {decision['action']} ({conf:.1%})")
            
            # Determine final action
            action_weights = defaultdict(float)
            for i, action in enumerate(actions):
                action_weights[action] += weighted_confidences[i]
            
            final_action = max(action_weights, key=action_weights.get)
            final_confidence = sum(weighted_confidences)
            
            # Apply ensemble voting threshold
            if final_confidence < self.config.ensemble_voting_threshold:
                final_action = 'hold'
                final_confidence = 0.5
                reasoning_parts.append("Ensemble confidence below threshold - defaulting to HOLD")
            
            # Calculate risk score
            risk_score = self._calculate_ensemble_risk_score(decisions, context)
            
            # Create final decision
            decision = AIDecision(
                decision_type='trade',
                action=final_action,
                confidence=final_confidence,
                reasoning=f"Ensemble decision: {' | '.join(reasoning_parts)}",
                supporting_data={
                    'individual_decisions': decisions,
                    'model_weights': model_weights,
                    'action_weights': dict(action_weights)
                },
                timestamp=datetime.now(),
                model_used='ensemble',
                risk_score=risk_score,
                expected_return=self._estimate_expected_return(final_action, final_confidence),
                time_horizon='short'
            )
            
            logger.info(f"Ensemble decision: {final_action} (confidence: {final_confidence:.1%}, risk: {risk_score:.1%})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error creating ensemble decision: {e}")
            return AIDecision(
                decision_type='trade',
                action='hold',
                confidence=0.0,
                reasoning=f"Ensemble error: {str(e)}",
                supporting_data={},
                timestamp=datetime.now(),
                model_used='error_fallback',
                risk_score=1.0
            )

    def _calculate_ensemble_risk_score(self, decisions: Dict, context: Dict) -> float:
        """Calculate risk score for ensemble decision"""
        try:
            # Base risk from market conditions
            market_risk = context['market_data'].get('volatility', 0) * 2
            
            # Portfolio risk
            portfolio_risk = context['risk_metrics'].get('portfolio_risk', 0.5)
            
            # Decision consensus risk (lower consensus = higher risk)
            actions = [d['action'] for d in decisions.values()]
            action_consensus = max(actions.count(action) for action in set(actions)) / len(actions)
            consensus_risk = 1 - action_consensus
            
            # Confidence variation risk
            confidences = [d['confidence'] for d in decisions.values()]
            confidence_std = np.std(confidences) if len(confidences) > 1 else 0
            variation_risk = confidence_std
            
            # Emergency mode risk
            emergency_risk = 0.5 if context['risk_metrics'].get('emergency_mode', False) else 0.0
            
            # Combine risk factors
            total_risk = (
                market_risk * 0.3 +
                portfolio_risk * 0.25 +
                consensus_risk * 0.2 +
                variation_risk * 0.15 +
                emergency_risk * 0.1
            )
            
            return min(max(total_risk, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.8  # Conservative default

    def _estimate_expected_return(self, action: str, confidence: float) -> float:
        """Estimate expected return based on action and confidence"""
        if action == 'hold':
            return 0.0
        
        # Base return estimation
        base_return = confidence * 0.02  # 2% max expected return at full confidence
        
        # Adjust for action type
        if action == 'sell':
            base_return *= -0.5  # Selling has lower expected return in crypto
        
        return base_return

    def _analyze_market_sentiment(self, market_data: Dict) -> str:
        """Analyze overall market sentiment"""
        try:
            price_change = market_data.get('price_change_24h', 0)
            volatility = market_data.get('volatility', 0)
            volume = market_data.get('volume', 0)
            
            # Determine sentiment
            if price_change > 0.05 and volume > 1000000:
                return "Strongly Bullish"
            elif price_change > 0.02:
                return "Bullish"
            elif price_change < -0.05 and volume > 1000000:
                return "Strongly Bearish"
            elif price_change < -0.02:
                return "Bearish"
            elif volatility > 0.05:
                return "Highly Volatile"
            else:
                return "Neutral"
                
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return "Unknown"

    def _get_recent_performance_summary(self) -> str:
        """Get summary of recent AI decision performance"""
        try:
            if len(self.decision_history) < 5:
                return "Insufficient history"
            
            recent_decisions = list(self.decision_history)[-10:]
            
            # Calculate metrics
            avg_confidence = np.mean([d.confidence for d in recent_decisions])
            action_distribution = defaultdict(int)
            
            for decision in recent_decisions:
                action_distribution[decision.action] += 1
            
            return f"Recent 10 decisions: Avg confidence {avg_confidence:.1%}, Actions: {dict(action_distribution)}"
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return "Performance data unavailable"

    def _create_fallback_decision(self, reason: str) -> Dict:
        """Create fallback decision when models fail"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'reasoning': f"Fallback decision: {reason}",
            'risk_level': 'high',
            'expected_return': 0.0,
            'time_horizon': 'short',
            'position_size_rec': 0.0
        }

    def update_model_performance(self, decision: AIDecision, actual_outcome: float):
        """Update model performance tracking based on actual outcomes"""
        try:
            model_name = decision.model_used
            
            # Determine if prediction was correct
            predicted_direction = 1 if decision.action == 'buy' else -1 if decision.action == 'sell' else 0
            actual_direction = 1 if actual_outcome > 0 else -1 if actual_outcome < 0 else 0
            
            is_correct = (predicted_direction == actual_direction) or (predicted_direction == 0 and abs(actual_outcome) < 0.01)
            
            # Update ensemble performance
            self.ensemble_performance['total_predictions'] += 1
            if is_correct:
                self.ensemble_performance['correct_predictions'] += 1
            
            self.ensemble_performance['cumulative_return'] += actual_outcome
            self.ensemble_performance['win_rate'] = (
                self.ensemble_performance['correct_predictions'] / 
                max(self.ensemble_performance['total_predictions'], 1)
            )
            
            # Update individual model performance (simplified)
            if model_name not in self.model_performance:
                self.model_performance[model_name] = ModelPerformance(
                    model_name=model_name,
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    last_updated=datetime.now(),
                    predictions_count=0,
                    correct_predictions=0
                )
            
            perf = self.model_performance[model_name]
            perf.predictions_count += 1
            if is_correct:
                perf.correct_predictions += 1
            
            perf.accuracy = perf.correct_predictions / max(perf.predictions_count, 1)
            perf.last_updated = datetime.now()
            
            logger.info(f"Updated performance for {model_name}: {perf.accuracy:.1%} accuracy")
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")

    def get_ai_status(self) -> Dict:
        """Get comprehensive AI system status"""
        return {
            'openai_available': self.openai_client is not None,
            'local_models_loaded': len(self.local_models),
            'total_decisions': len(self.decision_history),
            'ensemble_performance': self.ensemble_performance,
            'model_performance': {
                name: {
                    'accuracy': perf.accuracy,
                    'predictions_count': perf.predictions_count,
                    'last_updated': perf.last_updated.isoformat()
                }
                for name, perf in self.model_performance.items()
            },
            'recent_decisions': len([
                d for d in self.decision_history
                if d.timestamp > datetime.now() - timedelta(hours=1)
            ]),
            'average_confidence': np.mean([d.confidence for d in self.decision_history]) if self.decision_history else 0.0
        }

    def save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.local_models.items():
                if hasattr(model, 'fit') and model_name in ['random_forest', 'gradient_boosting']:
                    model_path = os.path.join(self.model_dir, f'{model_name}.joblib')
                    joblib.dump(model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, 'feature_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            # Save performance data
            performance_path = os.path.join(self.model_dir, 'model_performance.json')
            with open(performance_path, 'w') as f:
                performance_data = {
                    name: asdict(perf) for name, perf in self.model_performance.items()
                }
                # Convert datetime to string
                for data in performance_data.values():
                    data['last_updated'] = data['last_updated'].isoformat()
                
                json.dump(performance_data, f, indent=2)
            
            logger.info("AI models and performance data saved")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

# Global instance
ai_integrator = AIModelIntegrator()
