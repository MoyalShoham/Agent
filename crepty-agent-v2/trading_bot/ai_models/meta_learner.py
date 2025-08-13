"""
Meta-Learner System - Combines strategy signals and AI agent insights
Uses contextual bandits and ensemble methods for optimal allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import lightgbm as lgb
import optuna
from loguru import logger

from trading_bot.strategies.strategy_experts import StrategySignal, Signal
from trading_bot.ai_models.ai_agents import AIAgentOrchestrator

class MetaSignalType(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

@dataclass
class MetaSignal:
    """Combined signal from meta-learner"""
    symbol: str
    signal_type: MetaSignalType
    confidence: float
    position_size: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    contributing_strategies: List[str]
    ai_agent_insights: Dict[str, Any]
    risk_score: float
    expected_return: float
    expected_volatility: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class MarketRegimeFeatures:
    """Features for market regime detection"""
    volatility_regime: str  # 'low', 'medium', 'high'
    trend_regime: str  # 'bull', 'bear', 'sideways'
    volume_regime: str  # 'low', 'normal', 'high'
    funding_regime: str  # 'normal', 'extreme_positive', 'extreme_negative'
    sentiment_regime: str  # 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
    time_of_day: int  # Hour of day (0-23)
    day_of_week: int  # Day of week (0-6)

class ContextualBandit:
    """Contextual bandit for strategy allocation"""
    
    def __init__(self, n_strategies: int, alpha: float = 1.0):
        self.n_strategies = n_strategies
        self.alpha = alpha
        self.A = np.eye(n_strategies) * alpha  # Covariance matrix
        self.b = np.zeros(n_strategies)  # Reward vector
        self.theta = np.zeros(n_strategies)  # Parameter vector
        self.confidence_bound = 0.5
        
    def select_strategy(self, context: np.ndarray) -> Tuple[int, float]:
        """Select strategy using Upper Confidence Bound"""
        # Update theta
        try:
            self.theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            self.theta = np.linalg.lstsq(self.A, self.b, rcond=None)[0]
            
        # Calculate confidence bounds
        context = context.reshape(-1, 1) if context.ndim == 1 else context
        
        confidence_bounds = []
        for i in range(self.n_strategies):
            try:
                A_inv = np.linalg.inv(self.A)
                confidence = self.confidence_bound * np.sqrt(
                    context.T @ A_inv @ context
                )[0, 0]
            except:
                confidence = self.confidence_bound
                
            expected_reward = self.theta[i] * np.sum(context)
            upper_bound = expected_reward + confidence
            confidence_bounds.append(upper_bound)
            
        # Select strategy with highest upper confidence bound
        selected_strategy = np.argmax(confidence_bounds)
        confidence = confidence_bounds[selected_strategy]
        
        return selected_strategy, confidence
        
    def update(self, strategy_idx: int, context: np.ndarray, reward: float):
        """Update bandit with observed reward"""
        context = context.reshape(-1, 1) if context.ndim == 1 else context
        
        # Update covariance matrix and reward vector
        self.A += context @ context.T
        self.b[strategy_idx] += reward

class MetaLearner:
    """Meta-learning system for strategy allocation and signal generation"""
    
    def __init__(self, model_save_path: str = "meta_models"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        # Models
        self.ensemble_model = None
        self.regime_classifier = None
        self.position_sizer = None
        self.risk_predictor = None
        self.scaler = StandardScaler()
        
        # Contextual bandit
        self.bandit = None
        
        # Data storage
        self.training_data = []
        self.performance_history = []
        self.regime_history = []
        
        # Configuration
        self.retrain_frequency = 1000  # Retrain every N samples
        self.min_training_samples = 100
        self.confidence_threshold = 0.6
        
        # Feature engineering
        self.feature_columns = []
        
    async def initialize(self, n_strategies: int):
        """Initialize meta-learner with number of strategies"""
        self.bandit = ContextualBandit(n_strategies)
        self._load_models()
        logger.info(f"✅ Meta-learner initialized with {n_strategies} strategies")
        
    def extract_features(self, 
                        market_data: Dict[str, Any], 
                        strategy_signals: List[StrategySignal],
                        ai_insights: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for meta-learning"""
        features = {}
        
        # Market data features
        if 'ohlcv' in market_data:
            df = market_data['ohlcv']
            if len(df) > 0:
                # Price features
                features['price_change_1h'] = df['close'].pct_change().iloc[-1] if len(df) > 1 else 0
                features['price_change_4h'] = df['close'].pct_change(4).iloc[-1] if len(df) > 4 else 0
                features['price_change_24h'] = df['close'].pct_change(24).iloc[-1] if len(df) > 24 else 0
                
                # Volume features
                features['volume_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else 1
                features['volume_trend'] = df['volume'].rolling(5).mean().iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else 1
                
                # Volatility features
                returns = df['close'].pct_change()
                features['volatility_1h'] = returns.rolling(24).std().iloc[-1] if len(df) > 24 else 0
                features['volatility_24h'] = returns.rolling(24*7).std().iloc[-1] if len(df) > 24*7 else 0
                
                # Technical features
                if len(df) > 20:
                    sma_20 = df['close'].rolling(20).mean()
                    features['price_vs_sma20'] = (df['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
                    
                if len(df) > 50:
                    sma_50 = df['close'].rolling(50).mean()
                    features['sma20_vs_sma50'] = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        
        # Strategy signal features
        if strategy_signals:
            # Signal agreement
            buy_signals = sum(1 for s in strategy_signals if s.signal == Signal.LONG)
            sell_signals = sum(1 for s in strategy_signals if s.signal == Signal.SHORT)
            total_signals = len(strategy_signals)
            
            features['signal_agreement'] = abs(buy_signals - sell_signals) / max(total_signals, 1)
            features['signal_strength'] = (buy_signals - sell_signals) / max(total_signals, 1)
            
            # Average confidence
            features['avg_signal_confidence'] = np.mean([s.confidence for s in strategy_signals])
            features['max_signal_confidence'] = np.max([s.confidence for s in strategy_signals])
            features['min_signal_confidence'] = np.min([s.confidence for s in strategy_signals])
            
            # Position size recommendations
            features['avg_position_size'] = np.mean([s.position_size for s in strategy_signals])
            features['max_position_size'] = np.max([s.position_size for s in strategy_signals])
        else:
            features.update({
                'signal_agreement': 0, 'signal_strength': 0,
                'avg_signal_confidence': 0, 'max_signal_confidence': 0, 'min_signal_confidence': 0,
                'avg_position_size': 0, 'max_position_size': 0
            })
            
        # AI agent insights features
        if ai_insights:
            # Financial analysis features
            if 'financial_analysis' in ai_insights and ai_insights['financial_analysis']:
                fa = ai_insights['financial_analysis']
                features['fa_confidence'] = getattr(fa, 'confidence_score', 0)
                
                # Map market regime to numerical value
                regime_map = {'bull_trend': 1, 'bear_trend': -1, 'sideways': 0, 
                             'high_volatility': 0.5, 'low_volatility': -0.5}
                features['market_regime'] = regime_map.get(getattr(fa, 'market_regime', 'sideways'), 0)
                
            # Risk management features
            if 'risk_assessment' in ai_insights and ai_insights['risk_assessment']:
                risk = ai_insights['risk_assessment']
                risk_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'extreme': 1.0}
                features['risk_level'] = risk_map.get(getattr(risk, 'overall_risk_level', 'medium'), 0.5)
                features['risk_confidence'] = getattr(risk, 'confidence', 0)
                
            # Sentiment features
            if 'sentiment_analysis' in ai_insights and ai_insights['sentiment_analysis']:
                sentiment = ai_insights['sentiment_analysis']
                features['sentiment_score'] = getattr(sentiment, 'sentiment_score', 0)
                features['fear_greed_index'] = getattr(sentiment, 'fear_greed_index', 50) / 100
                features['sentiment_reliability'] = getattr(sentiment, 'sentiment_reliability', 0)
        
        # Time-based features
        now = datetime.now()
        features['hour_of_day'] = now.hour / 24
        features['day_of_week'] = now.weekday() / 7
        features['is_weekend'] = 1 if now.weekday() >= 5 else 0
        
        # Funding and derivatives features
        if 'funding_rate' in market_data:
            features['funding_rate'] = market_data['funding_rate']
            features['funding_extreme'] = 1 if abs(market_data['funding_rate']) > 0.01 else 0
            
        if 'open_interest' in market_data:
            features['oi_change'] = market_data.get('oi_change', 0)
            
        return features
        
    def detect_market_regime(self, features: Dict[str, float]) -> MarketRegimeFeatures:
        """Detect current market regime"""
        # Volatility regime
        vol = features.get('volatility_24h', 0)
        if vol > 0.05:
            vol_regime = 'high'
        elif vol < 0.02:
            vol_regime = 'low'
        else:
            vol_regime = 'medium'
            
        # Trend regime
        price_trend = features.get('price_change_24h', 0)
        if price_trend > 0.05:
            trend_regime = 'bull'
        elif price_trend < -0.05:
            trend_regime = 'bear'
        else:
            trend_regime = 'sideways'
            
        # Volume regime
        vol_ratio = features.get('volume_ratio', 1)
        if vol_ratio > 2:
            volume_regime = 'high'
        elif vol_ratio < 0.5:
            volume_regime = 'low'
        else:
            volume_regime = 'normal'
            
        # Funding regime
        funding = features.get('funding_rate', 0)
        if funding > 0.005:
            funding_regime = 'extreme_positive'
        elif funding < -0.005:
            funding_regime = 'extreme_negative'
        else:
            funding_regime = 'normal'
            
        # Sentiment regime
        fear_greed = features.get('fear_greed_index', 0.5)
        if fear_greed > 0.8:
            sentiment_regime = 'extreme_greed'
        elif fear_greed > 0.6:
            sentiment_regime = 'greed'
        elif fear_greed < 0.2:
            sentiment_regime = 'extreme_fear'
        elif fear_greed < 0.4:
            sentiment_regime = 'fear'
        else:
            sentiment_regime = 'neutral'
            
        return MarketRegimeFeatures(
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            volume_regime=volume_regime,
            funding_regime=funding_regime,
            sentiment_regime=sentiment_regime,
            time_of_day=int(features.get('hour_of_day', 0) * 24),
            day_of_week=int(features.get('day_of_week', 0) * 7)
        )
        
    async def generate_meta_signal(self,
                                  symbol: str,
                                  market_data: Dict[str, Any],
                                  strategy_signals: List[StrategySignal],
                                  ai_insights: Dict[str, Any]) -> Optional[MetaSignal]:
        """Generate meta-signal combining all inputs"""
        
        if not strategy_signals:
            return None
            
        try:
            # Extract features
            features = self.extract_features(market_data, strategy_signals, ai_insights)
            regime = self.detect_market_regime(features)
            
            # Create feature vector for models
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Strategy selection using contextual bandit
            if self.bandit:
                context_features = np.array([
                    features.get('signal_agreement', 0),
                    features.get('avg_signal_confidence', 0),
                    features.get('volatility_24h', 0),
                    features.get('market_regime', 0),
                    features.get('risk_level', 0.5)
                ])
                
                selected_strategy_idx, bandit_confidence = self.bandit.select_strategy(context_features)
            else:
                selected_strategy_idx = 0
                bandit_confidence = 0.5
                
            # Ensemble prediction
            signal_prediction = self._predict_signal(feature_vector, regime)
            confidence_prediction = self._predict_confidence(feature_vector, regime)
            position_size_prediction = self._predict_position_size(feature_vector, regime)
            risk_prediction = self._predict_risk(feature_vector, regime)
            
            # Combine strategy signals with meta-predictions
            combined_signal = self._combine_signals(
                strategy_signals, 
                signal_prediction, 
                confidence_prediction,
                selected_strategy_idx
            )
            
            if combined_signal['signal_type'] == MetaSignalType.HOLD:
                return None
                
            # Calculate final position size
            final_position_size = min(
                position_size_prediction,
                combined_signal['position_size'],
                self._get_max_position_size(risk_prediction, regime)
            )
            
            # Set entry/exit levels
            current_price = market_data.get('current_price', 0)
            if current_price == 0 and 'ohlcv' in market_data and len(market_data['ohlcv']) > 0:
                current_price = market_data['ohlcv']['close'].iloc[-1]
                
            stop_loss, take_profit = self._calculate_exit_levels(
                current_price, 
                combined_signal['signal_type'], 
                features, 
                regime
            )
            
            # Expected return and volatility
            expected_return = self._calculate_expected_return(features, combined_signal['signal_type'])
            expected_volatility = features.get('volatility_24h', 0.02)
            
            meta_signal = MetaSignal(
                symbol=symbol,
                signal_type=combined_signal['signal_type'],
                confidence=combined_signal['confidence'],
                position_size=final_position_size,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                contributing_strategies=[s.strategy_name for s in strategy_signals],
                ai_agent_insights=ai_insights,
                risk_score=risk_prediction,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                timestamp=datetime.now(),
                metadata={
                    'regime': asdict(regime),
                    'features': features,
                    'selected_strategy_idx': selected_strategy_idx,
                    'bandit_confidence': bandit_confidence,
                    'ensemble_components': {
                        'signal_prediction': signal_prediction,
                        'confidence_prediction': confidence_prediction,
                        'position_size_prediction': position_size_prediction,
                        'risk_prediction': risk_prediction
                    }
                }
            )
            
            return meta_signal
            
        except Exception as e:
            logger.error(f"Meta-signal generation error: {e}")
            return None
            
    def _predict_signal(self, feature_vector: np.ndarray, regime: MarketRegimeFeatures) -> float:
        """Predict signal strength using ensemble model"""
        if self.ensemble_model is None:
            # Default prediction based on simple rules
            return 0.0
            
        try:
            # Normalize features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            prediction = self.ensemble_model.predict_proba(feature_vector_scaled)[0]
            
            # Convert to signal strength (-1 to 1)
            if len(prediction) == 3:  # [sell, hold, buy]
                signal_strength = prediction[2] - prediction[0]  # buy - sell
            else:
                signal_strength = 0.0
                
            return signal_strength
            
        except Exception as e:
            logger.error(f"Signal prediction error: {e}")
            return 0.0
            
    def _predict_confidence(self, feature_vector: np.ndarray, regime: MarketRegimeFeatures) -> float:
        """Predict confidence in the signal"""
        # Simple confidence based on signal agreement and volatility
        try:
            if feature_vector.shape[1] > 5:
                signal_agreement = feature_vector[0, 0] if len(feature_vector[0]) > 0 else 0
                avg_confidence = feature_vector[0, 2] if len(feature_vector[0]) > 2 else 0
                volatility = feature_vector[0, 7] if len(feature_vector[0]) > 7 else 0.02
                
                # Higher confidence with higher agreement and lower volatility
                confidence = (signal_agreement * 0.6 + avg_confidence * 0.4) * (1 - min(volatility * 10, 0.5))
                return max(0.1, min(0.9, confidence))
        except:
            pass
            
        return 0.5
        
    def _predict_position_size(self, feature_vector: np.ndarray, regime: MarketRegimeFeatures) -> float:
        """Predict optimal position size"""
        if self.position_sizer is None:
            # Default position sizing based on volatility and confidence
            try:
                volatility = feature_vector[0, 7] if len(feature_vector[0]) > 7 else 0.02
                confidence = feature_vector[0, 2] if len(feature_vector[0]) > 2 else 0.5
                
                # Inverse volatility scaling with confidence adjustment
                base_size = 0.1 / max(volatility * 10, 1)
                adjusted_size = base_size * confidence
                
                return max(0.01, min(0.25, adjusted_size))
            except:
                return 0.05
                
        try:
            feature_vector_scaled = self.scaler.transform(feature_vector)
            prediction = self.position_sizer.predict(feature_vector_scaled)[0]
            return max(0.01, min(0.25, prediction))
        except Exception as e:
            logger.error(f"Position size prediction error: {e}")
            return 0.05
            
    def _predict_risk(self, feature_vector: np.ndarray, regime: MarketRegimeFeatures) -> float:
        """Predict risk score"""
        if self.risk_predictor is None:
            # Default risk based on volatility and market conditions
            try:
                volatility = feature_vector[0, 7] if len(feature_vector[0]) > 7 else 0.02
                risk_level = feature_vector[0, 10] if len(feature_vector[0]) > 10 else 0.5
                
                # Combine volatility and risk assessment
                risk_score = (volatility * 5 + risk_level) / 2
                return max(0.1, min(0.9, risk_score))
            except:
                return 0.5
                
        try:
            feature_vector_scaled = self.scaler.transform(feature_vector)
            prediction = self.risk_predictor.predict(feature_vector_scaled)[0]
            return max(0.1, min(0.9, prediction))
        except Exception as e:
            logger.error(f"Risk prediction error: {e}")
            return 0.5
            
    def _combine_signals(self, 
                        strategy_signals: List[StrategySignal], 
                        meta_prediction: float,
                        confidence_prediction: float,
                        selected_strategy_idx: int) -> Dict[str, Any]:
        """Combine strategy signals with meta-predictions"""
        
        # Weight strategy signals
        total_weight = 0
        weighted_signal = 0
        weighted_confidence = 0
        weighted_position_size = 0
        
        for i, signal in enumerate(strategy_signals):
            weight = signal.confidence
            if i == selected_strategy_idx:
                weight *= 1.5  # Boost selected strategy
                
            signal_value = signal.signal.value
            weighted_signal += signal_value * weight
            weighted_confidence += signal.confidence * weight
            weighted_position_size += signal.position_size * weight
            total_weight += weight
            
        if total_weight > 0:
            avg_signal = weighted_signal / total_weight
            avg_confidence = weighted_confidence / total_weight
            avg_position_size = weighted_position_size / total_weight
        else:
            avg_signal = 0
            avg_confidence = 0
            avg_position_size = 0
            
        # Combine with meta-prediction
        final_signal = (avg_signal * 0.7 + meta_prediction * 0.3)
        final_confidence = (avg_confidence * 0.8 + confidence_prediction * 0.2)
        
        # Convert to signal type
        if final_signal > 0.7:
            signal_type = MetaSignalType.STRONG_BUY
        elif final_signal > 0.2:
            signal_type = MetaSignalType.BUY
        elif final_signal < -0.7:
            signal_type = MetaSignalType.STRONG_SELL
        elif final_signal < -0.2:
            signal_type = MetaSignalType.SELL
        else:
            signal_type = MetaSignalType.HOLD
            
        return {
            'signal_type': signal_type,
            'confidence': final_confidence,
            'position_size': avg_position_size
        }
        
    def _get_max_position_size(self, risk_score: float, regime: MarketRegimeFeatures) -> float:
        """Get maximum allowed position size based on risk"""
        base_max = 0.25
        
        # Reduce based on risk score
        risk_adjustment = 1 - (risk_score * 0.5)
        
        # Reduce based on volatility regime
        vol_adjustment = {'low': 1.0, 'medium': 0.8, 'high': 0.5}[regime.volatility_regime]
        
        # Reduce based on funding extremes
        funding_adjustment = 0.7 if 'extreme' in regime.funding_regime else 1.0
        
        max_size = base_max * risk_adjustment * vol_adjustment * funding_adjustment
        return max(0.01, min(0.25, max_size))
        
    def _calculate_exit_levels(self, 
                              entry_price: float, 
                              signal_type: MetaSignalType, 
                              features: Dict[str, float], 
                              regime: MarketRegimeFeatures) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        
        if entry_price <= 0:
            return None, None
            
        # Base ATR or volatility
        volatility = features.get('volatility_24h', 0.02)
        atr_estimate = entry_price * volatility
        
        # Adjust based on regime
        vol_multiplier = {'low': 1.5, 'medium': 2.0, 'high': 2.5}[regime.volatility_regime]
        
        if signal_type in [MetaSignalType.BUY, MetaSignalType.STRONG_BUY]:
            stop_loss = entry_price - (atr_estimate * vol_multiplier)
            take_profit = entry_price + (atr_estimate * vol_multiplier * 2)
        elif signal_type in [MetaSignalType.SELL, MetaSignalType.STRONG_SELL]:
            stop_loss = entry_price + (atr_estimate * vol_multiplier)
            take_profit = entry_price - (atr_estimate * vol_multiplier * 2)
        else:
            return None, None
            
        return stop_loss, take_profit
        
    def _calculate_expected_return(self, features: Dict[str, float], signal_type: MetaSignalType) -> float:
        """Calculate expected return for the signal"""
        base_return = 0.02  # 2% base expected return
        
        # Adjust based on signal strength
        signal_multiplier = {
            MetaSignalType.STRONG_BUY: 2.0,
            MetaSignalType.BUY: 1.5,
            MetaSignalType.HOLD: 0.0,
            MetaSignalType.SELL: -1.5,
            MetaSignalType.STRONG_SELL: -2.0
        }[signal_type]
        
        # Adjust based on confidence and market conditions
        confidence = features.get('avg_signal_confidence', 0.5)
        volatility = features.get('volatility_24h', 0.02)
        
        expected_return = base_return * signal_multiplier * confidence * (1 + volatility)
        return expected_return
        
    async def update_performance(self, meta_signal: MetaSignal, actual_return: float, trade_duration_hours: int):
        """Update meta-learner with actual performance"""
        
        # Update contextual bandit
        if self.bandit and 'selected_strategy_idx' in meta_signal.metadata:
            selected_idx = meta_signal.metadata['selected_strategy_idx']
            context_features = np.array([
                meta_signal.metadata['features'].get('signal_agreement', 0),
                meta_signal.confidence,
                meta_signal.expected_volatility,
                meta_signal.metadata['features'].get('market_regime', 0),
                meta_signal.risk_score
            ])
            
            # Normalize reward (actual return relative to expected)
            reward = actual_return / max(abs(meta_signal.expected_return), 0.01)
            self.bandit.update(selected_idx, context_features, reward)
            
        # Store training data
        training_sample = {
            'features': meta_signal.metadata['features'],
            'regime': meta_signal.metadata['regime'],
            'signal_type': meta_signal.signal_type.value,
            'confidence': meta_signal.confidence,
            'position_size': meta_signal.position_size,
            'expected_return': meta_signal.expected_return,
            'actual_return': actual_return,
            'trade_duration': trade_duration_hours,
            'timestamp': meta_signal.timestamp
        }
        
        self.training_data.append(training_sample)
        
        # Performance tracking
        performance_record = {
            'timestamp': datetime.now(),
            'symbol': meta_signal.symbol,
            'signal_type': meta_signal.signal_type.value,
            'confidence': meta_signal.confidence,
            'expected_return': meta_signal.expected_return,
            'actual_return': actual_return,
            'position_size': meta_signal.position_size,
            'trade_duration': trade_duration_hours,
            'was_profitable': actual_return > 0
        }
        
        self.performance_history.append(performance_record)
        
        # Retrain if needed
        if len(self.training_data) % self.retrain_frequency == 0:
            await self._retrain_models()
            
        logger.info(f"📊 Updated meta-learner: {meta_signal.symbol} "
                   f"Expected: {meta_signal.expected_return:.3f}, "
                   f"Actual: {actual_return:.3f}, "
                   f"Confidence: {meta_signal.confidence:.3f}")
                   
    async def _retrain_models(self):
        """Retrain meta-learning models"""
        if len(self.training_data) < self.min_training_samples:
            return
            
        logger.info("🔄 Retraining meta-learning models...")
        
        try:
            # Prepare training data
            df = pd.DataFrame(self.training_data)
            
            # Feature matrix
            feature_cols = list(df['features'].iloc[0].keys())
            X = np.array([list(sample['features'].values()) for sample in self.training_data])
            
            # Targets
            y_signal = df['signal_type'].values
            y_return = df['actual_return'].values
            y_confidence = df['confidence'].values
            
            # Train/test split (time series)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_signal_train, y_signal_test = y_signal[:split_idx], y_signal[split_idx:]
            
            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble model for signal prediction
            ensemble_models = [
                ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ]
            
            best_score = 0
            best_model = None
            
            for name, model in ensemble_models:
                try:
                    model.fit(X_train_scaled, y_signal_train + 2)  # Shift to positive values
                    score = model.score(X_test_scaled, y_signal_test + 2)
                    if score > best_score:
                        best_score = score
                        best_model = model
                        logger.info(f"New best model: {name} (score: {score:.3f})")
                except Exception as e:
                    logger.error(f"Model {name} training error: {e}")
                    
            if best_model:
                self.ensemble_model = CalibratedClassifierCV(best_model, cv=3)
                self.ensemble_model.fit(X_train_scaled, y_signal_train + 2)
                
            # Save models
            self._save_models()
            
            logger.info(f"✅ Meta-learning models retrained with {len(self.training_data)} samples")
            
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
            
    def _save_models(self):
        """Save trained models"""
        try:
            models = {
                'ensemble_model': self.ensemble_model,
                'regime_classifier': self.regime_classifier,
                'position_sizer': self.position_sizer,
                'risk_predictor': self.risk_predictor,
                'scaler': self.scaler,
                'bandit': self.bandit
            }
            
            with open(self.model_save_path / 'meta_models.pkl', 'wb') as f:
                pickle.dump(models, f)
                
        except Exception as e:
            logger.error(f"Model saving error: {e}")
            
    def _load_models(self):
        """Load trained models"""
        try:
            model_file = self.model_save_path / 'meta_models.pkl'
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    models = pickle.load(f)
                    
                self.ensemble_model = models.get('ensemble_model')
                self.regime_classifier = models.get('regime_classifier')
                self.position_sizer = models.get('position_sizer')
                self.risk_predictor = models.get('risk_predictor')
                self.scaler = models.get('scaler', StandardScaler())
                self.bandit = models.get('bandit')
                
                logger.info("✅ Meta-learning models loaded")
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for meta-learner"""
        if not self.performance_history:
            return {}
            
        df = pd.DataFrame(self.performance_history)
        
        metrics = {
            'total_trades': len(df),
            'win_rate': df['was_profitable'].mean(),
            'avg_return': df['actual_return'].mean(),
            'avg_expected_return': df['expected_return'].mean(),
            'return_std': df['actual_return'].std(),
            'sharpe_ratio': df['actual_return'].mean() / max(df['actual_return'].std(), 0.001),
            'avg_confidence': df['confidence'].mean(),
            'last_30_days': {
                'trades': len(df[df['timestamp'] > datetime.now() - timedelta(days=30)]),
                'win_rate': df[df['timestamp'] > datetime.now() - timedelta(days=30)]['was_profitable'].mean() if len(df[df['timestamp'] > datetime.now() - timedelta(days=30)]) > 0 else 0,
                'avg_return': df[df['timestamp'] > datetime.now() - timedelta(days=30)]['actual_return'].mean() if len(df[df['timestamp'] > datetime.now() - timedelta(days=30)]) > 0 else 0
            }
        }
        
        return metrics

# Factory function
def create_meta_learner() -> MetaLearner:
    """Create and initialize meta-learner"""
    return MetaLearner()
