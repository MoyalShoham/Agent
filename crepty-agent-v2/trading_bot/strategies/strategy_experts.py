"""
Strategy Experts - Individual Trading Strategies
Each strategy emits probabilistic actions with confidence scores
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import talib
from loguru import logger
import asyncio
from datetime import datetime, timedelta

class Signal(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

@dataclass
class StrategySignal:
    """Signal output from strategy expert"""
    strategy_name: str
    symbol: str
    signal: Signal
    confidence: float  # 0.0 to 1.0
    position_size: float  # Suggested position size (0.0 to 1.0)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.is_enabled = True
        self.last_signal = None
        self.performance_metrics = {
            'total_signals': 0,
            'winning_signals': 0,
            'avg_confidence': 0.0,
            'last_updated': datetime.now()
        }
        
    @abstractmethod
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate trading signal based on market data"""
        pass
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        if len(df) < period:
            return np.array([np.nan] * len(df))
        return talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
    def calculate_position_size(self, confidence: float, volatility: float, base_size: float = 0.1) -> float:
        """Calculate position size based on confidence and volatility"""
        vol_adjusted = base_size * (1 / max(volatility, 0.1))  # Inverse volatility scaling
        confidence_adjusted = vol_adjusted * confidence
        return min(max(confidence_adjusted, 0.01), 0.25)  # Cap between 1% and 25%
        
    def update_performance(self, signal_result: Dict[str, Any]):
        """Update strategy performance metrics"""
        self.performance_metrics['total_signals'] += 1
        if signal_result.get('profitable', False):
            self.performance_metrics['winning_signals'] += 1
        self.performance_metrics['last_updated'] = datetime.now()

class TrendFollowingStrategy(BaseStrategy):
    """Trend Following Strategy using Donchian Breakouts + HTF Filter"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'donchian_period': 20,
            'htf_ema_fast': 50,
            'htf_ema_slow': 200,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'min_breakout_volume': 1.5  # Volume should be 1.5x average
        }
        super().__init__("TrendFollowing", {**default_params, **(params or {})})
        
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate trend following signals"""
        try:
            df = market_data.get('ohlcv')
            if df is None or len(df) < max(self.params['donchian_period'], self.params['htf_ema_slow']):
                return None
                
            # Calculate indicators
            donchian_high = df['high'].rolling(self.params['donchian_period']).max()
            donchian_low = df['low'].rolling(self.params['donchian_period']).min()
            
            ema_fast = df['close'].ewm(span=self.params['htf_ema_fast']).mean()
            ema_slow = df['close'].ewm(span=self.params['htf_ema_slow']).mean()
            
            atr = self.calculate_atr(df, self.params['atr_period'])
            
            volume_avg = df['volume'].rolling(20).mean()
            
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # HTF trend filter
            htf_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1]
            htf_bearish = ema_fast.iloc[-1] < ema_slow.iloc[-1]
            
            # Breakout conditions
            breakout_up = current_price > donchian_high.iloc[-2]
            breakout_down = current_price < donchian_low.iloc[-2]
            
            # Volume confirmation
            volume_confirmed = current_volume > (volume_avg.iloc[-1] * self.params['min_breakout_volume'])
            
            signal = Signal.FLAT
            confidence = 0.0
            stop_loss = None
            take_profit = None
            
            if breakout_up and htf_bullish and volume_confirmed:
                signal = Signal.LONG
                confidence = min(0.8, 0.5 + (current_price - donchian_high.iloc[-2]) / donchian_high.iloc[-2])
                stop_loss = current_price - (atr[-1] * self.params['atr_multiplier'])
                take_profit = current_price + (atr[-1] * self.params['atr_multiplier'] * 2)
                
            elif breakout_down and htf_bearish and volume_confirmed:
                signal = Signal.SHORT
                confidence = min(0.8, 0.5 + (donchian_low.iloc[-2] - current_price) / donchian_low.iloc[-2])
                stop_loss = current_price + (atr[-1] * self.params['atr_multiplier'])
                take_profit = current_price - (atr[-1] * self.params['atr_multiplier'] * 2)
                
            if signal != Signal.FLAT:
                volatility = atr[-1] / current_price
                position_size = self.calculate_position_size(confidence, volatility)
                
                return StrategySignal(
                    strategy_name=self.name,
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal=signal,
                    confidence=confidence,
                    position_size=position_size,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'donchian_high': donchian_high.iloc[-1],
                        'donchian_low': donchian_low.iloc[-1],
                        'htf_trend': 'bullish' if htf_bullish else 'bearish',
                        'volume_ratio': current_volume / volume_avg.iloc[-1],
                        'atr': atr[-1]
                    }
                )
                
        except Exception as e:
            logger.error(f"TrendFollowing strategy error: {e}")
            
        return None

class VolatilityBreakoutStrategy(BaseStrategy):
    """Volatility Compression to Expansion Strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rv_period': 20,
            'rv_percentile_low': 20,
            'min_expansion_factor': 1.5,
            'volume_surge_factor': 2.0
        }
        super().__init__("VolatilityBreakout", {**default_params, **(params or {})})
        
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate volatility breakout signals"""
        try:
            df = market_data.get('ohlcv')
            if df is None or len(df) < self.params['bb_period']:
                return None
                
            # Bollinger Bands
            bb_middle = df['close'].rolling(self.params['bb_period']).mean()
            bb_std = df['close'].rolling(self.params['bb_period']).std()
            bb_upper = bb_middle + (bb_std * self.params['bb_std'])
            bb_lower = bb_middle - (bb_std * self.params['bb_std'])
            bb_width = (bb_upper - bb_lower) / bb_middle
            
            # Realized Volatility
            returns = df['close'].pct_change()
            rv = returns.rolling(self.params['rv_period']).std() * np.sqrt(365 * 24)  # Annualized
            rv_percentile = rv.rolling(100).rank(pct=True) * 100
            
            # Volume analysis
            volume_avg = df['volume'].rolling(20).mean()
            
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            current_rv_percentile = rv_percentile.iloc[-1] if not pd.isna(rv_percentile.iloc[-1]) else 50
            
            # Compression detection
            low_volatility = current_rv_percentile < self.params['rv_percentile_low']
            tight_bb = bb_width.iloc[-1] < bb_width.rolling(50).quantile(0.2).iloc[-1]
            
            # Expansion detection
            volume_surge = current_volume > (volume_avg.iloc[-1] * self.params['volume_surge_factor'])
            price_break_up = current_price > bb_upper.iloc[-1]
            price_break_down = current_price < bb_lower.iloc[-1]
            
            signal = Signal.FLAT
            confidence = 0.0
            
            if low_volatility and tight_bb:
                if price_break_up and volume_surge:
                    signal = Signal.LONG
                    confidence = 0.7
                elif price_break_down and volume_surge:
                    signal = Signal.SHORT
                    confidence = 0.7
                    
            if signal != Signal.FLAT:
                atr = self.calculate_atr(df)
                volatility = atr[-1] / current_price if not np.isnan(atr[-1]) else 0.02
                position_size = self.calculate_position_size(confidence, volatility)
                
                stop_loss = bb_middle.iloc[-1] if signal == Signal.LONG else bb_middle.iloc[-1]
                take_profit = current_price + (2 * abs(current_price - bb_middle.iloc[-1])) if signal == Signal.LONG else current_price - (2 * abs(current_price - bb_middle.iloc[-1]))
                
                return StrategySignal(
                    strategy_name=self.name,
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal=signal,
                    confidence=confidence,
                    position_size=position_size,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'rv_percentile': current_rv_percentile,
                        'bb_width': bb_width.iloc[-1],
                        'volume_ratio': current_volume / volume_avg.iloc[-1],
                        'compression_detected': low_volatility and tight_bb
                    }
                )
                
        except Exception as e:
            logger.error(f"VolatilityBreakout strategy error: {e}")
            
        return None

class FundingSqueezeStrategy(BaseStrategy):
    """Funding Rate Squeeze/Reversal Strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'funding_extreme_threshold': 0.01,  # 1% funding rate
            'oi_change_threshold': 0.05,  # 5% OI change
            'cvd_divergence_period': 10,
            'min_confidence': 0.6
        }
        super().__init__("FundingSqueeze", {**default_params, **(params or {})})
        
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate funding squeeze signals"""
        try:
            funding_data = market_data.get('funding_history')
            oi_data = market_data.get('oi_history')
            price_data = market_data.get('ohlcv')
            
            if not all([funding_data, oi_data, price_data]) or len(funding_data) < 5:
                return None
                
            latest_funding = funding_data[-1]['funding_rate']
            latest_oi = oi_data[-1]['open_interest']
            previous_oi = oi_data[-2]['open_interest'] if len(oi_data) > 1 else latest_oi
            
            current_price = price_data['close'].iloc[-1]
            
            # OI change
            oi_change = (latest_oi - previous_oi) / previous_oi
            
            # Extreme funding conditions
            extreme_positive_funding = latest_funding > self.params['funding_extreme_threshold']
            extreme_negative_funding = latest_funding < -self.params['funding_extreme_threshold']
            
            # Rising OI indicates building pressure
            rising_oi = oi_change > self.params['oi_change_threshold']
            
            signal = Signal.FLAT
            confidence = 0.0
            
            # Long squeeze setup (extreme positive funding + rising OI + price stalling)
            if extreme_positive_funding and rising_oi:
                signal = Signal.SHORT  # Fade the longs
                confidence = min(0.8, abs(latest_funding) * 50)  # Scale confidence with funding extremity
                
            # Short squeeze setup (extreme negative funding + rising OI)
            elif extreme_negative_funding and rising_oi:
                signal = Signal.LONG  # Fade the shorts
                confidence = min(0.8, abs(latest_funding) * 50)
                
            if signal != Signal.FLAT and confidence >= self.params['min_confidence']:
                # Tight stops for reversal trades
                atr = self.calculate_atr(price_data)
                volatility = atr[-1] / current_price if not np.isnan(atr[-1]) else 0.02
                position_size = self.calculate_position_size(confidence, volatility, base_size=0.05)  # Smaller base size
                
                stop_distance = atr[-1] * 1.5 if not np.isnan(atr[-1]) else current_price * 0.02
                stop_loss = current_price + stop_distance if signal == Signal.SHORT else current_price - stop_distance
                take_profit = current_price - stop_distance * 2 if signal == Signal.SHORT else current_price + stop_distance * 2
                
                return StrategySignal(
                    strategy_name=self.name,
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal=signal,
                    confidence=confidence,
                    position_size=position_size,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'funding_rate': latest_funding,
                        'oi_change': oi_change,
                        'squeeze_type': 'long_squeeze' if signal == Signal.SHORT else 'short_squeeze'
                    }
                )
                
        except Exception as e:
            logger.error(f"FundingSqueeze strategy error: {e}")
            
        return None

class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy for Range-bound Markets"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'vwap_period': 20,
            'zscore_period': 20,
            'zscore_entry': 2.0,
            'zscore_exit': 0.5,
            'range_filter_period': 50,
            'max_spread_bps': 10  # Max 10 bps spread
        }
        super().__init__("MeanReversion", {**default_params, **(params or {})})
        
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Generate mean reversion signals"""
        try:
            df = market_data.get('ohlcv')
            orderbook = market_data.get('latest_orderbook')
            
            if df is None or len(df) < self.params['zscore_period']:
                return None
                
            # Check spread condition
            if orderbook:
                best_bid = orderbook.bids[0][0] if orderbook.bids else 0
                best_ask = orderbook.asks[0][0] if orderbook.asks else 0
                if best_bid > 0 and best_ask > 0:
                    spread_bps = ((best_ask - best_bid) / best_bid) * 10000
                    if spread_bps > self.params['max_spread_bps']:
                        return None  # Spread too wide
                        
            # VWAP calculation
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).rolling(self.params['vwap_period']).sum() / df['volume'].rolling(self.params['vwap_period']).sum()
            
            # Z-score from VWAP
            price_deviation = df['close'] - vwap
            zscore = (price_deviation - price_deviation.rolling(self.params['zscore_period']).mean()) / price_deviation.rolling(self.params['zscore_period']).std()
            
            # Range detection (HTF trend should be flat)
            ema_fast = df['close'].ewm(span=20).mean()
            ema_slow = df['close'].ewm(span=50).mean()
            range_condition = abs(ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] < 0.02  # Less than 2% difference
            
            current_price = df['close'].iloc[-1]
            current_zscore = zscore.iloc[-1]
            
            signal = Signal.FLAT
            confidence = 0.0
            
            if range_condition and not pd.isna(current_zscore):
                if current_zscore > self.params['zscore_entry']:  # Overbought
                    signal = Signal.SHORT
                    confidence = min(0.7, abs(current_zscore) / 3)
                elif current_zscore < -self.params['zscore_entry']:  # Oversold
                    signal = Signal.LONG
                    confidence = min(0.7, abs(current_zscore) / 3)
                    
            if signal != Signal.FLAT:
                # Mean reversion targets
                target_price = vwap.iloc[-1]
                stop_distance = abs(current_price - target_price) * 1.5
                
                stop_loss = current_price + stop_distance if signal == Signal.SHORT else current_price - stop_distance
                take_profit = target_price
                
                volatility = abs(current_zscore) / 3  # Use z-score as volatility proxy
                position_size = self.calculate_position_size(confidence, volatility, base_size=0.08)
                
                return StrategySignal(
                    strategy_name=self.name,
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    signal=signal,
                    confidence=confidence,
                    position_size=position_size,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'zscore': current_zscore,
                        'vwap': vwap.iloc[-1],
                        'range_detected': range_condition,
                        'spread_bps': spread_bps if 'spread_bps' in locals() else None
                    }
                )
                
        except Exception as e:
            logger.error(f"MeanReversion strategy error: {e}")
            
        return None

class StrategyManager:
    """Manages all strategy experts and coordinates signal generation"""
    
    def __init__(self):
        self.strategies: List[BaseStrategy] = []
        self.strategy_weights = {}
        self.signal_history = []
        self.performance_tracker = {}
        
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Add a strategy to the manager"""
        self.strategies.append(strategy)
        self.strategy_weights[strategy.name] = weight
        self.performance_tracker[strategy.name] = {
            'total_signals': 0,
            'profitable_signals': 0,
            'avg_pnl': 0.0,
            'win_rate': 0.0,
            'last_updated': datetime.now()
        }
        logger.info(f"✅ Added strategy: {strategy.name} (weight: {weight})")
        
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy"""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
        if strategy_name in self.strategy_weights:
            del self.strategy_weights[strategy_name]
        if strategy_name in self.performance_tracker:
            del self.performance_tracker[strategy_name]
        logger.info(f"❌ Removed strategy: {strategy_name}")
        
    async def generate_all_signals(self, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Generate signals from all enabled strategies"""
        signals = []
        
        tasks = []
        for strategy in self.strategies:
            if strategy.is_enabled:
                task = strategy.generate_signal(market_data)
                tasks.append(task)
            else:
                tasks.append(asyncio.coroutine(lambda: None)())
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {self.strategies[i].name} error: {result}")
            elif result is not None:
                # Apply strategy weight to confidence
                result.confidence *= self.strategy_weights.get(self.strategies[i].name, 1.0)
                signals.append(result)
                
        return signals
        
    def update_strategy_performance(self, strategy_name: str, pnl: float, was_profitable: bool):
        """Update strategy performance metrics"""
        if strategy_name in self.performance_tracker:
            perf = self.performance_tracker[strategy_name]
            perf['total_signals'] += 1
            if was_profitable:
                perf['profitable_signals'] += 1
            perf['avg_pnl'] = (perf['avg_pnl'] * (perf['total_signals'] - 1) + pnl) / perf['total_signals']
            perf['win_rate'] = perf['profitable_signals'] / perf['total_signals']
            perf['last_updated'] = datetime.now()
            
    def get_strategy_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for all strategies"""
        return self.performance_tracker.copy()
        
    def adjust_strategy_weights(self, performance_window: int = 100):
        """Automatically adjust strategy weights based on recent performance"""
        for strategy_name, perf in self.performance_tracker.items():
            if perf['total_signals'] >= 10:  # Minimum sample size
                # Weight based on win rate and average PnL
                base_weight = 1.0
                performance_multiplier = (perf['win_rate'] * 2) + (perf['avg_pnl'] / 100)  # Normalize PnL
                new_weight = max(0.1, min(2.0, base_weight * performance_multiplier))  # Cap between 0.1 and 2.0
                
                if abs(new_weight - self.strategy_weights.get(strategy_name, 1.0)) > 0.1:
                    logger.info(f"📊 Adjusting {strategy_name} weight: {self.strategy_weights.get(strategy_name, 1.0):.2f} → {new_weight:.2f}")
                    self.strategy_weights[strategy_name] = new_weight

# Factory function to create strategy manager with default strategies
def create_default_strategy_manager() -> StrategyManager:
    """Create strategy manager with default strategies"""
    manager = StrategyManager()
    
    # Add default strategies
    manager.add_strategy(TrendFollowingStrategy(), weight=1.2)
    manager.add_strategy(VolatilityBreakoutStrategy(), weight=1.0)
    manager.add_strategy(FundingSqueezeStrategy(), weight=0.8)
    manager.add_strategy(MeanReversionStrategy(), weight=0.9)
    
    return manager
