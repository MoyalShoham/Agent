"""
Emergency Risk Controls - Advanced safety mechanisms for high-risk trading environments.
Implements circuit breakers, dynamic position sizing, and emergency shutdown protocols.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from loguru import logger
import os
import json

@dataclass
class EmergencyConfig:
    """Emergency control configuration"""
    max_daily_loss_pct: float = 0.03  # 3% max daily loss
    max_drawdown_pct: float = 0.10    # 10% max drawdown
    min_confidence_threshold: float = 0.6  # 60% minimum signal confidence
    max_position_count: int = 8       # Maximum concurrent positions
    volatility_threshold: float = 0.05  # 5% volatility threshold
    cooldown_period_minutes: int = 60  # 1 hour cooldown after emergency
    max_trades_per_hour: int = 10     # Rate limiting
    correlation_limit: float = 0.8    # Maximum position correlation
    emergency_stop_loss_pct: float = 0.02  # 2% emergency stop loss

@dataclass
class RiskEvent:
    """Risk event tracking"""
    timestamp: datetime
    event_type: str
    symbol: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    action_taken: str
    value: float = 0.0

class EmergencyRiskControls:
    """
    Advanced emergency risk control system with real-time monitoring
    and automatic safety interventions.
    """
    
    def __init__(self, config: EmergencyConfig = None):
        self.config = config or EmergencyConfig()
        
        # State tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_drawdown = 0.0
        self.portfolio_value = 0.0
        self.positions: Dict[str, Dict] = {}
        
        # Risk event tracking
        self.risk_events: deque = deque(maxlen=1000)
        self.trade_history: deque = deque(maxlen=500)
        self.hourly_trades: deque = deque(maxlen=24)  # Last 24 hours
        
        # Emergency state
        self.emergency_mode = False
        self.emergency_start_time: Optional[datetime] = None
        self.symbol_cooldowns: Dict[str, datetime] = {}
        self.blocked_symbols: set = set()
        
        # Performance tracking
        self.peak_portfolio_value = 0.0
        self.daily_start_value = 0.0
        self.last_daily_reset = datetime.now().date()
        
        # Configuration persistence
        self.state_file = 'emergency_risk_state.json'
        self.load_state()
        
        logger.info("Emergency Risk Controls initialized")

    def record_trade(self, symbol: str, signal: str, pnl: float, 
                    confidence: float = 0.5, price: float = 0.0):
        """Record a trade and update risk metrics"""
        try:
            trade_time = datetime.now()
            
            # Update daily PnL
            self.daily_pnl += pnl
            self.daily_trades += 1
            
            # Record trade
            trade_record = {
                'timestamp': trade_time,
                'symbol': symbol,
                'signal': signal,
                'pnl': pnl,
                'confidence': confidence,
                'price': price
            }
            
            self.trade_history.append(trade_record)
            
            # Update hourly trade count
            current_hour = trade_time.replace(minute=0, second=0, microsecond=0)
            
            # Clean old hourly records
            self.hourly_trades = deque([
                record for record in self.hourly_trades 
                if record['hour'] > current_hour - timedelta(hours=24)
            ], maxlen=24)
            
            # Add current hour trade
            hour_record = next((r for r in self.hourly_trades if r['hour'] == current_hour), None)
            if hour_record:
                hour_record['count'] += 1
            else:
                self.hourly_trades.append({'hour': current_hour, 'count': 1})
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Check for risk events
            self._check_risk_events(symbol, signal, pnl, confidence)
            
            # Daily reset check
            self._check_daily_reset()
            
            logger.debug(f"Trade recorded: {symbol} {signal} PnL: {pnl:.4f}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")

    def should_allow_trade(self, symbol: str, confidence: float, 
                          signal_strength: float = 1.0) -> Tuple[bool, str]:
        """Determine if a trade should be allowed based on risk controls"""
        try:
            # Emergency mode check
            if self.emergency_mode:
                if not self._can_exit_emergency():
                    return False, "Emergency mode active - trading suspended"
            
            # Symbol cooldown check
            if symbol in self.symbol_cooldowns:
                cooldown_end = self.symbol_cooldowns[symbol]
                if datetime.now() < cooldown_end:
                    remaining = cooldown_end - datetime.now()
                    return False, f"Symbol {symbol} in cooldown for {remaining.seconds//60}m"
            
            # Blocked symbols check
            if symbol in self.blocked_symbols:
                return False, f"Symbol {symbol} is blocked due to previous issues"
            
            # Confidence threshold
            if confidence < self.config.min_confidence_threshold:
                return False, f"Signal confidence {confidence:.1%} below threshold {self.config.min_confidence_threshold:.1%}"
            
            # Daily loss limit
            if self._check_daily_loss_limit():
                return False, f"Daily loss limit exceeded: {self.daily_pnl:.4f}"
            
            # Drawdown limit
            if self.current_drawdown > self.config.max_drawdown_pct:
                return False, f"Drawdown limit exceeded: {self.current_drawdown:.1%}"
            
            # Position count limit
            active_positions = len([p for p in self.positions.values() if p.get('size', 0) != 0])
            if active_positions >= self.config.max_position_count:
                return False, f"Maximum positions ({self.config.max_position_count}) reached"
            
            # Rate limiting
            current_hour_trades = self._get_current_hour_trades()
            if current_hour_trades >= self.config.max_trades_per_hour:
                return False, f"Hourly trade limit ({self.config.max_trades_per_hour}) exceeded"
            
            # Volatility check (if available)
            if self._is_high_volatility_period():
                if confidence < 0.8:  # Higher confidence required in volatile periods
                    return False, "High volatility detected - higher confidence required"
            
            return True, "Trade approved"
            
        except Exception as e:
            logger.error(f"Error in trade approval check: {e}")
            return False, f"Error in risk check: {str(e)}"

    def adjust_position_size(self, base_size: float, symbol: str = "", 
                           confidence: float = 0.5) -> float:
        """Adjust position size based on current risk conditions"""
        try:
            adjusted_size = base_size
            
            # Emergency mode reduction
            if self.emergency_mode:
                adjusted_size *= 0.5
                logger.info("Emergency mode: Position size reduced by 50%")
            
            # Daily loss adjustment
            daily_loss_pct = abs(min(self.daily_pnl, 0)) / max(self.portfolio_value, 1)
            if daily_loss_pct > 0.01:  # 1% daily loss
                loss_factor = 1 - min(daily_loss_pct * 2, 0.8)  # Max 80% reduction
                adjusted_size *= loss_factor
                logger.info(f"Daily loss adjustment: Size reduced by {(1-loss_factor)*100:.1f}%")
            
            # Drawdown adjustment
            if self.current_drawdown > 0.05:  # 5% drawdown threshold
                dd_factor = 1 - min(self.current_drawdown, 0.5)  # Max 50% reduction
                adjusted_size *= dd_factor
                logger.info(f"Drawdown adjustment: Size reduced by {(1-dd_factor)*100:.1f}%")
            
            # Confidence-based adjustment
            if confidence < 0.7:
                conf_factor = confidence / 0.7  # Scale based on confidence
                adjusted_size *= conf_factor
                logger.info(f"Confidence adjustment: Size adjusted by {conf_factor:.1%}")
            
            # Volatility adjustment
            if self._is_high_volatility_period():
                adjusted_size *= 0.7  # Reduce size by 30% in high volatility
                logger.info("High volatility: Position size reduced by 30%")
            
            # Ensure minimum viable size
            min_size = base_size * 0.1  # Never go below 10% of original
            adjusted_size = max(adjusted_size, min_size)
            
            # Ensure maximum size doesn't exceed limits
            max_size = base_size * 1.5  # Never exceed 150% of original
            adjusted_size = min(adjusted_size, max_size)
            
            size_change = (adjusted_size / base_size - 1) * 100
            if abs(size_change) > 1:  # Only log significant changes
                logger.info(f"Position size adjusted: {size_change:+.1f}% (from {base_size:.4f} to {adjusted_size:.4f})")
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error adjusting position size: {e}")
            return base_size * 0.5  # Conservative fallback

    def _check_risk_events(self, symbol: str, signal: str, pnl: float, confidence: float):
        """Check for and record risk events"""
        events = []
        
        # Large loss event
        if pnl < -100:  # Loss > $100
            events.append(RiskEvent(
                timestamp=datetime.now(),
                event_type='large_loss',
                symbol=symbol,
                severity='high' if pnl < -500 else 'medium',
                description=f'Large loss detected: ${pnl:.2f}',
                action_taken='Position size reduction triggered',
                value=pnl
            ))
        
        # Low confidence signal
        if confidence < 0.4:
            events.append(RiskEvent(
                timestamp=datetime.now(),
                event_type='low_confidence',
                symbol=symbol,
                severity='medium',
                description=f'Low confidence signal: {confidence:.1%}',
                action_taken='Signal confidence warning',
                value=confidence
            ))
        
        # Daily loss threshold
        if self.daily_pnl < -1000:  # Daily loss > $1000
            events.append(RiskEvent(
                timestamp=datetime.now(),
                event_type='daily_loss_threshold',
                symbol='PORTFOLIO',
                severity='critical',
                description=f'Daily loss threshold exceeded: ${self.daily_pnl:.2f}',
                action_taken='Emergency protocols activated',
                value=self.daily_pnl
            ))
            self._activate_emergency_mode('daily_loss_threshold')
        
        # Record events
        for event in events:
            self.risk_events.append(event)
            self._log_risk_event(event)

    def _log_risk_event(self, event: RiskEvent):
        """Log risk event with appropriate severity"""
        message = f"[{event.severity.upper()}] {event.event_type}: {event.description} | Action: {event.action_taken}"
        
        if event.severity == 'critical':
            logger.critical(message)
        elif event.severity == 'high':
            logger.error(message)
        elif event.severity == 'medium':
            logger.warning(message)
        else:
            logger.info(message)

    def _activate_emergency_mode(self, trigger: str):
        """Activate emergency mode with specific trigger"""
        if not self.emergency_mode:
            self.emergency_mode = True
            self.emergency_start_time = datetime.now()
            
            logger.critical(f"🚨 EMERGENCY MODE ACTIVATED - Trigger: {trigger}")
            logger.critical("🛑 All trading suspended - Risk limits exceeded")
            
            # Additional emergency actions
            self._emergency_position_review()
            
            # Save state
            self.save_state()

    def _emergency_position_review(self):
        """Review all positions during emergency and apply emergency stops if needed"""
        try:
            for symbol, position in self.positions.items():
                if position.get('size', 0) != 0:
                    # Apply emergency stop loss if position is significantly negative
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    position_value = abs(position.get('size', 0)) * position.get('price', 1)
                    
                    if position_value > 0:
                        loss_pct = abs(min(unrealized_pnl, 0)) / position_value
                        
                        if loss_pct > self.config.emergency_stop_loss_pct:
                            logger.critical(f"Emergency stop loss triggered for {symbol}: {loss_pct:.1%} loss")
                            # Would trigger position closure in actual implementation
                            
        except Exception as e:
            logger.error(f"Error in emergency position review: {e}")

    def _can_exit_emergency(self) -> bool:
        """Check if conditions allow exiting emergency mode"""
        if not self.emergency_mode:
            return True
        
        # Must wait minimum cooldown period
        min_cooldown = timedelta(minutes=self.config.cooldown_period_minutes)
        if datetime.now() - self.emergency_start_time < min_cooldown:
            return False
        
        # Risk conditions must improve
        conditions_met = [
            self.daily_pnl > -500,  # Daily loss improved
            self.current_drawdown < 0.08,  # Drawdown under 8%
            len(self.risk_events) == 0 or self.risk_events[-1].timestamp < datetime.now() - timedelta(minutes=30)
        ]
        
        return all(conditions_met)

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        if self.portfolio_value <= 0:
            return False
        
        daily_loss_pct = abs(min(self.daily_pnl, 0)) / self.portfolio_value
        return daily_loss_pct > self.config.max_daily_loss_pct

    def _get_current_hour_trades(self) -> int:
        """Get number of trades in current hour"""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        for record in self.hourly_trades:
            if record['hour'] == current_hour:
                return record['count']
        
        return 0

    def _is_high_volatility_period(self) -> bool:
        """Detect if we're in a high volatility period"""
        if len(self.trade_history) < 10:
            return False
        
        try:
            # Calculate recent PnL volatility
            recent_pnls = [trade['pnl'] for trade in list(self.trade_history)[-20:]]
            volatility = np.std(recent_pnls) if len(recent_pnls) > 1 else 0
            
            # Normalize by typical trade size
            avg_trade_size = np.mean([abs(pnl) for pnl in recent_pnls]) if recent_pnls else 1
            normalized_volatility = volatility / max(avg_trade_size, 1)
            
            return normalized_volatility > self.config.volatility_threshold
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return False

    def _update_portfolio_metrics(self):
        """Update portfolio-level risk metrics"""
        try:
            # Update peak portfolio value
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
            
            # Calculate current drawdown
            if self.peak_portfolio_value > 0:
                self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")

    def _check_daily_reset(self):
        """Check if we need to reset daily counters"""
        today = datetime.now().date()
        
        if today != self.last_daily_reset:
            logger.info(f"Daily reset: Previous day PnL: ${self.daily_pnl:.2f}, Trades: {self.daily_trades}")
            
            # Reset daily counters
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_start_value = self.portfolio_value
            self.last_daily_reset = today
            
            # Clear expired cooldowns
            current_time = datetime.now()
            self.symbol_cooldowns = {
                symbol: cooldown_time 
                for symbol, cooldown_time in self.symbol_cooldowns.items()
                if cooldown_time > current_time
            }
            
            self.save_state()

    def get_status(self) -> Dict:
        """Get comprehensive emergency control status"""
        return {
            'emergency_mode': self.emergency_mode,
            'emergency_duration_minutes': (
                (datetime.now() - self.emergency_start_time).total_seconds() / 60
                if self.emergency_start_time else 0
            ),
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'current_drawdown': self.current_drawdown,
            'portfolio_value': self.portfolio_value,
            'active_positions': len([p for p in self.positions.values() if p.get('size', 0) != 0]),
            'blocked_symbols': list(self.blocked_symbols),
            'active_cooldowns': {
                symbol: (cooldown_time - datetime.now()).total_seconds() / 60
                for symbol, cooldown_time in self.symbol_cooldowns.items()
                if cooldown_time > datetime.now()
            },
            'recent_risk_events': len([
                event for event in self.risk_events
                if event.timestamp > datetime.now() - timedelta(hours=1)
            ]),
            'hourly_trade_count': self._get_current_hour_trades(),
            'high_volatility': self._is_high_volatility_period(),
            'can_exit_emergency': self._can_exit_emergency() if self.emergency_mode else True
        }

    def save_state(self):
        """Save emergency control state to file"""
        try:
            state = {
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'portfolio_value': self.portfolio_value,
                'peak_portfolio_value': self.peak_portfolio_value,
                'emergency_mode': self.emergency_mode,
                'emergency_start_time': self.emergency_start_time.isoformat() if self.emergency_start_time else None,
                'last_daily_reset': self.last_daily_reset.isoformat(),
                'blocked_symbols': list(self.blocked_symbols),
                'symbol_cooldowns': {
                    symbol: cooldown_time.isoformat()
                    for symbol, cooldown_time in self.symbol_cooldowns.items()
                }
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving emergency state: {e}")

    def load_state(self):
        """Load emergency control state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.daily_pnl = state.get('daily_pnl', 0.0)
                self.daily_trades = state.get('daily_trades', 0)
                self.portfolio_value = state.get('portfolio_value', 0.0)
                self.peak_portfolio_value = state.get('peak_portfolio_value', 0.0)
                self.emergency_mode = state.get('emergency_mode', False)
                
                if state.get('emergency_start_time'):
                    self.emergency_start_time = datetime.fromisoformat(state['emergency_start_time'])
                
                if state.get('last_daily_reset'):
                    self.last_daily_reset = datetime.fromisoformat(state['last_daily_reset']).date()
                
                self.blocked_symbols = set(state.get('blocked_symbols', []))
                
                # Load cooldowns
                cooldowns = state.get('symbol_cooldowns', {})
                self.symbol_cooldowns = {
                    symbol: datetime.fromisoformat(cooldown_time)
                    for symbol, cooldown_time in cooldowns.items()
                }
                
                logger.info("Emergency control state loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading emergency state: {e}")

# Global instance
emergency_controls = EmergencyRiskControls()
