"""
Advanced Risk Manager - Professional-grade risk management system
with portfolio optimization, VaR calculation, and dynamic risk adjustment.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from loguru import logger
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PositionInfo:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_score: float = 0.0

@dataclass
class RiskMetrics:
    portfolio_var: float
    portfolio_volatility: float
    max_correlation: float
    concentration_risk: float
    leverage_ratio: float
    drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_daily_loss: float
    position_count: int
    diversification_ratio: float

@dataclass
class RiskLimits:
    max_portfolio_var: float = 0.02
    max_position_size: float = 0.1
    max_correlation: float = 0.7
    max_concentration: float = 0.25
    max_leverage: float = 2.0
    max_drawdown: float = 0.15
    max_daily_loss: float = 0.05
    max_positions: int = 10
    min_diversification: float = 0.3

class AdvancedRiskManager:
    """
    Advanced risk management system with portfolio optimization,
    correlation analysis, and dynamic risk adjustment.
    """
    
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        
        # Portfolio tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.historical_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        self.portfolio_value_history: deque = deque(maxlen=252)
        self.realized_pnl_history: deque = deque(maxlen=1000)
        self.daily_pnl_history: deque = deque(maxlen=30)
        
        # Risk metrics
        self.current_var = 0.0
        self.current_volatility = 0.0
        self.max_drawdown = 0.0
        self.current_leverage = 0.0
        self.sharpe_ratio = 0.0
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Portfolio optimization
        self.optimal_weights: Dict[str, float] = {}
        self.last_optimization: Optional[datetime] = None
        self.risk_budget: Dict[str, float] = {}
        
        # Emergency controls
        self.emergency_mode = False
        self.last_emergency_check = datetime.now()
        
        logger.info("Advanced Risk Manager initialized with enhanced features")

    def update_position(self, symbol: str, size: float, entry_price: float, 
                       current_price: float, realized_pnl: float = 0.0):
        """Update position information with enhanced tracking"""
        unrealized_pnl = (current_price - entry_price) * size
        
        # Calculate position risk score
        if symbol in self.historical_prices and len(self.historical_prices[symbol]) > 1:
            price_history = list(self.historical_prices[symbol])
            returns = pd.Series([price_history[i]/price_history[i-1] - 1 
                               for i in range(1, len(price_history))])
            volatility = returns.std() * np.sqrt(252)  # Annualized
            risk_score = min(volatility * 10, 1.0)  # Normalize to 0-1
        else:
            risk_score = 0.5  # Default moderate risk
        
        self.positions[symbol] = PositionInfo(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            timestamp=datetime.now(),
            risk_score=risk_score
        )
        
        # Update price history
        self.historical_prices[symbol].append(current_price)
        
        # Update portfolio metrics
        self._update_risk_metrics()
        
        # Record daily PnL
        if realized_pnl != 0:
            self.realized_pnl_history.append(realized_pnl)
            self._update_daily_pnl(realized_pnl)

    def _update_daily_pnl(self, pnl: float):
        """Update daily PnL tracking"""
        today = datetime.now().date()
        if not self.daily_pnl_history or self.daily_pnl_history[-1]['date'] != today:
            self.daily_pnl_history.append({'date': today, 'pnl': pnl})
        else:
            self.daily_pnl_history[-1]['pnl'] += pnl

    def calculate_value_at_risk(self, confidence_level: float = 0.05, 
                               horizon_days: int = 1) -> float:
        """Calculate portfolio Value at Risk (VaR)"""
        if len(self.portfolio_value_history) < 30:
            return 0.0
        
        portfolio_values = np.array(list(self.portfolio_value_history))
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Scale to horizon
        scaled_returns = returns * np.sqrt(horizon_days)
        
        # Calculate VaR
        var = np.percentile(scaled_returns, confidence_level * 100)
        
        current_value = portfolio_values[-1] if len(portfolio_values) > 0 else 1.0
        return abs(var * current_value)

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix of position returns"""
        if len(self.positions) < 2:
            return pd.DataFrame()
        
        # Build returns matrix
        returns_data = {}
        min_length = float('inf')
        
        for symbol in self.positions.keys():
            if len(self.historical_prices[symbol]) > 1:
                prices = list(self.historical_prices[symbol])
                returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
                returns_data[symbol] = returns
                min_length = min(min_length, len(returns))
        
        if min_length < 10:  # Need sufficient data
            return pd.DataFrame()
        
        # Align lengths
        aligned_returns = {}
        for symbol, returns in returns_data.items():
            aligned_returns[symbol] = returns[-min_length:]
        
        df = pd.DataFrame(aligned_returns)
        correlation_matrix = df.corr()
        self.correlation_matrix = correlation_matrix
        
        return correlation_matrix

    def check_position_limits(self, symbol: str, proposed_size: float, 
                            price: float) -> Tuple[bool, str, float]:
        """Enhanced position limit checking"""
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value == 0:
            portfolio_value = 10000  # Default starting value
        
        proposed_value = abs(proposed_size) * price
        
        # Check position size limit
        position_weight = proposed_value / portfolio_value
        if position_weight > self.limits.max_position_size:
            max_size = (self.limits.max_position_size * portfolio_value) / price
            return False, f"Position size {position_weight:.1%} exceeds {self.limits.max_position_size:.1%} limit", max_size
        
        # Check concentration limit
        current_concentration = self._calculate_concentration_risk()
        if current_concentration + position_weight > self.limits.max_concentration:
            max_size = ((self.limits.max_concentration - current_concentration) * portfolio_value) / price
            return False, f"Would exceed concentration limit of {self.limits.max_concentration:.1%}", max_size
        
        # Check correlation limits
        if len(self.positions) > 0:
            corr_matrix = self.calculate_correlation_matrix()
            if not corr_matrix.empty and symbol in corr_matrix.columns:
                max_corr = corr_matrix[symbol].drop(symbol, errors='ignore').abs().max()
                if max_corr > self.limits.max_correlation:
                    return False, f"Correlation {max_corr:.2f} exceeds {self.limits.max_correlation:.2f} limit", 0.0
        
        # Check position count
        active_positions = sum(1 for pos in self.positions.values() if abs(pos.size) > 0.001)
        if active_positions >= self.limits.max_positions:
            return False, f"Maximum positions ({self.limits.max_positions}) reached", 0.0
        
        # Check leverage
        total_leverage = self.calculate_leverage()
        additional_leverage = proposed_value / portfolio_value
        if total_leverage + additional_leverage > self.limits.max_leverage:
            max_additional = self.limits.max_leverage - total_leverage
            max_size = (max_additional * portfolio_value) / price
            return False, f"Leverage would exceed {self.limits.max_leverage:.1f}x limit", max_size
        
        return True, "Position allowed", proposed_size

    def calculate_optimal_position_size(self, symbol: str, signal_strength: float, 
                                      confidence: float, price: float, 
                                      volatility: float = None) -> float:
        """Calculate optimal position size using Kelly Criterion and risk parity"""
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value == 0:
            return 0.0
        
        # Base size from Kelly Criterion
        if volatility is None:
            if symbol in self.historical_prices and len(self.historical_prices[symbol]) > 10:
                prices = list(self.historical_prices[symbol])
                returns = pd.Series([prices[i]/prices[i-1] - 1 for i in range(1, len(prices))])
                volatility = returns.std()
            else:
                volatility = 0.02  # Default 2% daily volatility
        
        # Kelly fraction: f = (p*b - q) / b
        # Approximate: win_prob = 0.5 + confidence/2, odds = signal_strength
        win_prob = 0.5 + (confidence * 0.3)  # Max 80% win probability
        loss_prob = 1 - win_prob
        
        # Expected return based on signal strength and confidence
        expected_return = signal_strength * confidence * 0.01  # 1% max expected return
        
        if expected_return <= 0 or volatility <= 0:
            return 0.0
        
        # Kelly fraction
        kelly_fraction = expected_return / (volatility ** 2)
        
        # Apply safety factor (use 25% of Kelly)
        safe_kelly = kelly_fraction * 0.25
        
        # Position size in USD
        position_size_usd = safe_kelly * portfolio_value
        
        # Apply position limits
        max_position_usd = self.limits.max_position_size * portfolio_value
        position_size_usd = min(position_size_usd, max_position_usd)
        
        # Convert to quantity
        quantity = position_size_usd / price
        
        # Apply minimum threshold
        min_position_usd = portfolio_value * 0.001  # 0.1% minimum
        if position_size_usd < min_position_usd:
            return 0.0
        
        return quantity

    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk"""
        if not self.positions:
            return 0.0
        
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value == 0:
            return 0.0
        
        position_weights = []
        for pos in self.positions.values():
            if abs(pos.size) > 0.001:
                weight = abs(pos.size * pos.current_price) / portfolio_value
                position_weights.append(weight)
        
        if not position_weights:
            return 0.0
        
        # Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in position_weights)
        return hhi

    def calculate_leverage(self) -> float:
        """Calculate current portfolio leverage"""
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value == 0:
            return 0.0
        
        total_exposure = sum(abs(pos.size * pos.current_price) 
                           for pos in self.positions.values())
        
        return total_exposure / portfolio_value

    def _update_risk_metrics(self):
        """Update all risk metrics"""
        self.current_var = self.calculate_value_at_risk()
        self.current_leverage = self.calculate_leverage()
        self.max_drawdown = self._calculate_max_drawdown()
        self.sharpe_ratio = self._calculate_sharpe_ratio()

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        values = np.array(list(self.portfolio_value_history))
        peaks = np.maximum.accumulate(values)
        drawdowns = (peaks - values) / peaks
        
        return float(np.max(drawdowns))

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.portfolio_value_history) < 30:
            return 0.0
        
        values = np.array(list(self.portfolio_value_history))
        returns = np.diff(values) / values[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - (risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns / np.std(returns)) * np.sqrt(252)  # Annualized

    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = sum(abs(pos.size) * pos.current_price 
                         for pos in self.positions.values())
        return total_value

    def emergency_risk_check(self) -> bool:
        """Enhanced emergency risk check"""
        if datetime.now() - self.last_emergency_check < timedelta(minutes=1):
            return self.emergency_mode
        
        self.last_emergency_check = datetime.now()
        
        # Critical risk conditions
        critical_conditions = [
            self.max_drawdown > self.limits.max_drawdown,
            self.current_var > self.limits.max_portfolio_var * 2,
            self.current_leverage > self.limits.max_leverage * 1.5,
            self._check_daily_loss_limit(),
            len(self.positions) > self.limits.max_positions * 1.2
        ]
        
        violation_count = sum(critical_conditions)
        
        if violation_count >= 2:  # Multiple violations
            self.emergency_mode = True
            logger.critical(f"EMERGENCY MODE ACTIVATED: {violation_count} critical violations detected")
            return True
        
        # Check for improvement to exit emergency mode
        if self.emergency_mode and violation_count == 0:
            self.emergency_mode = False
            logger.info("Emergency mode deactivated - risk conditions normalized")
        
        return self.emergency_mode

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        if not self.daily_pnl_history:
            return False
        
        today_pnl = self.daily_pnl_history[-1]['pnl'] if self.daily_pnl_history else 0
        portfolio_value = self.calculate_portfolio_value()
        
        if portfolio_value == 0:
            return False
        
        daily_loss_pct = abs(min(today_pnl, 0)) / portfolio_value
        return daily_loss_pct > self.limits.max_daily_loss

    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        corr_matrix = self.calculate_correlation_matrix()
        max_correlation = corr_matrix.abs().values.max() if not corr_matrix.empty else 0.0
        
        # Calculate diversification ratio
        if not corr_matrix.empty and len(corr_matrix) > 1:
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            diversification_ratio = 1 - avg_correlation
        else:
            diversification_ratio = 1.0
        
        return RiskMetrics(
            portfolio_var=self.current_var,
            portfolio_volatility=self.current_volatility,
            max_correlation=max_correlation,
            concentration_risk=self._calculate_concentration_risk(),
            leverage_ratio=self.current_leverage,
            drawdown=self.max_drawdown,
            sharpe_ratio=self.sharpe_ratio,
            sortino_ratio=self._calculate_sortino_ratio(),
            calmar_ratio=self._calculate_calmar_ratio(),
            max_daily_loss=self._get_max_daily_loss(),
            position_count=len([p for p in self.positions.values() if abs(p.size) > 0.001]),
            diversification_ratio=diversification_ratio
        )

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(self.portfolio_value_history) < 30:
            return 0.0
        
        values = np.array(list(self.portfolio_value_history))
        returns = np.diff(values) / values[:-1]
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - (0.02 / 252)  # Risk-free rate
        return (excess_returns / downside_deviation) * np.sqrt(252)

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        if self.max_drawdown == 0:
            return 0.0
        
        if len(self.portfolio_value_history) < 252:  # Need at least 1 year
            return 0.0
        
        values = np.array(list(self.portfolio_value_history))
        annual_return = (values[-1] / values[0]) ** (252 / len(values)) - 1
        
        return annual_return / self.max_drawdown

    def _get_max_daily_loss(self) -> float:
        """Get maximum daily loss from recent history"""
        if not self.daily_pnl_history:
            return 0.0
        
        daily_losses = [day['pnl'] for day in self.daily_pnl_history if day['pnl'] < 0]
        return abs(min(daily_losses)) if daily_losses else 0.0

    def optimize_portfolio(self) -> Dict[str, float]:
        """Optimize portfolio allocation using modern portfolio theory"""
        if len(self.positions) < 2:
            return {}
        
        try:
            # Get correlation matrix
            corr_matrix = self.calculate_correlation_matrix()
            if corr_matrix.empty:
                return {}
            
            symbols = list(corr_matrix.columns)
            n_assets = len(symbols)
            
            # Calculate expected returns (simplified)
            expected_returns = []
            for symbol in symbols:
                if len(self.historical_prices[symbol]) > 10:
                    prices = list(self.historical_prices[symbol])
                    returns = [(prices[i]/prices[i-1] - 1) for i in range(1, len(prices))]
                    expected_returns.append(np.mean(returns))
                else:
                    expected_returns.append(0.001)  # Default 0.1% daily return
            
            expected_returns = np.array(expected_returns)
            
            # Objective function: minimize risk for given return
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(corr_matrix.values, weights))
                # Risk-adjusted return (Sharpe-like)
                return -portfolio_return / np.sqrt(portfolio_variance)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
            ]
            
            # Bounds (no short selling, max 30% per position)
            bounds = [(0.0, 0.3) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                self.optimal_weights = optimal_weights
                self.last_optimization = datetime.now()
                return optimal_weights
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
        
        return {}

    def get_position_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """Get position sizing recommendations"""
        recommendations = {}
        
        for symbol, position in self.positions.items():
            if abs(position.size) < 0.001:
                continue
            
            # Calculate recommended adjustments
            current_weight = abs(position.size * position.current_price) / self.calculate_portfolio_value()
            optimal_weight = self.optimal_weights.get(symbol, current_weight)
            
            recommendation = {
                'current_weight': current_weight,
                'optimal_weight': optimal_weight,
                'adjustment': optimal_weight - current_weight,
                'risk_score': position.risk_score,
                'action': 'hold'
            }
            
            # Determine action
            if abs(recommendation['adjustment']) > 0.05:  # 5% threshold
                if recommendation['adjustment'] > 0:
                    recommendation['action'] = 'increase'
                else:
                    recommendation['action'] = 'decrease'
            
            recommendations[symbol] = recommendation
        
        return recommendations

    def should_allow_trade(self, symbol: str, size: float, price: float, 
                          confidence: float = 0.5) -> Tuple[bool, str]:
        """Determine if a trade should be allowed"""
        # Emergency mode check
        if self.emergency_risk_check():
            return False, "Emergency mode active - all trading suspended"
        
        # Position limits check
        allowed, reason, _ = self.check_position_limits(symbol, size, price)
        if not allowed:
            return False, reason
        
        # Confidence threshold
        if confidence < 0.3:
            return False, f"Signal confidence {confidence:.1%} below minimum threshold"
        
        # Daily loss limit
        if self._check_daily_loss_limit():
            return False, "Daily loss limit exceeded"
        
        return True, "Trade approved"

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for risk dashboard"""
        metrics = self.get_risk_metrics()
        
        return {
            'risk_metrics': metrics,
            'positions': {
                symbol: {
                    'size': pos.size,
                    'value': abs(pos.size * pos.current_price),
                    'pnl': pos.unrealized_pnl + pos.realized_pnl,
                    'risk_score': pos.risk_score
                } for symbol, pos in self.positions.items()
            },
            'portfolio_value': self.calculate_portfolio_value(),
            'emergency_mode': self.emergency_mode,
            'daily_pnl': self.daily_pnl_history[-1]['pnl'] if self.daily_pnl_history else 0,
            'var_95': self.current_var,
            'recommendations': self.get_position_recommendations(),
            'limits': {
                'max_var': self.limits.max_portfolio_var,
                'max_drawdown': self.limits.max_drawdown,
                'max_leverage': self.limits.max_leverage,
                'max_positions': self.limits.max_positions,
                'max_daily_loss': self.limits.max_daily_loss
            }
        }
