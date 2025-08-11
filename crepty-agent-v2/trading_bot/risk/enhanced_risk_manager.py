"""
Enhanced Risk Manager - Advanced risk management with portfolio optimization,
correlation analysis, and dynamic position sizing.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import logging
from loguru import logger
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PositionInfo:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime

@dataclass
class RiskMetrics:
    portfolio_var: float
    portfolio_volatility: float
    max_correlation: float
    concentration_risk: float
    leverage_ratio: float
    drawdown: float

class EnhancedRiskManager:
    def __init__(self, 
                 max_portfolio_var: float = 0.02,
                 max_position_size: float = 0.1,
                 max_correlation: float = 0.7,
                 max_concentration: float = 0.3,
                 max_leverage: float = 3.0,
                 lookback_period: int = 252,
                 rebalance_threshold: float = 0.05):
        """
        Enhanced Risk Manager with portfolio optimization capabilities.
        
        Args:
            max_portfolio_var: Maximum portfolio Value-at-Risk (daily)
            max_position_size: Maximum position size as fraction of portfolio
            max_correlation: Maximum correlation between positions
            max_concentration: Maximum concentration in single asset
            max_leverage: Maximum portfolio leverage
            lookback_period: Days of historical data for calculations
            rebalance_threshold: Threshold for triggering rebalancing
        """
        self.max_portfolio_var = max_portfolio_var
        self.max_position_size = max_position_size
        self.max_correlation = max_correlation
        self.max_concentration = max_concentration
        self.max_leverage = max_leverage
        self.lookback_period = lookback_period
        self.rebalance_threshold = rebalance_threshold
        
        # Portfolio tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.historical_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_period))
        self.portfolio_value_history: deque = deque(maxlen=lookback_period)
        self.realized_pnl_history: deque = deque(maxlen=lookback_period)
        
        # Risk metrics
        self.current_var = 0.0
        self.current_volatility = 0.0
        self.max_drawdown = 0.0
        self.current_leverage = 0.0
        
        # Portfolio optimization
        self.optimal_weights: Dict[str, float] = {}
        self.last_optimization: Optional[datetime] = None
        
        logger.info("Enhanced Risk Manager initialized")
    
    def update_position(self, symbol: str, size: float, entry_price: float, 
                       current_price: float, realized_pnl: float = 0.0):
        """Update position information"""
        unrealized_pnl = (current_price - entry_price) * size
        
        self.positions[symbol] = PositionInfo(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            timestamp=datetime.now()
        )
        
        # Update price history
        self.historical_prices[symbol].append(current_price)
        
        # Update portfolio value
        portfolio_value = self.calculate_portfolio_value()
        self.portfolio_value_history.append(portfolio_value)
        
        if realized_pnl != 0:
            self.realized_pnl_history.append(realized_pnl)
        
        # Update risk metrics
        self._update_risk_metrics()
    
    def remove_position(self, symbol: str):
        """Remove a position from tracking"""
        if symbol in self.positions:
            del self.positions[symbol]
            self._update_risk_metrics()
    
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = 0.0
        for pos in self.positions.values():
            total_value += abs(pos.size) * pos.current_price
        return total_value
    
    def calculate_portfolio_pnl(self) -> Tuple[float, float]:
        """Calculate unrealized and realized PnL"""
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized = sum(self.realized_pnl_history) if self.realized_pnl_history else 0.0
        return unrealized, realized
    
    def calculate_var(self, confidence_level: float = 0.05) -> float:
        """Calculate portfolio Value-at-Risk"""
        if len(self.portfolio_value_history) < 30:
            return 0.0
        
        returns = pd.Series(self.portfolio_value_history).pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        return float(np.percentile(returns, confidence_level * 100))
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix of positions"""
        price_data = {}
        
        for symbol, prices in self.historical_prices.items():
            if len(prices) >= 30:  # Minimum data required
                price_data[symbol] = list(prices)
        
        if len(price_data) < 2:
            return pd.DataFrame()
        
        # Align series lengths
        min_length = min(len(prices) for prices in price_data.values())
        aligned_data = {symbol: prices[-min_length:] for symbol, prices in price_data.items()}
        
        df = pd.DataFrame(aligned_data)
        returns = df.pct_change().dropna()
        
        return returns.corr()
    
    def calculate_position_weights(self) -> Dict[str, float]:
        """Calculate current position weights"""
        total_value = self.calculate_portfolio_value()
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, pos in self.positions.items():
            weights[symbol] = abs(pos.size * pos.current_price) / total_value
        
        return weights
    
    def check_position_limits(self, symbol: str, proposed_size: float, 
                            price: float) -> Tuple[bool, str, float]:
        """
        Check if proposed position violates risk limits.
        
        Returns:
            (allowed, reason, max_allowed_size)
        """
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value == 0:
            portfolio_value = 100000  # Default for new portfolio
        
        proposed_value = abs(proposed_size * price)
        proposed_weight = proposed_value / portfolio_value
        
        # Check individual position size
        if proposed_weight > self.max_position_size:
            max_size = (self.max_position_size * portfolio_value) / price
            return False, f"Position size exceeds {self.max_position_size:.1%} limit", max_size
        
        # Check concentration risk
        current_weight = 0.0
        if symbol in self.positions:
            current_weight = abs(self.positions[symbol].size * price) / portfolio_value
        
        if proposed_weight > self.max_concentration:
            max_size = (self.max_concentration * portfolio_value) / price
            return False, f"Concentration exceeds {self.max_concentration:.1%} limit", max_size
        
        # Check correlation with existing positions
        if len(self.positions) > 0:
            corr_matrix = self.calculate_correlation_matrix()
            if not corr_matrix.empty and symbol in corr_matrix.columns:
                max_corr = corr_matrix[symbol].drop(symbol, errors='ignore').abs().max()
                if max_corr > self.max_correlation:
                    return False, f"Correlation {max_corr:.2f} exceeds {self.max_correlation:.2f} limit", 0.0
        
        # Check leverage
        total_leverage = self.calculate_leverage()
        additional_leverage = proposed_value / portfolio_value
        if total_leverage + additional_leverage > self.max_leverage:
            max_additional = self.max_leverage - total_leverage
            max_size = (max_additional * portfolio_value) / price
            return False, f"Leverage would exceed {self.max_leverage:.1f}x limit", max_size
        
        return True, "Position allowed", proposed_size
    
    def calculate_leverage(self) -> float:
        """Calculate current portfolio leverage"""
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value == 0:
            return 0.0
        
        # Assuming cash base of portfolio value (simplified)
        gross_exposure = sum(abs(pos.size * pos.current_price) for pos in self.positions.values())
        return gross_exposure / portfolio_value
    
    def calculate_optimal_weights(self) -> Dict[str, float]:
        """Calculate optimal portfolio weights using mean-variance optimization"""
        if len(self.positions) < 2:
            return {symbol: 1.0 for symbol in self.positions.keys()}
        
        # Get historical returns
        returns_data = {}
        for symbol, prices in self.historical_prices.items():
            if len(prices) >= 30:
                price_series = pd.Series(prices)
                returns = price_series.pct_change().dropna()
                returns_data[symbol] = returns
        
        if len(returns_data) < 2:
            equal_weight = 1.0 / len(self.positions)
            return {symbol: equal_weight for symbol in self.positions.keys()}
        
        # Align data
        min_length = min(len(returns) for returns in returns_data.values())
        aligned_returns = {symbol: returns.iloc[-min_length:] for symbol, returns in returns_data.items()}
        returns_df = pd.DataFrame(aligned_returns)
        
        # Calculate expected returns and covariance
        expected_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Simple mean-variance optimization (equal risk contribution)
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones((len(expected_returns), 1))
            weights = inv_cov @ ones
            weights = weights / weights.sum()
            
            optimal_weights = {}
            for i, symbol in enumerate(returns_df.columns):
                optimal_weights[symbol] = float(weights[i])
            
            return optimal_weights
            
        except np.linalg.LinAlgError:
            # Fallback to equal weights if matrix is singular
            equal_weight = 1.0 / len(returns_df.columns)
            return {symbol: equal_weight for symbol in returns_df.columns}
    
    def suggest_rebalancing(self) -> Dict[str, float]:
        """Suggest position adjustments for portfolio rebalancing"""
        current_weights = self.calculate_position_weights()
        optimal_weights = self.calculate_optimal_weights()
        
        suggestions = {}
        for symbol in set(list(current_weights.keys()) + list(optimal_weights.keys())):
            current_weight = current_weights.get(symbol, 0.0)
            optimal_weight = optimal_weights.get(symbol, 0.0)
            difference = optimal_weight - current_weight
            
            if abs(difference) > self.rebalance_threshold:
                suggestions[symbol] = difference
        
        return suggestions
    
    def _update_risk_metrics(self):
        """Update all risk metrics"""
        self.current_var = abs(self.calculate_var())
        
        if len(self.portfolio_value_history) > 1:
            returns = pd.Series(self.portfolio_value_history).pct_change().dropna()
            self.current_volatility = float(returns.std()) if len(returns) > 0 else 0.0
            
            # Calculate drawdown
            portfolio_values = pd.Series(self.portfolio_value_history)
            rolling_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - rolling_max) / rolling_max
            self.max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        self.current_leverage = self.calculate_leverage()
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        corr_matrix = self.calculate_correlation_matrix()
        max_correlation = 0.0
        if not corr_matrix.empty:
            # Get maximum off-diagonal correlation
            np.fill_diagonal(corr_matrix.values, 0)
            max_correlation = float(corr_matrix.abs().max().max())
        
        weights = self.calculate_position_weights()
        concentration_risk = max(weights.values()) if weights else 0.0
        
        return RiskMetrics(
            portfolio_var=self.current_var,
            portfolio_volatility=self.current_volatility,
            max_correlation=max_correlation,
            concentration_risk=concentration_risk,
            leverage_ratio=self.current_leverage,
            drawdown=self.max_drawdown
        )
    
    def check_risk_limits(self) -> List[str]:
        """Check all risk limits and return violations"""
        violations = []
        metrics = self.get_risk_metrics()
        
        if metrics.portfolio_var > self.max_portfolio_var:
            violations.append(f"Portfolio VaR {metrics.portfolio_var:.3f} exceeds limit {self.max_portfolio_var:.3f}")
        
        if metrics.max_correlation > self.max_correlation:
            violations.append(f"Max correlation {metrics.max_correlation:.3f} exceeds limit {self.max_correlation:.3f}")
        
        if metrics.concentration_risk > self.max_concentration:
            violations.append(f"Concentration {metrics.concentration_risk:.3f} exceeds limit {self.max_concentration:.3f}")
        
        if metrics.leverage_ratio > self.max_leverage:
            violations.append(f"Leverage {metrics.leverage_ratio:.1f} exceeds limit {self.max_leverage:.1f}")
        
        return violations
    
    def calculate_position_size(self, symbol: str, price: float, 
                              target_risk: float = None) -> float:
        """
        Calculate optimal position size based on portfolio risk management.
        
        Args:
            symbol: Trading symbol
            price: Current price
            target_risk: Target risk for this position (optional)
        
        Returns:
            Optimal position size in base currency
        """
        if target_risk is None:
            target_risk = self.max_position_size
        
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value == 0:
            portfolio_value = 100000  # Default starting value
        
        # Calculate volatility-adjusted position size
        if symbol in self.historical_prices and len(self.historical_prices[symbol]) >= 20:
            prices = pd.Series(self.historical_prices[symbol])
            returns = prices.pct_change().dropna()
            volatility = float(returns.std()) if len(returns) > 0 else 0.02  # Default 2% volatility
        else:
            volatility = 0.02  # Default volatility for new symbols
        
        # Kelly criterion inspired sizing
        # Position size = (target_risk * portfolio_value) / (volatility * price)
        base_size = (target_risk * portfolio_value) / price
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_adjustment = min(1.0, 0.02 / max(volatility, 0.005))  # Target 2% volatility
        adjusted_size = base_size * volatility_adjustment
        
        # Ensure position doesn't violate risk limits
        allowed, reason, max_size = self.check_position_limits(symbol, adjusted_size, price)
        
        if not allowed:
            logger.warning(f"Position size limited for {symbol}: {reason}")
            return max_size
        
        return adjusted_size
    
    def emergency_risk_check(self) -> bool:
        """Emergency risk check - should we stop all trading?"""
        violations = self.check_risk_limits()
        
        # Critical violations that require immediate action
        critical_conditions = [
            self.max_drawdown < -0.15,  # 15% drawdown
            self.current_var > self.max_portfolio_var * 2,  # 2x VaR limit
            self.current_leverage > self.max_leverage * 1.5  # 1.5x leverage limit
        ]
        
        if any(critical_conditions) or len(violations) >= 3:
            logger.critical(f"Emergency risk conditions detected: {violations}")
            return True
        
        return False
    
    def get_dashboard_data(self) -> Dict:
        """Get data for risk management dashboard"""
        metrics = self.get_risk_metrics()
        violations = self.check_risk_limits()
        
        return {
            'portfolio_value': self.calculate_portfolio_value(),
            'unrealized_pnl': self.calculate_portfolio_pnl()[0],
            'realized_pnl': self.calculate_portfolio_pnl()[1],
            'positions_count': len(self.positions),
            'metrics': metrics.__dict__,
            'violations': violations,
            'position_weights': self.calculate_position_weights(),
            'rebalancing_suggestions': self.suggest_rebalancing(),
            'last_update': datetime.now().isoformat()
        }
