import numpy as np
import pandas as pd
import logging
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PositionInfo:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float

@dataclass
class RiskMetrics:
    portfolio_var: float
    max_correlation: float
    concentration_risk: float
    leverage_ratio: float
    drawdown: float

class RiskManager:
    def __init__(self, max_drawdown=0.10, window=50, min_position=0.1, default_position=1.0):
        """
        Enhanced Risk Manager with portfolio optimization capabilities.
        
        Args:
            max_drawdown: Maximum allowed drawdown (fraction, e.g., 0.10 for 10%)
            window: Number of recent trades to consider
            min_position: Minimum allowed position size (fraction of default)
            default_position: Default position size (fraction, e.g., 1.0 for 100%)
        """
        self.max_drawdown = max_drawdown
        self.window = window
        self.min_position = min_position
        # 🔧 REDUCED DEFAULT POSITION SIZE - From 1.0 to 0.3 (30% of original)
        self.default_position = default_position * 0.3
        
        # Enhanced features - 🔧 MORE CONSERVATIVE LIMITS
        self.max_portfolio_var = 0.01  # Reduced from 0.02 to 0.01
        self.max_position_size = 0.05  # Reduced from 0.1 to 0.05
        self.max_correlation = 0.5     # Reduced from 0.7 to 0.5
        self.max_concentration = 0.2   # Reduced from 0.3 to 0.2
        self.max_leverage = 1.5        # Reduced from 3.0 to 1.5
        self.max_concurrent_positions = 5  # 🔧 NEW: Limit concurrent positions
        
        # Portfolio tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.pnl_history = deque(maxlen=window)
        self.equity_curve = [1.0]  # Start with 1.0 as initial equity
        self.historical_prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        
        # Risk metrics
        self.current_var = 0.0
        self.current_leverage = 0.0
        
        self.logger = logging.getLogger("RiskManager")

    def update(self, realized_pnl):
        """Update with realized PnL"""
        self.pnl_history.append(realized_pnl)
        new_equity = self.equity_curve[-1] + realized_pnl
        self.equity_curve.append(new_equity)

    def update_position(self, symbol: str, size: float, entry_price: float, 
                       current_price: float):
        """Update position information"""
        unrealized_pnl = (current_price - entry_price) * size
        
        self.positions[symbol] = PositionInfo(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl
        )
        
        # Update price history
        self.historical_prices[symbol].append(current_price)
        
        # Update risk metrics
        self._update_risk_metrics()

    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = 0.0
        for pos in self.positions.values():
            total_value += abs(pos.size) * pos.current_price
        return total_value

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
        
        return True, "Position allowed", proposed_size

    def calculate_leverage(self) -> float:
        """Calculate current portfolio leverage"""
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value == 0:
            return 0.0
        
        gross_exposure = sum(abs(pos.size * pos.current_price) for pos in self.positions.values())
        return gross_exposure / portfolio_value

    def _update_risk_metrics(self):
        """Update all risk metrics"""
        # Update leverage
        self.current_leverage = self.calculate_leverage()
        
        # Update VaR (simplified)
        if len(self.pnl_history) >= 30:
            returns = list(self.pnl_history)
            self.current_var = abs(np.percentile(returns, 5))  # 5% VaR

    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        corr_matrix = self.calculate_correlation_matrix()
        max_correlation = 0.0
        if not corr_matrix.empty:
            np.fill_diagonal(corr_matrix.values, 0)
            max_correlation = float(corr_matrix.abs().max().max())
        
        # Calculate concentration risk
        weights = {}
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value > 0:
            for symbol, pos in self.positions.items():
                weights[symbol] = abs(pos.size * pos.current_price) / portfolio_value
        
        concentration_risk = max(weights.values()) if weights else 0.0
        
        return RiskMetrics(
            portfolio_var=self.current_var,
            max_correlation=max_correlation,
            concentration_risk=concentration_risk,
            leverage_ratio=self.current_leverage,
            drawdown=self.get_drawdown()
        )

    def get_drawdown(self):
        curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(curve)
        drawdown = (peak - curve) / peak
        return float(np.max(drawdown)) if len(drawdown) > 1 else 0.0

    def get_position_size(self):
        drawdown = self.get_drawdown()
        if drawdown > self.max_drawdown:
            self.logger.warning(f"Drawdown {drawdown:.2%} exceeds max {self.max_drawdown:.2%}. Reducing position size.")
            return self.min_position
        return self.default_position

    def allow_trade(self):
        drawdown = self.get_drawdown()
        if drawdown > self.max_drawdown * 1.5:
            self.logger.error(f"Drawdown {drawdown:.2%} critical. Blocking new trades.")
            return False
        
        # 🔧 NEW: Check position count limit
        active_positions = sum(1 for pos in self.positions.values() if abs(pos.size) > 0.001)
        if active_positions >= self.max_concurrent_positions:
            self.logger.warning(f"Max positions ({self.max_concurrent_positions}) reached. Blocking new trades.")
            return False
            
        return True

    def calculate_position_size_for_symbol(self, symbol: str, price: float, 
                                         target_risk: float = None) -> float:
        """Calculate optimal position size based on portfolio risk management"""
        if target_risk is None:
            target_risk = self.max_position_size
        
        # 🔧 CHECK: Position count limit
        active_positions = sum(1 for pos in self.positions.values() if abs(pos.size) > 0.001)
        if active_positions >= self.max_concurrent_positions:
            self.logger.warning(f"Position limit reached ({active_positions}/{self.max_concurrent_positions})")
            return 0.0
        
        portfolio_value = self.calculate_portfolio_value()
        if portfolio_value == 0:
            portfolio_value = 100000  # Default starting value
        
        # Calculate volatility-adjusted position size
        if symbol in self.historical_prices and len(self.historical_prices[symbol]) >= 20:
            prices = pd.Series(self.historical_prices[symbol])
            returns = prices.pct_change().dropna()
            volatility = float(returns.std()) if len(returns) > 0 else 0.02
        else:
            volatility = 0.02  # Default volatility
        
        # Base position size - 🔧 REDUCED by 70%
        base_size = (target_risk * portfolio_value * 0.3) / price  # Apply 30% reduction
        
        # Adjust for volatility
        volatility_adjustment = min(1.0, 0.02 / max(volatility, 0.005))
        adjusted_size = base_size * volatility_adjustment
        
        # Check limits
        allowed, reason, max_size = self.check_position_limits(symbol, adjusted_size, price)
        
        if not allowed:
            self.logger.warning(f"Position size limited for {symbol}: {reason}")
            return max_size
        
        return adjusted_size

    def emergency_risk_check(self) -> bool:
        """Emergency risk check - should we stop all trading?"""
        critical_conditions = [
            self.get_drawdown() > 0.15,  # 15% drawdown
            self.current_var > self.max_portfolio_var * 2,  # 2x VaR limit
            self.current_leverage > self.max_leverage * 1.5  # 1.5x leverage limit
        ]
        
        if any(critical_conditions):
            self.logger.critical("Emergency risk conditions detected")
            return True
        
        return False

    def reset(self):
        self.pnl_history.clear()
        self.equity_curve = [1.0]
        self.positions.clear()

# Example usage:
# risk_mgr = RiskManager(max_drawdown=0.10)
# for trade in trades:
#     risk_mgr.update(trade['realized_pnl'])
#     if risk_mgr.allow_trade():
#         pos_size = risk_mgr.get_position_size()
#         ...
