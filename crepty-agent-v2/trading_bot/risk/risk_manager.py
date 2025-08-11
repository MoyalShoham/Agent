import numpy as np
import logging
from collections import deque

class RiskManager:
    def __init__(self, max_drawdown=0.10, window=50, min_position=0.1, default_position=1.0):
        """
        max_drawdown: Maximum allowed drawdown (fraction, e.g., 0.10 for 10%)
        window: Number of recent trades to consider
        min_position: Minimum allowed position size (fraction of default)
        default_position: Default position size (fraction, e.g., 1.0 for 100%)
        """
        self.max_drawdown = max_drawdown
        self.window = window
        self.min_position = min_position
        self.default_position = default_position
        self.pnl_history = deque(maxlen=window)
        self.equity_curve = [1.0]  # Start with 1.0 as initial equity
        self.logger = logging.getLogger("RiskManager")

    def update(self, realized_pnl):
        self.pnl_history.append(realized_pnl)
        new_equity = self.equity_curve[-1] + realized_pnl
        self.equity_curve.append(new_equity)

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
        return True

    def reset(self):
        self.pnl_history.clear()
        self.equity_curve = [1.0]

# Example usage:
# risk_mgr = RiskManager(max_drawdown=0.10)
# for trade in trades:
#     risk_mgr.update(trade['realized_pnl'])
#     if risk_mgr.allow_trade():
#         pos_size = risk_mgr.get_position_size()
#         ...
