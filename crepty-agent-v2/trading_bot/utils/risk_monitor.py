"""
RiskMonitor - Monitors and adapts risk dynamically based on drawdown, volatility, and win rate.
"""
import numpy as np

class RiskMonitor:
    def __init__(self, window=50, min_position=0.1, max_position=0.8):
        self.pnl_history = []
        self.window = window
        self.min_position = min_position
        self.max_position = max_position
        self.last_position_size = max_position

    def update(self, trade_pnl):
        self.pnl_history.append(trade_pnl)
        if len(self.pnl_history) > self.window:
            self.pnl_history.pop(0)

    def get_drawdown(self):
        if not self.pnl_history:
            return 0
        cum_pnl = np.cumsum(self.pnl_history)
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = (peak - cum_pnl).max()
        return drawdown

    def get_volatility(self):
        if len(self.pnl_history) < 2:
            return 0
        return float(np.std(self.pnl_history[-self.window:]))

    def adapt_position_size(self):
        drawdown = self.get_drawdown()
        vol = self.get_volatility()
        # Example: scale down if drawdown or volatility is high
        if drawdown > 100 or vol > 50:
            self.last_position_size = max(self.min_position, self.last_position_size * 0.8)
        else:
            self.last_position_size = min(self.max_position, self.last_position_size * 1.05)
        return self.last_position_size
