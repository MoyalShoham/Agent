"""
Adaptive Position Sizing Module
Dynamically adjusts position size based on volatility, drawdown, and win rate.
"""
import numpy as np

def calculate_position_size(balance, volatility, max_drawdown, win_rate, base_frac=0.7, min_frac=0.01, max_frac=1.0):
    # Example: reduce size if volatility or drawdown is high, increase if win rate is high
    risk_factor = 1.0
    if volatility > 0.03:
        risk_factor *= 0.5
    if max_drawdown > 0.1:
        risk_factor *= 0.5
    if win_rate < 0.5:
        risk_factor *= 0.7
    elif win_rate > 0.7:
        risk_factor *= 1.2
    frac = np.clip(base_frac * risk_factor, min_frac, max_frac)
    return balance * frac
