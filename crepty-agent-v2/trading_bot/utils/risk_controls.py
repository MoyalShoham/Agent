"""
Automated Risk Controls Module
Implements circuit breakers, trailing stops, and max drawdown protection.
"""
def check_circuit_breaker(pnl, max_drawdown, starting_balance=None):
    """
    If max_drawdown < 0: treat as absolute value (e.g., -1000 USDT)
    If max_drawdown > 0: treat as fraction of starting_balance (e.g., 0.2 for 20%)
    starting_balance must be provided for fractional drawdown.
    """
    if max_drawdown < 0:
        return pnl <= max_drawdown
    elif starting_balance is not None:
        return pnl <= -abs(max_drawdown) * starting_balance
    else:
        # If starting_balance not provided, fallback to old logic
        return pnl <= -abs(max_drawdown)

def trailing_stop(entry, current, trail_perc=0.02):
    return current < entry * (1 - trail_perc)
