"""
Automated Risk Controls Module
Implements circuit breakers, trailing stops, and max drawdown protection.
"""
def check_circuit_breaker(pnl, max_drawdown):
    # Example: pause trading if drawdown exceeds threshold
    return max_drawdown > 0.2

def trailing_stop(entry, current, trail_perc=0.02):
    return current < entry * (1 - trail_perc)
