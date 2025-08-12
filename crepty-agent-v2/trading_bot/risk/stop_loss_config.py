#!/usr/bin/env python3
"""
Enhanced Stop Loss Configuration
"""

# Emergency stop-loss settings to limit losses
STOP_LOSS_CONFIG = {
    'enable_tight_stops': True,
    'max_loss_per_trade': 0.02,  # 2% max loss per trade
    'atr_stop_multiplier': 1.5,  # Tighter stops (was 2.0)
    'trailing_stop_enabled': True,
    'trailing_stop_distance': 0.01,  # 1% trailing stop
}

def get_stop_loss_price(entry_price, side, atr):
    """Calculate tight stop loss price"""
    multiplier = STOP_LOSS_CONFIG['atr_stop_multiplier']
    
    if side == 'buy':
        return entry_price - (atr * multiplier)
    else:
        return entry_price + (atr * multiplier)

print("Enhanced stop-loss system activated")
