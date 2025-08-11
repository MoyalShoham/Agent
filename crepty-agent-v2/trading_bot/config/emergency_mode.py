#!/usr/bin/env python3
"""
Emergency Trading Mode - Conservative settings to stop losses
"""

EMERGENCY_MODE = {
    'enabled': True,
    'max_daily_loss': 2.0,  # Stop trading if daily loss exceeds $2
    'min_win_rate': 0.4,   # Pause if win rate drops below 40%
    'position_size_reduction': 0.5,  # Additional 50% position size reduction
    'require_unanimous_signals': True,  # Need stronger signal consensus
    'pause_on_consecutive_losses': 3,  # Pause after 3 consecutive losses
    'blacklisted_symbols': ['ADAUSDT', 'XRPUSDT'],  # Worst performers
}

def should_trade(daily_pnl, recent_win_rate, consecutive_losses, symbol=None):
    """Check if trading should continue"""
    if not EMERGENCY_MODE['enabled']:
        return True
    
    # Check symbol blacklist
    if symbol and symbol in EMERGENCY_MODE['blacklisted_symbols']:
        return False
    
    if daily_pnl < -EMERGENCY_MODE['max_daily_loss']:
        return False
    
    if recent_win_rate < EMERGENCY_MODE['min_win_rate']:
        return False
    
    if consecutive_losses >= EMERGENCY_MODE['pause_on_consecutive_losses']:
        return False
    
    return True

def get_emergency_position_multiplier():
    """Get additional position size reduction factor"""
    if EMERGENCY_MODE['enabled']:
        return EMERGENCY_MODE['position_size_reduction']
    return 1.0

print("Emergency trading mode: ACTIVE")
