#!/usr/bin/env python3
"""
Emergency Trading Mode - Relaxed for training data gathering
"""

EMERGENCY_MODE = {
    'enabled': True,
    'max_daily_loss': 3.0,  # Increased from $2 to $3 temporarily
    'min_win_rate': 0.3,   # Lowered from 40% to 30% temporarily
    'position_size_reduction': 0.7,  # Less aggressive reduction
    'require_unanimous_signals': False,  # Allow more signals
    'pause_on_consecutive_losses': 5,  # Increased from 3 to 5
    'blacklisted_symbols': ['ADAUSDT', 'KASUSDT'],  # Worst performer + Invalid symbol
    'training_mode': True,  # Flag for training data gathering
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

print("Emergency mode: TRAINING DATA GATHERING")
