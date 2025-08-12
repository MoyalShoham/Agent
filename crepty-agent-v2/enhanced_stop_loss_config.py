#!/usr/bin/env python3
"""
Enhanced Stop Loss Configuration
Customize your stop loss settings for optimal risk management
"""

# 🛡️ ENHANCED STOP LOSS CONFIGURATION
ENHANCED_STOP_LOSS_CONFIG = {
    # === BASIC STOP LOSS SETTINGS ===
    'atr_multiplier': 2.0,              # ATR distance for stop loss (2.0 = 2x ATR)
    'max_loss_per_trade': 0.02,         # Maximum loss per trade (2%)
    'trailing_stop_enabled': True,       # Enable trailing stops
    'trailing_distance': 0.015,         # Trailing stop distance (1.5%)
    
    # === TIME-BASED STOPS ===
    'time_based_stop_hours': 24,        # Close positions after 24 hours
    'stuck_position_hours': 12,         # Consider position stuck after 12 hours
    'weekend_close_enabled': False,     # Close positions before weekend
    
    # === EMERGENCY PROTECTION ===
    'emergency_loss_threshold': 0.05,   # Emergency stop at 5% loss
    'profit_protection_threshold': 0.03, # Protect profits above 3%
    'flash_crash_protection': True,     # Protect against flash crashes
    'gap_protection_enabled': True,     # Protect against gaps
    
    # === MARKET CONDITION ADJUSTMENTS ===
    'volatility_stop_multiplier': 1.5,  # Tighter stops in high volatility
    'volume_based_stops': True,         # Adjust stops based on volume
    'correlation_stops': True,          # Portfolio correlation protection
    'news_event_protection': False,     # Tighter stops during news (requires news feed)
    
    # === POSITION MANAGEMENT ===
    'max_concurrent_positions': 8,      # Maximum number of positions
    'position_size_scaling': True,      # Scale down position sizes on losses
    'drawdown_protection': True,        # Reduce sizes during drawdown
    'correlation_limit': 0.7,           # Maximum correlation between positions
    
    # === PROFIT TAKING ===
    'partial_profit_enabled': True,     # Take partial profits
    'profit_levels': [0.02, 0.04, 0.06], # Profit taking levels (2%, 4%, 6%)
    'profit_percentages': [0.25, 0.5, 0.25], # Percentage to close at each level
    
    # === ADVANCED FEATURES ===
    'machine_learning_stops': False,    # Use ML for dynamic stops (experimental)
    'sentiment_based_stops': False,     # Adjust stops based on sentiment
    'technical_confirmation': True,     # Require technical confirmation
    'market_structure_stops': True,     # Adjust based on market structure
    
    # === EXECUTION SETTINGS ===
    'order_type_preference': 'LIMIT',   # Preferred order type: LIMIT or MARKET
    'slippage_tolerance': 0.001,        # Acceptable slippage (0.1%)
    'retry_attempts': 3,                # Number of retry attempts for failed orders
    'execution_delay': 0.5,             # Delay between order executions (seconds)
    
    # === MONITORING ===
    'monitoring_interval': 30,          # Check interval in seconds
    'alert_enabled': True,              # Enable alerts for stop triggers
    'log_level': 'INFO',                # Logging level: DEBUG, INFO, WARNING, ERROR
    'performance_tracking': True,       # Track stop loss performance
    
    # === SAFETY FEATURES ===
    'max_daily_stops': 10,              # Maximum stops per day
    'circuit_breaker_enabled': True,    # Stop trading on excessive losses
    'manual_override_enabled': True,    # Allow manual intervention
    'emergency_shutdown_loss': 0.1,     # Shutdown at 10% portfolio loss
}

# 🎯 SYMBOL-SPECIFIC CONFIGURATIONS
SYMBOL_SPECIFIC_CONFIG = {
    'BTCUSDT': {
        'atr_multiplier': 1.8,           # Tighter stops for BTC
        'trailing_distance': 0.02,      # Wider trailing for BTC
        'max_loss_per_trade': 0.025,    # Slightly higher loss tolerance
    },
    'ETHUSDT': {
        'atr_multiplier': 2.2,           # Standard stops for ETH
        'trailing_distance': 0.018,     # Standard trailing
    },
    'SOLUSDT': {
        'atr_multiplier': 2.5,           # Wider stops for high volatility
        'trailing_distance': 0.025,     # Wider trailing
        'volatility_stop_multiplier': 2.0, # More conservative in high vol
    },
    # Add more symbols as needed
}

# 📊 RISK PROFILES
RISK_PROFILES = {
    'conservative': {
        'atr_multiplier': 1.5,
        'max_loss_per_trade': 0.015,
        'trailing_distance': 0.01,
        'max_concurrent_positions': 5,
        'emergency_loss_threshold': 0.03,
    },
    'moderate': {
        'atr_multiplier': 2.0,
        'max_loss_per_trade': 0.02,
        'trailing_distance': 0.015,
        'max_concurrent_positions': 8,
        'emergency_loss_threshold': 0.05,
    },
    'aggressive': {
        'atr_multiplier': 2.5,
        'max_loss_per_trade': 0.03,
        'trailing_distance': 0.02,
        'max_concurrent_positions': 12,
        'emergency_loss_threshold': 0.07,
    }
}

def get_config(risk_profile='moderate', symbol=None):
    """
    Get configuration for enhanced stop loss
    
    Args:
        risk_profile: 'conservative', 'moderate', or 'aggressive'
        symbol: Optional symbol for symbol-specific config
    
    Returns:
        dict: Configuration dictionary
    """
    # Start with base config
    config = ENHANCED_STOP_LOSS_CONFIG.copy()
    
    # Apply risk profile
    if risk_profile in RISK_PROFILES:
        config.update(RISK_PROFILES[risk_profile])
    
    # Apply symbol-specific config
    if symbol and symbol in SYMBOL_SPECIFIC_CONFIG:
        config.update(SYMBOL_SPECIFIC_CONFIG[symbol])
    
    return config

def validate_config(config):
    """Validate configuration values"""
    errors = []
    
    # Check required fields
    required_fields = [
        'atr_multiplier', 'max_loss_per_trade', 'trailing_distance',
        'max_concurrent_positions', 'emergency_loss_threshold'
    ]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate ranges
    if config.get('atr_multiplier', 0) <= 0:
        errors.append("atr_multiplier must be positive")
    
    if not (0 < config.get('max_loss_per_trade', 0) <= 0.1):
        errors.append("max_loss_per_trade must be between 0 and 0.1 (10%)")
    
    if not (0 < config.get('trailing_distance', 0) <= 0.05):
        errors.append("trailing_distance must be between 0 and 0.05 (5%)")
    
    if config.get('max_concurrent_positions', 0) <= 0:
        errors.append("max_concurrent_positions must be positive")
    
    return errors

# 🔧 QUICK SETUP FUNCTIONS

def get_conservative_config():
    """Get conservative stop loss configuration"""
    return get_config('conservative')

def get_moderate_config():
    """Get moderate stop loss configuration"""
    return get_config('moderate')

def get_aggressive_config():
    """Get aggressive stop loss configuration"""
    return get_config('aggressive')

def get_symbol_config(symbol, risk_profile='moderate'):
    """Get configuration for specific symbol"""
    return get_config(risk_profile, symbol)

# Example usage:
if __name__ == '__main__':
    print("🛡️ Enhanced Stop Loss Configuration")
    print("=" * 50)
    
    # Show different risk profiles
    for profile in ['conservative', 'moderate', 'aggressive']:
        config = get_config(profile)
        print(f"\n{profile.upper()} Profile:")
        print(f"  ATR Multiplier: {config['atr_multiplier']}")
        print(f"  Max Loss: {config['max_loss_per_trade']:.1%}")
        print(f"  Trailing Distance: {config['trailing_distance']:.1%}")
        print(f"  Max Positions: {config['max_concurrent_positions']}")
        print(f"  Emergency Stop: {config['emergency_loss_threshold']:.1%}")
    
    # Show symbol-specific configs
    print("\nSYMBOL-SPECIFIC Configurations:")
    for symbol in SYMBOL_SPECIFIC_CONFIG:
        config = get_symbol_config(symbol)
        print(f"  {symbol}: ATR={config['atr_multiplier']}, "
              f"Trail={config['trailing_distance']:.1%}")
    
    print("\n✅ Configuration loaded successfully!")
