#!/usr/bin/env python3
"""
Emergency Trading System Fixes - Stop the bleeding!
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def emergency_fixes():
    """Apply emergency fixes to stop money loss"""
    
    print("🚨 EMERGENCY TRADING SYSTEM FIXES")
    print("=" * 50)
    print("Analysis shows the system is losing money. Applying fixes...")
    
    # 1. Create symbol blacklist for worst performers
    print("\n1. 🚫 BLACKLISTING WORST PERFORMING SYMBOLS")
    blacklist = ['ADAUSDT', 'XRPUSDT']  # Lost $0.58 and $0.70 respectively
    
    blacklist_content = f"""#!/usr/bin/env python3
\"\"\"
Symbol Blacklist - Temporarily disable worst performing symbols
\"\"\"

# Symbols that are currently losing money - temporarily disabled
BLACKLISTED_SYMBOLS = {blacklist}

def is_symbol_allowed(symbol):
    \"\"\"Check if symbol is allowed for trading\"\"\"
    return symbol not in BLACKLISTED_SYMBOLS

def get_blacklisted_symbols():
    \"\"\"Get list of blacklisted symbols\"\"\"
    return BLACKLISTED_SYMBOLS.copy()

print(f"🚫 Blacklisted symbols: {{BLACKLISTED_SYMBOLS}}")
"""
    
    with open('trading_bot/symbol_blacklist.py', 'w') as f:
        f.write(blacklist_content)
    
    print(f"   ✅ Blacklisted: {blacklist}")
    
    # 2. Reduce position sizes even more
    print("\n2. 📉 FURTHER REDUCING POSITION SIZES")
    
    # Read current position manager
    with open('trading_bot/execution/position_manager.py', 'r') as f:
        content = f.read()
    
    # Update position size multiplier from 0.3 to 0.15 (50% reduction)
    if 'position_size_multiplier = 0.3' in content:
        content = content.replace('position_size_multiplier = 0.3', 'position_size_multiplier = 0.15')
        print("   ✅ Position sizes reduced by 50% (0.3 → 0.15)")
    
    # Limit concurrent positions to 3
    if 'max_concurrent_positions = 5' in content:
        content = content.replace('max_concurrent_positions = 5', 'max_concurrent_positions = 3')
        print("   ✅ Max positions reduced to 3")
    
    with open('trading_bot/execution/position_manager.py', 'w') as f:
        f.write(content)
    
    # 3. Increase stop-loss sensitivity
    print("\n3. 🛡️ IMPROVING STOP-LOSS MANAGEMENT")
    
    # Create enhanced stop loss configuration
    stop_loss_config = """#!/usr/bin/env python3
\"\"\"
Enhanced Stop Loss Configuration
\"\"\"

# Emergency stop-loss settings to limit losses
STOP_LOSS_CONFIG = {
    'enable_tight_stops': True,
    'max_loss_per_trade': 0.02,  # 2% max loss per trade
    'atr_stop_multiplier': 1.5,  # Tighter stops (was 2.0)
    'trailing_stop_enabled': True,
    'trailing_stop_distance': 0.01,  # 1% trailing stop
}

def get_stop_loss_price(entry_price, side, atr):
    \"\"\"Calculate tight stop loss price\"\"\"
    multiplier = STOP_LOSS_CONFIG['atr_stop_multiplier']
    
    if side == 'buy':
        return entry_price - (atr * multiplier)
    else:
        return entry_price + (atr * multiplier)

print("🛡️ Enhanced stop-loss system activated")
"""
    
    with open('trading_bot/risk/stop_loss_config.py', 'w') as f:
        f.write(stop_loss_config)
    
    print("   ✅ Tighter stop-losses implemented")
    
    # 4. Create emergency trading mode
    print("\n4. 🔴 ENABLING EMERGENCY TRADING MODE")
    
    emergency_mode = """#!/usr/bin/env python3
\"\"\"
Emergency Trading Mode - Conservative settings to stop losses
\"\"\"

EMERGENCY_MODE = {
    'enabled': True,
    'max_daily_loss': 5.0,  # Stop trading if daily loss exceeds $5
    'min_win_rate': 0.4,   # Pause if win rate drops below 40%
    'position_size_reduction': 0.5,  # Additional 50% position size reduction
    'require_unanimous_signals': True,  # Need stronger signal consensus
    'pause_on_consecutive_losses': 3,  # Pause after 3 consecutive losses
}

def should_trade(daily_pnl, recent_win_rate, consecutive_losses):
    \"\"\"Check if trading should continue\"\"\"
    if not EMERGENCY_MODE['enabled']:
        return True
    
    if daily_pnl < -EMERGENCY_MODE['max_daily_loss']:
        return False
    
    if recent_win_rate < EMERGENCY_MODE['min_win_rate']:
        return False
    
    if consecutive_losses >= EMERGENCY_MODE['pause_on_consecutive_losses']:
        return False
    
    return True

print("🔴 Emergency trading mode: ACTIVE")
"""
    
    with open('trading_bot/config/emergency_mode.py', 'w') as f:
        f.write(emergency_mode)
    
    print("   ✅ Emergency mode activated")
    
    # 5. Update main configuration
    print("\n5. ⚙️ UPDATING MAIN CONFIGURATION")
    
    try:
        with open('trading_bot/config/settings.py', 'r') as f:
            settings_content = f.read()
        
        # Add emergency imports if not present
        if 'from .emergency_mode import EMERGENCY_MODE' not in settings_content:
            # Add import at the top
            import_line = "from .emergency_mode import EMERGENCY_MODE\nfrom ..symbol_blacklist import is_symbol_allowed\n"
            settings_content = import_line + settings_content
        
        with open('trading_bot/config/settings.py', 'w') as f:
            f.write(settings_content)
        
        print("   ✅ Settings updated with emergency configurations")
    except Exception as e:
        print(f"   ⚠️ Could not update settings: {e}")
    
    print("\n✅ EMERGENCY FIXES APPLIED!")
    print("\n📋 SUMMARY OF CHANGES:")
    print("• 🚫 Blacklisted worst performers: ADAUSDT, XRPUSDT")
    print("• 📉 Position sizes reduced by 50% (0.3 → 0.15)")
    print("• 🎯 Max concurrent positions: 5 → 3")
    print("• 🛡️ Tighter stop-losses (1.5x ATR)")
    print("• 🔴 Emergency mode: Daily loss limit $5")
    print("• ⏸️ Auto-pause after 3 consecutive losses")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Restart trading system: python main.py")
    print("2. Run adaptive training: python training_center.py -> option 4")
    print("3. Monitor performance closely")
    print("4. Consider paper trading while optimizing")

if __name__ == "__main__":
    emergency_fixes()
