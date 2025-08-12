#!/usr/bin/env python3
"""
Relaxed Training Mode - Temporarily reduce restrictions to gather training data
"""

def relax_for_training():
    """Temporarily relax restrictions to gather more training data"""
    
    print("🧠 RELAXED TRAINING MODE")
    print("=" * 40)
    print("Goal: Generate more training data, then retrain")
    
    # Read current position manager
    with open('trading_bot/execution/position_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📈 TEMPORARY ADJUSTMENTS:")
    
    # Moderate increase for data gathering: 0.15 -> 0.3
    if 'position_size_multiplier = 0.15' in content:
        content = content.replace('position_size_multiplier = 0.15', 'position_size_multiplier = 0.3')
        print("✅ Position size: 0.15 → 0.3 (temporary)")
    
    # More positions for diversity: 3 -> 5
    if 'max_concurrent_positions = 3' in content:
        content = content.replace('max_concurrent_positions = 3', 'max_concurrent_positions = 5')
        print("✅ Max positions: 3 → 5 (temporary)")
    
    with open('trading_bot/execution/position_manager.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Temporarily allow one blacklisted symbol for more data
    blacklist_content = """#!/usr/bin/env python3
\"\"\"
Symbol Blacklist - Temporarily reduced for training data
\"\"\"

# Temporarily allow XRPUSDT, keep ADAUSDT blocked (worst performer)
BLACKLISTED_SYMBOLS = ['ADAUSDT']  # Removed XRPUSDT temporarily

def is_symbol_allowed(symbol):
    \"\"\"Check if symbol is allowed for trading\"\"\"
    return symbol not in BLACKLISTED_SYMBOLS

def get_blacklisted_symbols():
    \"\"\"Get list of blacklisted symbols\"\"\"
    return BLACKLISTED_SYMBOLS.copy()

print(f"Blacklisted symbols (temporary): {BLACKLISTED_SYMBOLS}")
"""
    
    with open('trading_bot/symbol_blacklist.py', 'w', encoding='utf-8') as f:
        f.write(blacklist_content)
    
    print("✅ Allowed XRPUSDT back (temporarily)")
    print("🚫 Still blocking ADAUSDT (worst performer)")
    
    # Update emergency mode for data gathering
    emergency_content = """#!/usr/bin/env python3
\"\"\"
Emergency Trading Mode - Relaxed for training data gathering
\"\"\"

EMERGENCY_MODE = {
    'enabled': True,
    'max_daily_loss': 3.0,  # Increased from $2 to $3 temporarily
    'min_win_rate': 0.3,   # Lowered from 40% to 30% temporarily
    'position_size_reduction': 0.7,  # Less aggressive reduction
    'require_unanimous_signals': False,  # Allow more signals
    'pause_on_consecutive_losses': 5,  # Increased from 3 to 5
    'blacklisted_symbols': ['ADAUSDT'],  # Only worst performer
    'training_mode': True,  # Flag for training data gathering
}

def should_trade(daily_pnl, recent_win_rate, consecutive_losses, symbol=None):
    \"\"\"Check if trading should continue\"\"\"
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
    \"\"\"Get additional position size reduction factor\"\"\"
    if EMERGENCY_MODE['enabled']:
        return EMERGENCY_MODE['position_size_reduction']
    return 1.0

print("Emergency mode: TRAINING DATA GATHERING")
"""
    
    with open('trading_bot/config/emergency_mode.py', 'w', encoding='utf-8') as f:
        f.write(emergency_content)
    
    print("✅ Relaxed emergency limits temporarily")
    
    print("\n🎯 TRAINING DATA GATHERING PLAN:")
    print("1. 🚀 Run system for 2-4 hours")
    print("2. 📊 Gather 20-50 new trades")
    print("3. 🧠 Retrain with larger dataset")
    print("4. 🛡️ Restore strict safety settings")
    print("5. ✅ Resume normal trading")
    
    print("\n⚠️ MONITORING REQUIRED:")
    print("• Watch for losses > $3")
    print("• Stop if consecutive losses > 5")
    print("• Manual stop after 4 hours max")
    
    print("\n🚀 Ready to start: python main.py")

if __name__ == "__main__":
    print("🧠 TRAINING DATA GATHERING MODE")
    print("This temporarily relaxes safety settings to gather ML training data.")
    print("\n⚠️ RISKS:")
    print("• Slightly higher position sizes")
    print("• XRPUSDT re-enabled temporarily")
    print("• Higher loss limits")
    print("\n✅ BENEFITS:")
    print("• More training data")
    print("• Better ML models")
    print("• Improved long-term performance")
    
    choice = input("\nProceed with training data gathering? (yes/no): ").lower()
    
    if choice == 'yes':
        relax_for_training()
        print("\n🎯 Next steps:")
        print("1. python main.py (run for 2-4 hours)")
        print("2. Monitor closely")
        print("3. Stop and retrain when you have 20+ new trades")
    else:
        print("✅ Keeping current strict settings")
        print("Alternative: Wait for better market conditions")
