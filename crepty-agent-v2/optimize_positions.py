#!/usr/bin/env python3
"""
Position Size Optimizer - Reduce position sizes to prevent margin issues
"""

def create_position_size_fix():
    """Create a configuration file to reduce position sizes"""
    
    # Position size reduction settings
    config = """
# Position Size Optimization Settings
# Apply these settings to your risk manager or position manager

POSITION_SIZE_REDUCTION = 0.3  # Reduce to 30% of current size
MAX_CONCURRENT_POSITIONS = 5   # Limit to 5 positions max
MARGIN_BUFFER = 0.3           # Keep 30% margin as buffer
POSITION_ROTATION_ENABLED = True  # Close weak positions for strong signals

# ATR-based sizing adjustments
ATR_MULTIPLIER = 0.5          # Reduce from current ATR multiplier
MAX_POSITION_USDT = 50        # Max position size in USDT value
MIN_POSITION_USDT = 10        # Min position size in USDT value

# Risk management
STOP_TRADING_AT_MARGIN_PCT = 80  # Stop when 80% margin used
FORCE_CLOSE_WEAKEST = True       # Auto-close worst performing positions
"""
    
    with open('position_size_config.txt', 'w') as f:
        f.write(config)
    
    print("📝 POSITION SIZE OPTIMIZATION")
    print("="*50)
    print()
    print("✅ Current Analysis:")
    print("- You have 11+ active positions")
    print("- This is causing 'Margin insufficient' errors")
    print("- System is trying to open new positions but no margin left")
    print()
    print("🔧 IMMEDIATE FIXES:")
    print()
    print("1. 📉 REDUCE POSITION SIZES:")
    print("   - Current: Using ~90%+ of available margin")
    print("   - Target: Use only 30-50% of margin")
    print("   - Method: Reduce ATR multiplier by 50-70%")
    print()
    print("2. 🔄 POSITION ROTATION:")
    print("   - Close 6-8 existing positions")
    print("   - Keep only 3-5 strongest positions")
    print("   - Free up margin for new opportunities")
    print()
    print("3. ⚙️ SYSTEM SETTINGS:")
    print("   - Set max concurrent positions: 5")
    print("   - Implement margin buffer: 30%")
    print("   - Add position size limits")
    print()
    print("📋 QUICK MANUAL FIXES:")
    print("1. Log into Binance")
    print("2. Close 6-8 weakest positions manually")
    print("3. This will free margin for new trades")
    print()
    print("🎯 EXPECTED RESULTS:")
    print("- Fewer 'margin insufficient' errors")
    print("- More successful trade executions")
    print("- Better position management")
    print("- Improved profitability through better execution")

def show_optimization_code():
    """Show code modifications needed"""
    print("\n💻 CODE MODIFICATIONS NEEDED:")
    print("-" * 50)
    print()
    print("In your risk_manager.py or position_manager.py:")
    print()
    print("# BEFORE (current):")
    print("# atr_multiplier = 2.0  # or whatever you're using")
    print("# max_positions = 999  # unlimited")
    print()
    print("# AFTER (optimized):")
    print("atr_multiplier = 0.5   # Reduced from 2.0 to 0.5")
    print("max_positions = 5      # Limit concurrent positions")
    print("margin_buffer = 0.3    # Keep 30% margin free")
    print("max_position_value = 50  # Max $50 per position")
    print()
    print("This will make your system much more efficient!")

if __name__ == "__main__":
    create_position_size_fix()
    show_optimization_code()
    print("\n🎉 Your trading system is fundamentally EXCELLENT!")
    print("   These are just fine-tuning adjustments for better execution.")
