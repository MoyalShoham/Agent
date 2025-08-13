#!/usr/bin/env python3
"""
Minimal Position Size Adjustment - ONLY if you must trade now
"""

def minimal_adjustment():
    """Very small increase to position sizes - risky!"""
    
    print("⚠️ MINIMAL POSITION ADJUSTMENT")
    print("=" * 40)
    print("WARNING: This is risky! Better to train models first.")
    
    # Read current position manager
    with open('trading_bot/execution/position_manager.py', 'r') as f:
        content = f.read()
    
    # Very small increase: 0.15 -> 0.25 (still 75% smaller than original)
    if 'position_size_multiplier = 0.15' in content:
        content = content.replace('position_size_multiplier = 0.15', 'position_size_multiplier = 0.25')
        print("✅ Position size: 0.15 → 0.25 (still 75% reduced)")
    
    # Slightly increase max positions: 3 -> 4
    if 'max_concurrent_positions = 3' in content:
        content = content.replace('max_concurrent_positions = 3', 'max_concurrent_positions = 4')
        print("✅ Max positions: 3 → 4")
    
    with open('trading_bot/execution/position_manager.py', 'w') as f:
        f.write(content)
    
    print("\n🚨 CRITICAL WARNINGS:")
    print("• Still very conservative settings")
    print("• Monitor closely - stop if losses continue")
    print("• Consider paper trading instead")
    print("• Train models ASAP")
    
    print("\n🚀 Next: python main.py")

if __name__ == "__main__":
    print("⚠️ WARNING: This increases risk!")
    print("Better options:")
    print("1. Train models first: python training_center.py")
    print("2. Paper trading mode")
    print("3. Wait for better market conditions")
    print()
    choice = input("Still want to increase positions? (yes/no): ").lower()
    
    if choice == 'yes':
        minimal_adjustment()
    else:
        print("✅ Smart choice! Train models first.")
        print("Run: python training_center.py")
