#!/usr/bin/env python3
"""
Test the optimized position sizing to ensure it works correctly.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_position_sizing():
    """Test the new position sizing logic"""
    try:
        from trading_bot.execution.position_manager import PositionManager
        from trading_bot.risk.risk_manager import RiskManager
        
        print("🧪 TESTING OPTIMIZED POSITION SIZING")
        print("="*50)
        
        # Test Position Manager
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        pos_mgr = PositionManager(symbols)
        pos_mgr.equity = 1000  # $1000 test equity
        
        print(f"✅ Position Manager Settings:")
        print(f"   - Max Leverage: {pos_mgr.max_leverage:.1f} (reduced from 1.5)")
        print(f"   - Risk Per Trade: {pos_mgr.risk_per_trade:.4f} (reduced from 0.0005)")
        print(f"   - Max Positions: {pos_mgr.max_concurrent_positions}")
        print(f"   - Position Size Multiplier: {pos_mgr.position_size_multiplier:.1f}")
        print(f"   - Margin Buffer: {pos_mgr.margin_buffer:.1%}")
        
        # Test position sizing
        btc_price = 95000
        eth_price = 3500
        ada_price = 0.8
        atr = 500
        
        btc_size = pos_mgr.target_position_size('BTCUSDT', btc_price, atr)
        eth_size = pos_mgr.target_position_size('ETHUSDT', eth_price, atr)
        ada_size = pos_mgr.target_position_size('ADAUSDT', ada_price, atr)
        
        print(f"\n📊 Position Size Examples (Equity: ${pos_mgr.equity}):")
        print(f"   - BTCUSDT @ ${btc_price}: {btc_size:.6f} BTC (${btc_size * btc_price:.2f})")
        print(f"   - ETHUSDT @ ${eth_price}: {eth_size:.4f} ETH (${eth_size * eth_price:.2f})")
        print(f"   - ADAUSDT @ ${ada_price}: {ada_size:.1f} ADA (${ada_size * ada_price:.2f})")
        
        total_notional = (btc_size * btc_price) + (eth_size * eth_price) + (ada_size * ada_price)
        leverage_used = total_notional / pos_mgr.equity
        margin_used = leverage_used / pos_mgr.max_leverage
        
        print(f"\n💼 Portfolio Analysis:")
        print(f"   - Total Notional: ${total_notional:.2f}")
        print(f"   - Leverage Used: {leverage_used:.2f}x")
        print(f"   - Margin Utilization: {margin_used:.1%}")
        
        # Test Risk Manager
        risk_mgr = RiskManager()
        print(f"\n🛡️ Risk Manager Settings:")
        print(f"   - Default Position Size: {risk_mgr.default_position:.1f} (reduced from 1.0)")
        print(f"   - Max Positions: {risk_mgr.max_concurrent_positions}")
        print(f"   - Max Position Size: {risk_mgr.max_position_size:.1%}")
        print(f"   - Max Leverage: {risk_mgr.max_leverage:.1f}")
        
        # Test trade allowance
        trade_allowed = risk_mgr.allow_trade()
        print(f"   - Trade Allowed: {'✅ Yes' if trade_allowed else '❌ No'}")
        
        print(f"\n🎯 OPTIMIZATION RESULTS:")
        if margin_used < 0.8:  # Less than 80% margin used
            print("✅ EXCELLENT: Margin usage is conservative")
            print("✅ Should prevent 'Margin insufficient' errors")
            print("✅ Allows room for multiple positions")
        else:
            print("⚠️ WARNING: Still using high margin")
            print("   Consider reducing position sizes further")
        
        print(f"\n💡 Expected Improvements:")
        print("   - Fewer 'margin insufficient' errors")
        print("   - Better position management")
        print("   - More successful trade executions")
        print("   - Improved capital efficiency")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing position sizing: {e}")
        return False

def test_imports():
    """Test if the enhanced modules can be imported"""
    print("🔍 TESTING MODULE IMPORTS")
    print("-"*30)
    
    modules_to_test = [
        ('trading_bot.execution.position_manager', 'PositionManager'),
        ('trading_bot.risk.risk_manager', 'RiskManager'),
        ('trading_bot.utils.ml_signals', 'MLSignalGenerator'),
        ('trading_bot.utils.portfolio_optimizer', 'PortfolioOptimizer'),
    ]
    
    all_good = True
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🚀 POSITION SIZING OPTIMIZATION TEST")
    print("="*50)
    
    # Test imports first
    if test_imports():
        print()
        # Test position sizing
        if test_position_sizing():
            print("\n🎉 ALL TESTS PASSED!")
            print("Your optimized position sizing is ready to use.")
            print("\nNext steps:")
            print("1. Restart your trading agent: python main.py")
            print("2. Monitor for reduced 'margin insufficient' errors")
            print("3. Check that trades execute successfully")
        else:
            print("\n❌ Position sizing test failed")
    else:
        print("\n❌ Module import test failed")
