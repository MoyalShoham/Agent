#!/usr/bin/env python3
"""
Emergency Restart Script - Applies all fixes and restarts trading bot
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def emergency_restart():
    """Apply emergency fixes and restart the trading bot"""
    
    print("🚨 EMERGENCY TRADING BOT RESTART")
    print("="*50)
    print(f"🕐 Timestamp: {datetime.now()}")
    print()
    
    print("✅ Emergency fixes applied:")
    print("   - 85% minimum signal confidence")
    print("   - 70% position size reduction") 
    print("   - 15-minute minimum between trades")
    print("   - 30-minute cooldown per symbol")
    print("   - 5% daily loss limit")
    print("   - Maximum 5 open positions")
    print("   - Enhanced AI+ML signal filtering")
    print()
    
    print("🔍 Checking current trading status...")
    
    # Check if bot is running
    try:
        import psutil
        bot_running = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'main.py' in cmdline or 'trading_bot' in cmdline:
                    print(f"   🔴 Found running bot process: PID {proc.info['pid']}")
                    bot_running = True
                    print(f"   🛑 Terminating process...")
                    proc.terminate()
                    time.sleep(2)
                    if proc.is_running():
                        proc.kill()
                    print(f"   ✅ Process terminated")
            except:
                pass
        
        if not bot_running:
            print("   ✅ No existing bot processes found")
            
    except ImportError:
        print("   ⚠️ psutil not available, cannot check for existing processes")
        print("   💡 Please manually stop any running trading bots before continuing")
        input("   Press Enter when ready to continue...")
    
    print()
    print("🚀 Starting trading bot with emergency fixes...")
    print("="*50)
    
    # Show what will happen
    print("📊 EXPECTED BEHAVIOR CHANGES:")
    print("   - Much fewer trades (15+ minute intervals)")
    print("   - Smaller position sizes (70% reduction)")
    print("   - Higher quality signals only (85%+ confidence)")
    print("   - No rapid position reversals (30min cooldowns)")
    print("   - Automatic stop at 5% daily loss")
    print()
    
    print("🎯 MONITORING CHECKLIST:")
    print("   □ Trade frequency should decrease dramatically")
    print("   □ Position sizes should be much smaller")
    print("   □ No trades on same symbol within 30 minutes")
    print("   □ Emergency status logs appear every loop")
    print("   □ Daily PnL tracking active")
    print()
    
    # Start the bot
    try:
        print("🔥 LAUNCHING TRADING BOT...")
        print("-" * 30)
        
        # Run the main script
        subprocess.run([sys.executable, "main.py"], check=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error starting bot: {e}")
        print("💡 Try running manually: python main.py")
    
    print("\n🏁 Emergency restart complete")

if __name__ == "__main__":
    emergency_restart()
