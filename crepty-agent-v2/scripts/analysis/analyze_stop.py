#!/usr/bin/env python3
"""
Analyze why trading stopped
"""
import pandas as pd
from datetime import datetime, timedelta

def analyze_trading_stop():
    """Analyze why the trading system stopped"""
    
    print("🔍 WHY TRADING STOPPED ANALYSIS")
    print("=" * 40)
    
    # Read trade log
    df = pd.read_csv('futures_trades_log.csv')
    real_trades = df[df['paper_trade'] == False].copy()
    
    if len(real_trades) == 0:
        print("❌ No real trades found!")
        return
    
    # Convert timestamp
    real_trades['timestamp'] = pd.to_datetime(real_trades['timestamp'])
    
    # Find last trade
    last_trade = real_trades.iloc[-1]
    last_time = last_trade['timestamp']
    
    print(f"⏰ LAST TRADE: {last_time}")
    print(f"📊 SYMBOL: {last_trade['symbol']}")
    print(f"💰 PnL: ${last_trade['realized_pnl']:.4f}")
    
    # Hours since last trade
    now = datetime.now()
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=None)
    else:
        now = now.astimezone(last_time.tzinfo)
    
    hours_ago = (now - last_time).total_seconds() / 3600
    print(f"🕐 HOURS AGO: {hours_ago:.1f}")
    
    # Check recent performance
    last_10 = real_trades.tail(10)
    losing_count = len(last_10[last_10['realized_pnl'] < 0])
    winning_count = len(last_10[last_10['realized_pnl'] > 0])
    
    print(f"\n📊 LAST 10 TRADES:")
    print(f"✅ Winners: {winning_count}")
    print(f"❌ Losers: {losing_count}")
    
    # Check consecutive losses
    consecutive_losses = 0
    for pnl in reversed(real_trades.tail(20)['realized_pnl'].tolist()):
        if pnl < 0:
            consecutive_losses += 1
        elif pnl > 0:
            break
    
    print(f"📉 CONSECUTIVE LOSSES: {consecutive_losses}")
    
    # Total PnL
    total_pnl = real_trades['realized_pnl'].sum()
    print(f"💰 TOTAL PnL: ${total_pnl:.4f}")
    
    # Check if emergency limits triggered
    print(f"\n🚨 EMERGENCY LIMITS CHECK:")
    
    reasons_stopped = []
    
    if total_pnl < -2.0:
        reasons_stopped.append("❌ Hit daily loss limit ($2.00)")
    
    if consecutive_losses >= 3:
        reasons_stopped.append("❌ Hit consecutive loss limit (3)")
    
    # Check position sizes
    recent_positions = real_trades.tail(5)
    avg_position_size = abs(recent_positions['new_pos']).mean()
    
    if avg_position_size < 1.0:
        reasons_stopped.append("❌ Position sizes too small (emergency reduction)")
    
    # Check blacklisted symbols
    blacklisted = ['ADAUSDT', 'XRPUSDT']
    recent_symbols = real_trades.tail(10)['symbol'].unique()
    using_blacklisted = any(symbol in blacklisted for symbol in recent_symbols)
    
    if using_blacklisted:
        reasons_stopped.append("❌ Was trading blacklisted symbols")
    
    if reasons_stopped:
        print("🛑 REASONS TRADING STOPPED:")
        for reason in reasons_stopped:
            print(f"  {reason}")
    else:
        print("✅ No clear emergency triggers - system might be waiting for signals")
    
    print(f"\n💡 CURRENT EMERGENCY PROTECTIONS:")
    print("• Position sizes: 85% reduction")
    print("• Daily loss limit: $2.00")
    print("• Consecutive loss limit: 3")
    print("• Blacklisted: ADAUSDT, XRPUSDT")
    print("• Max positions: 3 (was 5)")
    
    print(f"\n🔧 RECOMMENDATIONS:")
    
    if total_pnl < -1.0:
        print("1. 🛑 DO NOT increase risk/leverage!")
        print("2. 📝 Switch to paper trading first")
        print("3. 🧠 Run more ML training")
        print("4. 📊 Wait for better market conditions")
    else:
        print("1. 🔄 Try restarting: python main.py")
        print("2. 📊 Monitor closely for first hour")
        print("3. 🧠 Consider additional training")

if __name__ == "__main__":
    analyze_trading_stop()
