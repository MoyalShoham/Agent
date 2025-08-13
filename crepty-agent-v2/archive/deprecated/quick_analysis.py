#!/usr/bin/env python3
"""Quick analysis of trading performance and position management issues"""

import pandas as pd
import numpy as np

def analyze_trading_issues():
    """Analyze recent trading performance and identify issues"""
    
    print("🔍 TRADING PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Load trading data
    df = pd.read_csv('futures_trades_log.csv')
    
    print(f"📊 Total trades: {len(df)}")
    print(f"💰 Total PnL: {df['realized_pnl'].sum():.6f}")
    print(f"📈 Recent 50 trades PnL: {df.tail(50)['realized_pnl'].sum():.6f}")
    print(f"📉 Recent 20 trades PnL: {df.tail(20)['realized_pnl'].sum():.6f}")
    print()
    
    print("🎯 POSITION MANAGEMENT ANALYSIS")
    print("-"*40)
    
    # Check current positions
    latest_by_symbol = df.groupby('symbol').last()
    open_positions = latest_by_symbol[latest_by_symbol['new_pos'] != 0]
    
    print(f"🔄 Open positions: {len(open_positions)}")
    if len(open_positions) > 0:
        print("Current open positions:")
        for symbol, row in open_positions.iterrows():
            print(f"   {symbol}: {row['new_pos']:.2f} (Last PnL: {row['realized_pnl']:.6f})")
    else:
        print("   ✅ No open positions")
    print()
    
    print("⚠️ PROBLEM IDENTIFICATION")
    print("-"*40)
    
    # Check for frequent position flips
    position_changes = df[df['new_pos'] != df['prev_pos']]
    recent_changes = position_changes.tail(30)
    
    print(f"🔄 Recent position changes (last 30): {len(recent_changes)}")
    
    # Check for rapid buy/sell cycles
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_sorted = df.sort_values(['symbol', 'timestamp'])
    
    # Find rapid position reversals (same symbol, opposite positions within 5 minutes)
    rapid_reversals = []
    for symbol in df['symbol'].unique():
        symbol_data = df_sorted[df_sorted['symbol'] == symbol].copy()
        for i in range(1, len(symbol_data)):
            time_diff = (symbol_data.iloc[i]['timestamp'] - symbol_data.iloc[i-1]['timestamp']).total_seconds() / 60
            pos_diff = symbol_data.iloc[i]['new_pos'] - symbol_data.iloc[i-1]['new_pos']
            
            if time_diff < 5 and abs(pos_diff) > 0.1:  # Position change within 5 minutes
                rapid_reversals.append({
                    'symbol': symbol,
                    'time_diff_min': time_diff,
                    'pos_change': pos_diff,
                    'pnl': symbol_data.iloc[i]['realized_pnl']
                })
    
    print(f"⚡ Rapid position reversals (< 5min): {len(rapid_reversals)}")
    if rapid_reversals:
        recent_reversals = rapid_reversals[-10:]
        print("Recent rapid reversals:")
        for rev in recent_reversals:
            print(f"   {rev['symbol']}: {rev['pos_change']:.2f} change in {rev['time_diff_min']:.1f}min, PnL: {rev['pnl']:.6f}")
    print()
    
    print("📈 PERFORMANCE METRICS")
    print("-"*40)
    
    # Win rate analysis
    profitable_trades = df[df['realized_pnl'] > 0]
    losing_trades = df[df['realized_pnl'] < 0]
    
    print(f"🎯 Win rate: {len(profitable_trades)/len(df)*100:.1f}%")
    print(f"💚 Profitable trades: {len(profitable_trades)}")
    print(f"💔 Losing trades: {len(losing_trades)}")
    print(f"📊 Average win: {profitable_trades['realized_pnl'].mean():.6f}")
    print(f"📊 Average loss: {losing_trades['realized_pnl'].mean():.6f}")
    
    # Recent performance trend
    recent_50 = df.tail(50)
    recent_profitable = recent_50[recent_50['realized_pnl'] > 0]
    recent_win_rate = len(recent_profitable) / len(recent_50) * 100
    
    print(f"🔥 Recent win rate (50 trades): {recent_win_rate:.1f}%")
    print()
    
    print("🚨 IDENTIFIED ISSUES")
    print("-"*40)
    
    issues = []
    
    if len(rapid_reversals) > 10:
        issues.append("❌ Too many rapid position reversals - indicates overtrading")
    
    if recent_win_rate < 40:
        issues.append("❌ Low recent win rate - strategy not working in current market")
    
    if df.tail(20)['realized_pnl'].sum() < -0.01:
        issues.append("❌ Significant recent losses - need better risk management")
    
    if len(open_positions) > 10:
        issues.append("❌ Too many open positions - spreading risk too thin")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ No major issues detected")
    
    print()
    print("💡 RECOMMENDATIONS")
    print("-"*40)
    print("1. 🛑 Reduce position frequency - add minimum hold time")
    print("2. 🎯 Increase signal confidence threshold")
    print("3. 📉 Implement better stop-loss management")
    print("4. 🔄 Add position size limits based on recent performance")
    print("5. ⏰ Add cooldown period between trades on same symbol")

if __name__ == "__main__":
    analyze_trading_issues()
