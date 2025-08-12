#!/usr/bin/env python3
"""
Quick Trading Performance Analysis
"""
import pandas as pd
import numpy as np

def analyze_trading_performance():
    """Analyze current trading performance to identify loss patterns"""
    try:
        # Read trade log
        df = pd.read_csv('futures_trades_log.csv')
        
        # Filter for real trades only (paper_trade=False)
        real_trades = df[df['paper_trade'] == False].copy()
        
        print('📊 TRADING PERFORMANCE ANALYSIS')
        print('=' * 50)
        print(f'Total real trades: {len(real_trades)}')
        
        if len(real_trades) == 0:
            print("❌ No real trades found! All trades are paper trades.")
            return
        
        # Calculate total PnL
        total_pnl = real_trades['realized_pnl'].sum()
        print(f'💰 Total Realized PnL: ${total_pnl:.4f}')
        
        # Winning vs losing trades
        winning_trades = real_trades[real_trades['realized_pnl'] > 0]
        losing_trades = real_trades[real_trades['realized_pnl'] < 0]
        neutral_trades = real_trades[real_trades['realized_pnl'] == 0]
        
        print(f'✅ Winning trades: {len(winning_trades)} (${winning_trades["realized_pnl"].sum():.4f})')
        print(f'❌ Losing trades: {len(losing_trades)} (${losing_trades["realized_pnl"].sum():.4f})')
        print(f'⚪ Neutral trades: {len(neutral_trades)}')
        
        # Win rate
        trades_with_pnl = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / trades_with_pnl * 100 if trades_with_pnl > 0 else 0
        print(f'📈 Win rate: {win_rate:.1f}%')
        
        # Average profit/loss per trade
        avg_win = winning_trades['realized_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['realized_pnl'].mean() if len(losing_trades) > 0 else 0
        
        print(f'💚 Average winning trade: ${avg_win:.4f}')
        print(f'💔 Average losing trade: ${avg_loss:.4f}')
        
        # Risk-reward ratio
        if avg_loss != 0:
            risk_reward = abs(avg_win / avg_loss)
            print(f'⚖️ Risk-Reward Ratio: {risk_reward:.2f}:1')
        
        # Performance by symbol
        print('\n📈 PERFORMANCE BY SYMBOL:')
        symbol_pnl = real_trades.groupby('symbol')['realized_pnl'].agg(['sum', 'count', 'mean'])
        symbol_pnl = symbol_pnl.sort_values('sum', ascending=False)
        
        for symbol, data in symbol_pnl.iterrows():
            pnl_sum = data['sum']
            trade_count = data['count']
            avg_pnl = data['mean']
            
            # Color code performance
            emoji = "🟢" if pnl_sum > 0 else "🔴" if pnl_sum < 0 else "⚪"
            print(f'{emoji} {symbol}: ${pnl_sum:.4f} ({trade_count} trades, avg: ${avg_pnl:.4f})')
        
        print('\n🔍 ISSUES IDENTIFIED:')
        issues = []
        
        if total_pnl < 0:
            issues.append('• System is losing money overall')
        
        if win_rate < 50:
            issues.append('• Win rate is below 50%')
        
        if abs(avg_loss) > avg_win and avg_loss != 0:
            issues.append('• Average losses exceed average wins')
        
        # Check for overtrading specific symbols
        worst_performers = symbol_pnl[symbol_pnl['sum'] < -0.1]
        if len(worst_performers) > 0:
            issues.append(f'• Worst performing symbols: {", ".join(worst_performers.index.tolist())}')
        
        # Check recent performance trend
        if len(real_trades) >= 10:
            recent_trades = real_trades.tail(10)
            recent_pnl = recent_trades['realized_pnl'].sum()
            if recent_pnl < 0:
                issues.append(f'• Recent 10 trades are losing: ${recent_pnl:.4f}')
        
        if issues:
            for issue in issues:
                print(issue)
        else:
            print('✅ No major issues detected!')
        
        # Recommendations
        print('\n💡 RECOMMENDATIONS:')
        
        if total_pnl < 0:
            print('1. 🛑 Consider stopping live trading temporarily')
            print('2. 🧠 Run more ML training to improve signals')
            print('3. 📊 Analyze strategy performance individually')
        
        if win_rate < 50:
            print('4. 🎯 Focus on improving signal quality')
            print('5. ⚙️ Adjust strategy parameters')
        
        if abs(avg_loss) > avg_win:
            print('6. 🛡️ Implement better stop-loss management')
            print('7. 📏 Reduce position sizes further')
        
        # Check if models need updating
        print('\n🔄 NEXT STEPS:')
        print('• Run adaptive training with real data: python training_center.py -> option 4')
        print('• Consider paper trading while optimizing')
        print('• Monitor individual strategy performance')
        
    except Exception as e:
        print(f"❌ Error analyzing performance: {e}")

if __name__ == "__main__":
    analyze_trading_performance()
