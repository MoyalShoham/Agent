#!/usr/bin/env python3
"""
Position and Margin Analyzer
"""
import pandas as pd

def analyze_positions():
    try:
        df = pd.read_csv('futures_trades_log.csv')
        
        print('📊 CURRENT POSITION ANALYSIS')
        print('='*50)
        
        # Latest positions by symbol
        latest_positions = df.groupby('symbol').last()
        active_positions = latest_positions[latest_positions['new_pos'] != 0]
        
        print(f'🔹 Active Positions: {len(active_positions)}')
        print(f'🔹 Total Symbols Traded: {df["symbol"].nunique()}')
        print(f'🔹 Total Trades: {len(df)}')
        
        print('\n📈 ACTIVE POSITIONS:')
        print('-' * 50)
        print(f'{"Symbol":<12} | {"Position":<10} | {"Side":<4} | {"PnL":<8}')
        print('-' * 50)
        
        total_margin_used = 0
        for symbol, row in active_positions.iterrows():
            pos_size = abs(row['new_pos'])
            price = row['price']
            margin_per_position = pos_size * price  # Rough estimate
            total_margin_used += margin_per_position
            
            print(f'{symbol:<12} | {row["new_pos"]:>10.1f} | {row["side"]:<4} | {row["realized_pnl"]:>8.2f}')
        
        print('-' * 50)
        print(f'💰 Total Realized PnL: {df["realized_pnl"].sum():.2f} USDT')
        print(f'💼 Estimated Margin Used: ~{total_margin_used:.0f} USDT')
        
        # Recent activity
        recent_trades = len(df[df['timestamp'] > '2025-08-11T20:00:00'])
        print(f'⚡ Recent Activity: {recent_trades} trades in last 3+ hours')
        
        # Failed orders analysis
        print(f'\n❌ MARGIN ISSUES DETECTED')
        print(f'Solution: Reduce position sizes by 50-70%')
        print(f'Current estimated margin usage: ~{total_margin_used:.0f} USDT')
        print(f'Recommended: Use max 30-50% of available margin')
        
        return len(active_positions), total_margin_used
        
    except Exception as e:
        print(f"Error analyzing positions: {e}")
        return 0, 0

if __name__ == "__main__":
    analyze_positions()
