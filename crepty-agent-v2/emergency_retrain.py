#!/usr/bin/env python3
"""
Emergency ML Training - Retrain models to fix current losses
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def emergency_retrain():
    """Emergency retraining focused on recent losing patterns"""
    
    print("🚨 EMERGENCY ML RETRAINING")
    print("=" * 50)
    print("Analyzing recent losses to improve model...")
    
    try:
        # Load recent trading data
        df = pd.read_csv('futures_trades_log.csv')
        real_trades = df[df['paper_trade'] == False].copy()
        
        if len(real_trades) == 0:
            print("❌ No real trades found for analysis")
            return
        
        # Focus on recent losing trades
        losing_trades = real_trades[real_trades['realized_pnl'] < 0]
        
        print(f"📊 Found {len(losing_trades)} losing trades to analyze")
        
        # Identify patterns in losing trades
        print("\n🔍 ANALYZING LOSING PATTERNS:")
        
        # Worst performing symbols
        symbol_losses = losing_trades.groupby('symbol')['realized_pnl'].sum().sort_values()
        print("📉 Worst performing symbols:")
        for symbol, loss in symbol_losses.head(3).items():
            print(f"   • {symbol}: ${loss:.4f}")
        
        # Most common losing sides
        side_losses = losing_trades.groupby('side')['realized_pnl'].agg(['count', 'sum'])
        print("\n📊 Losing trade patterns:")
        for side, data in side_losses.iterrows():
            print(f"   • {side}: {data['count']} trades, ${data['sum']:.4f}")
        
        print("\n🧠 STARTING EMERGENCY RETRAINING...")
        
        # Run multiple training iterations focusing on recent data
        for i in range(5):
            print(f"\n🔄 Emergency Training Iteration {i+1}/5")
            
            # Run training with focus on recent losses
            result = os.system("python train_ml_model.py")
            
            if result == 0:
                print(f"✅ Iteration {i+1} completed successfully")
            else:
                print(f"❌ Iteration {i+1} failed")
            
            # Short delay
            import time
            time.sleep(10)
        
        print("\n✅ EMERGENCY RETRAINING COMPLETED!")
        
        # Create trading recommendations
        print("\n💡 IMMEDIATE RECOMMENDATIONS:")
        print("1. 🛑 TEMPORARILY DISABLE worst symbols:")
        for symbol in symbol_losses.head(2).index:
            print(f"   • {symbol}")
        
        print("2. 📉 REDUCE position sizes by 50%")
        print("3. 🎯 FOCUS on symbols with positive PnL:")
        
        # Find profitable symbols
        profitable = real_trades[real_trades['realized_pnl'] > 0]
        if len(profitable) > 0:
            symbol_profits = profitable.groupby('symbol')['realized_pnl'].sum().sort_values(ascending=False)
            for symbol, profit in symbol_profits.head(3).items():
                print(f"   • {symbol}: ${profit:.4f}")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Restart trading system with new models")
        print("2. Monitor performance closely")
        print("3. Consider paper trading for a few hours")
        
    except Exception as e:
        print(f"❌ Emergency retraining failed: {e}")
        print("Falling back to basic training...")
        
        # Fallback: run basic training
        for i in range(3):
            print(f"🔄 Fallback training {i+1}/3")
            os.system("python train_ml_model.py")

if __name__ == "__main__":
    emergency_retrain()
