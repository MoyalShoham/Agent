#!/usr/bin/env python3
"""
EMERGENCY TRADING FIX - Addresses critical overtrading and loss issues
"""

import os
import sys
import json
from datetime import datetime, timedelta

def create_emergency_fixes():
    """Create emergency configuration to stop the bleeding"""
    
    print("🚨 IMPLEMENTING EMERGENCY TRADING FIXES")
    print("="*50)
    
    # 1. Update .env with stricter settings
    env_fixes = {
        'AI_SIGNAL_CONFIDENCE_THRESHOLD': '85',  # Much higher confidence required
        'AI_POSITION_SIZE_MULTIPLIER': '0.3',    # Much smaller positions
        'MIN_TRADE_INTERVAL_MINUTES': '15',      # Minimum 15 minutes between trades
        'MAX_OPEN_POSITIONS': '5',               # Reduce open positions
        'STOP_LOSS_PERCENTAGE': '2.0',           # Tighter stop losses
        'TAKE_PROFIT_PERCENTAGE': '4.0',         # Better risk/reward ratio
        'MAX_DAILY_LOSSES': '0.05',              # Stop trading after 5% daily loss
        'ENABLE_POSITION_COOLDOWN': 'true',      # Prevent rapid reversals
        'COOLDOWN_PERIOD_MINUTES': '30',         # 30-minute cooldown per symbol
    }
    
    print("📝 Updating .env with emergency settings...")
    
    # Read current .env
    env_content = ""
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            env_content = f.read()
    
    # Update or add each setting
    for key, value in env_fixes.items():
        if f"{key}=" in env_content:
            # Update existing
            lines = env_content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}"
                    break
            env_content = '\n'.join(lines)
        else:
            # Add new
            env_content += f"\n{key}={value}"
    
    # Save updated .env
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Emergency .env settings applied")
    
    # 2. Create position manager override
    position_override = '''#!/usr/bin/env python3
"""
Emergency Position Manager Override - Prevents overtrading
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

class EmergencyPositionManager:
    """Emergency position manager to prevent overtrading"""
    
    def __init__(self):
        self.last_trades_file = 'last_trades.json'
        self.daily_losses_file = 'daily_losses.json'
        self.load_state()
    
    def load_state(self):
        """Load trading state"""
        self.last_trades = {}
        self.daily_losses = {}
        
        if os.path.exists(self.last_trades_file):
            with open(self.last_trades_file, 'r') as f:
                self.last_trades = json.load(f)
        
        if os.path.exists(self.daily_losses_file):
            with open(self.daily_losses_file, 'r') as f:
                self.daily_losses = json.load(f)
    
    def save_state(self):
        """Save trading state"""
        with open(self.last_trades_file, 'w') as f:
            json.dump(self.last_trades, f)
        
        with open(self.daily_losses_file, 'w') as f:
            json.dump(self.daily_losses, f)
    
    def can_trade_symbol(self, symbol: str) -> tuple[bool, str]:
        """Check if symbol can be traded based on cooldown"""
        now = datetime.now()
        
        # Check cooldown
        if symbol in self.last_trades:
            last_trade_time = datetime.fromisoformat(self.last_trades[symbol])
            cooldown_minutes = int(os.getenv('COOLDOWN_PERIOD_MINUTES', '30'))
            
            if now < last_trade_time + timedelta(minutes=cooldown_minutes):
                remaining = (last_trade_time + timedelta(minutes=cooldown_minutes) - now).total_seconds() / 60
                return False, f"Cooldown: {remaining:.1f}min remaining"
        
        # Check daily losses
        today = now.strftime('%Y-%m-%d')
        max_daily_loss = float(os.getenv('MAX_DAILY_LOSSES', '0.05'))
        
        if today in self.daily_losses:
            if abs(self.daily_losses[today]) > max_daily_loss:
                return False, f"Daily loss limit exceeded: {self.daily_losses[today]:.4f}"
        
        return True, "OK"
    
    def record_trade(self, symbol: str, pnl: float):
        """Record a trade for cooldown and loss tracking"""
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        # Update last trade time
        self.last_trades[symbol] = now.isoformat()
        
        # Update daily losses
        if today not in self.daily_losses:
            self.daily_losses[today] = 0.0
        self.daily_losses[today] += pnl
        
        # Save state
        self.save_state()
        
        print(f"📊 Trade recorded: {symbol} PnL: {pnl:.6f}, Daily total: {self.daily_losses[today]:.6f}")
    
    def get_max_position_size(self, base_size: float) -> float:
        """Get maximum allowed position size based on recent performance"""
        # Reduce position size based on recent losses
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today in self.daily_losses and self.daily_losses[today] < -0.02:
            # If daily losses > 2%, reduce position size by 50%
            return base_size * 0.5
        elif today in self.daily_losses and self.daily_losses[today] < -0.01:
            # If daily losses > 1%, reduce position size by 25%
            return base_size * 0.75
        
        return base_size
'''
    
    with open('emergency_position_manager.py', 'w') as f:
        f.write(position_override)
    
    print("✅ Emergency position manager created")
    
    # 3. Create signal filter
    signal_filter = '''#!/usr/bin/env python3
"""
Emergency Signal Filter - Only allows high-confidence signals
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

class EmergencySignalFilter:
    """Emergency signal filter to improve signal quality"""
    
    def __init__(self):
        self.min_confidence = float(os.getenv('AI_SIGNAL_CONFIDENCE_THRESHOLD', '85'))
        self.required_indicators = 3  # Require at least 3 indicators to agree
    
    def filter_signal(self, signal_data: Dict) -> tuple[Optional[str], float, str]:
        """Filter signal and return only high-confidence ones"""
        
        # Extract signal components
        ml_signal = signal_data.get('ml_signal', 'hold')
        ml_confidence = signal_data.get('ml_confidence', 0)
        ai_signal = signal_data.get('ai_signal', 'hold')
        ai_confidence = signal_data.get('ai_confidence', 0)
        
        # Technical indicators
        rsi = signal_data.get('rsi', 50)
        bb_signal = signal_data.get('bollinger_signal', 'hold')
        macd_signal = signal_data.get('macd_signal', 'hold')
        ema_signal = signal_data.get('ema_signal', 'hold')
        
        # Count agreeing indicators
        buy_votes = 0
        sell_votes = 0
        
        if ml_signal == 'buy':
            buy_votes += 1
        elif ml_signal == 'sell':
            sell_votes += 1
            
        if ai_signal == 'buy':
            buy_votes += 1
        elif ai_signal == 'sell':
            sell_votes += 1
            
        if bb_signal == 'buy':
            buy_votes += 1
        elif bb_signal == 'sell':
            sell_votes += 1
            
        if macd_signal == 'buy':
            buy_votes += 1
        elif macd_signal == 'sell':
            sell_votes += 1
            
        if ema_signal == 'buy':
            buy_votes += 1
        elif ema_signal == 'sell':
            sell_votes += 1
        
        # Calculate overall confidence
        total_confidence = (ml_confidence + ai_confidence) / 2
        
        # Apply strict filters
        if total_confidence < self.min_confidence:
            return None, total_confidence, f"Low confidence: {total_confidence:.1f}% < {self.min_confidence}%"
        
        # Require indicator agreement
        if buy_votes >= self.required_indicators:
            final_signal = 'buy'
        elif sell_votes >= self.required_indicators:
            final_signal = 'sell'
        else:
            return None, total_confidence, f"Insufficient indicator agreement: {buy_votes} buy, {sell_votes} sell"
        
        # Additional filters for market conditions
        if rsi > 80 and final_signal == 'buy':
            return None, total_confidence, "Overbought condition (RSI > 80)"
        
        if rsi < 20 and final_signal == 'sell':
            return None, total_confidence, "Oversold condition (RSI < 20)"
        
        return final_signal, total_confidence, "Signal approved"
    
    def should_close_position(self, current_pnl: float, entry_price: float, current_price: float) -> tuple[bool, str]:
        """Determine if position should be closed based on emergency rules"""
        
        stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENTAGE', '2.0')) / 100
        take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '4.0')) / 100
        
        price_change = (current_price - entry_price) / entry_price
        
        # Stop loss
        if abs(price_change) > stop_loss_pct and current_pnl < 0:
            return True, f"Stop loss triggered: {price_change*100:.2f}% move"
        
        # Take profit
        if abs(price_change) > take_profit_pct and current_pnl > 0:
            return True, f"Take profit triggered: {price_change*100:.2f}% move"
        
        return False, "Hold position"
'''
    
    with open('emergency_signal_filter.py', 'w') as f:
        f.write(signal_filter)
    
    print("✅ Emergency signal filter created")
    
    print("\n🎯 EMERGENCY FIXES SUMMARY")
    print("-"*40)
    print("✅ Confidence threshold raised to 85%")
    print("✅ Position sizes reduced by 70%")
    print("✅ 15-minute minimum between trades")
    print("✅ 30-minute cooldown per symbol")
    print("✅ Maximum 5 open positions")
    print("✅ 2% stop loss, 4% take profit")
    print("✅ 5% daily loss limit")
    print("✅ Signal filter requiring 3+ indicators")
    
    print("\n🚀 NEXT STEPS")
    print("-"*40)
    print("1. Restart the trading bot to apply fixes")
    print("2. Monitor for reduced trade frequency")
    print("3. Check that positions are smaller")
    print("4. Verify cooldown periods are working")
    print("5. Watch for improved win rate")

if __name__ == "__main__":
    create_emergency_fixes()
