#!/usr/bin/env python3
"""
Emergency Trading Patch - Adds immediate position and signal controls
"""

import os
import json
from datetime import datetime, timedelta

class EmergencyTradingControls:
    """Emergency controls to prevent overtrading and losses"""
    
    def __init__(self):
        self.last_trades_file = 'emergency_last_trades.json'
        self.daily_pnl_file = 'emergency_daily_pnl.json'
        self.load_state()
        # Interpret confidence threshold; allow legacy percent (>1) or fraction (<=1)
        raw_conf = os.getenv('AI_SIGNAL_CONFIDENCE_THRESHOLD', '0.6')
        try:
            conf_val = float(raw_conf)
        except Exception:
            conf_val = 0.6
        self.min_confidence = conf_val  # store raw; caller will adapt scale
        self.min_trade_interval = int(os.getenv('MIN_TRADE_INTERVAL_MINUTES', '15'))
        self.cooldown_period = int(os.getenv('COOLDOWN_PERIOD_MINUTES', '30'))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSSES', '0.05'))
        self.max_open_positions = int(os.getenv('MAX_OPEN_POSITIONS', '5'))
        self.position_multiplier = float(os.getenv('AI_POSITION_SIZE_MULTIPLIER', '0.3'))
        # --- Auto confidence adaptation settings ---
        self.auto_lower_enabled = os.getenv('AI_CONFIDENCE_AUTO_LOWER', '0') == '1'
        self.idle_minutes = int(os.getenv('AI_CONFIDENCE_IDLE_MINUTES', '60'))  # minutes with no trades before lowering
        self.lower_step = float(os.getenv('AI_CONFIDENCE_STEP', '5'))  # if percent mode else 0.05
        self.min_floor = float(os.getenv('AI_CONFIDENCE_MIN_FLOOR', '25'))  # percent floor (or 0.25 if fractional)
        self.restore_on_trade = os.getenv('AI_CONFIDENCE_RESTORE_ON_TRADE', '1') == '1'
        self._baseline_min_confidence = self.min_confidence
        self._last_any_trade_time = datetime.now()

    def load_state(self):
        """Load trading state"""
        self.last_trades = {}
        self.daily_pnl = {}
        
        try:
            if os.path.exists(self.last_trades_file):
                with open(self.last_trades_file, 'r') as f:
                    self.last_trades = json.load(f)
        except:
            self.last_trades = {}
            
        try:
            if os.path.exists(self.daily_pnl_file):
                with open(self.daily_pnl_file, 'r') as f:
                    self.daily_pnl = json.load(f)
        except:
            self.daily_pnl = {}
    
    def save_state(self):
        """Save trading state"""
        try:
            with open(self.last_trades_file, 'w') as f:
                json.dump(self.last_trades, f)
            with open(self.daily_pnl_file, 'w') as f:
                json.dump(self.daily_pnl, f)
        except Exception as e:
            print(f"Error saving emergency state: {e}")
    
    def should_allow_trade(self, symbol: str, signal_confidence: float = 0) -> tuple[bool, str]:
        """Emergency check if trade should be allowed.
        signal_confidence passed in should match scale of self.min_confidence (auto handled by caller)."""
        # Auto-lower logic (before evaluation)
        self._maybe_auto_lower_threshold()
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        # Normalize: if min_confidence > 1 treat as percent values
        min_conf = self.min_confidence
        sc = signal_confidence
        if min_conf > 1 and sc <= 1:
            sc *= 100
        # 1. Check signal confidence threshold
        if sc < min_conf:
            return False, f"Signal confidence {sc:.2f} < required {min_conf}"
        
        # 2. Check minimum trade interval
        if symbol in self.last_trades:
            last_trade_time = datetime.fromisoformat(self.last_trades[symbol])
            if now < last_trade_time + timedelta(minutes=self.min_trade_interval):
                remaining = (last_trade_time + timedelta(minutes=self.min_trade_interval) - now).total_seconds() / 60
                return False, f"Min trade interval: {remaining:.1f}min remaining"
        
        # 3. Check cooldown period
        cooldown_key = f"{symbol}_cooldown"
        if cooldown_key in self.last_trades:
            cooldown_time = datetime.fromisoformat(self.last_trades[cooldown_key])
            if now < cooldown_time + timedelta(minutes=self.cooldown_period):
                remaining = (cooldown_time + timedelta(minutes=self.cooldown_period) - now).total_seconds() / 60
                return False, f"Cooldown: {remaining:.1f}min remaining"
        
        # 4. Check daily loss limit
        if today in self.daily_pnl and self.daily_pnl[today] < -self.max_daily_loss:
            return False, f"Daily loss limit exceeded: {self.daily_pnl[today]:.4f}"
        
        return True, "ALLOWED"
    
    def record_trade(self, symbol: str, signal: str, pnl: float = 0):
        """Record trade for tracking"""
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        # Record last trade time
        self.last_trades[symbol] = now.isoformat()
        
        # If position was closed, set cooldown
        if signal in ['close', 'sell'] and pnl != 0:
            self.last_trades[f"{symbol}_cooldown"] = now.isoformat()
        
        # Update daily PnL
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0.0
        self.daily_pnl[today] += pnl
        
        # Track last any trade time
        self._last_any_trade_time = now
        # Optionally restore baseline after a trade (activity resumed)
        if self.auto_lower_enabled and self.restore_on_trade and self.min_confidence != self._baseline_min_confidence:
            self.min_confidence = self._baseline_min_confidence
            print(f"[AUTO_CONF] Restored confidence threshold to baseline {self.min_confidence}")
        
        self.save_state()
        
        print(f"🚨 EMERGENCY: Trade recorded {symbol} {signal} PnL:{pnl:.6f} Daily:{self.daily_pnl[today]:.6f}")
    
    def _maybe_auto_lower_threshold(self):
        """Lower confidence threshold gradually if no trades have occurred for idle_minutes.
        Supports percent or fractional scale automatically.
        """
        if not self.auto_lower_enabled:
            return
        now = datetime.now()
        idle_delta = now - getattr(self, '_last_any_trade_time', now)
        if idle_delta < timedelta(minutes=self.idle_minutes):
            return
        # Determine mode
        percent_mode = self._baseline_min_confidence > 1
        step = self.lower_step if percent_mode else (self.lower_step / 100.0 if self.lower_step > 1 else self.lower_step)
        floor = self.min_floor if percent_mode else (self.min_floor / 100.0 if self.min_floor > 1 else self.min_floor)
        if self.min_confidence > floor:
            old = self.min_confidence
            self.min_confidence = max(floor, self.min_confidence - step)
            print(f"[AUTO_CONF] Lowered confidence threshold {old} -> {self.min_confidence} after {idle_delta.total_seconds()/60:.1f}m idle")
    
    def adjust_position_size(self, original_size: float) -> float:
        """Apply emergency position size reduction"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Base reduction
        adjusted_size = original_size * self.position_multiplier
        
        # Additional reduction based on daily losses
        if today in self.daily_pnl:
            daily_loss = self.daily_pnl[today]
            if daily_loss < -0.02:  # If daily loss > 2%
                adjusted_size *= 0.5  # Further 50% reduction
            elif daily_loss < -0.01:  # If daily loss > 1%
                adjusted_size *= 0.75  # 25% reduction
        
        return adjusted_size
    
    def get_status(self) -> dict:
        """Get emergency control status"""
        today = datetime.now().strftime('%Y-%m-%d')
        return {
            'daily_pnl': self.daily_pnl.get(today, 0.0),
            'max_daily_loss': self.max_daily_loss,
            'position_multiplier': self.position_multiplier,
            'min_confidence': self.min_confidence,
            'active_cooldowns': len([k for k in self.last_trades.keys() if '_cooldown' in k])
        }

# Global instance
emergency_controls = EmergencyTradingControls()
