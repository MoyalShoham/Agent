"""Position Manager for per-symbol futures positions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import time

@dataclass
class Position:
    symbol: str
    size: float = 0.0  # positive long, negative short
    entry_price: float = 0.0
    realized_pnl: float = 0.0
    last_update: float = field(default_factory=time.time)

    def unrealized_pnl(self, mark_price: float) -> float:
        return (mark_price - self.entry_price) * self.size

    def update_fill(self, fill_qty: float, fill_price: float):
        # If adding to same direction
        if self.size == 0 or (self.size > 0 and fill_qty > 0) or (self.size < 0 and fill_qty < 0):
            new_size = self.size + fill_qty
            if new_size != 0:
                self.entry_price = (self.entry_price * abs(self.size) + fill_price * abs(fill_qty)) / abs(new_size)
            self.size = new_size
        else:
            # Opposite direction => reduce or flip
            if abs(fill_qty) < abs(self.size):
                # Partial close
                realized = (fill_price - self.entry_price) * (-fill_qty if self.size > 0 else fill_qty)
                self.realized_pnl += realized
                self.size += fill_qty
            elif abs(fill_qty) == abs(self.size):
                # Flat
                realized = (fill_price - self.entry_price) * (self.size if self.size > 0 else -self.size)
                self.realized_pnl += realized
                self.size = 0
                self.entry_price = 0
            else:
                # Flip: close existing + open new remainder
                close_qty = -self.size
                realized = (fill_price - self.entry_price) * (self.size if self.size > 0 else -self.size)
                self.realized_pnl += realized
                remainder = fill_qty + self.size  # because fill_qty has opposite sign
                self.size = remainder
                self.entry_price = fill_price
        self.last_update = time.time()

class PositionManager:
    def __init__(self, symbols, max_leverage: float = 1.5, risk_per_trade: float = 0.0005):
        self.positions: Dict[str, Position] = {s: Position(symbol=s) for s in symbols}
        # 🔧 EMERGENCY SETTINGS - Adjusted for minimum order sizes  
        self.max_leverage = max_leverage * 0.5  # Reduce leverage by 50%
        self.risk_per_trade = risk_per_trade * 2.0  # Increase to 200% for minimum order sizes
        self.max_concurrent_positions = 5  # Allow 5 positions for diversity
        self.margin_buffer = 0.3  # Keep 30% margin as buffer
        self.position_size_multiplier = 3.0  # Increase to 3.0 to reach minimum order sizes
        self.equity = 0.0
        self.last_equity_update = 0.0

    def update_equity(self, equity: float):
        self.equity = equity
        self.last_equity_update = time.time()

    def get_position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol=symbol))

    def target_position_size(self, symbol: str, price: float, atr: Optional[float] = None) -> float:
        # Position sizing: risk_per_trade * equity / (atr or price*0.01)
        if self.equity <= 0:
            return 0.0
        
        # 🔧 CHECK: Limit concurrent positions to prevent margin issues
        active_positions = sum(1 for pos in self.positions.values() if abs(pos.size) > 0.001)
        if active_positions >= self.max_concurrent_positions:
            return 0.0  # Don't open new positions if at limit
        
        dollar_risk = self.equity * self.risk_per_trade
        unit_risk = atr if (atr and atr > 0) else price * 0.01
        if unit_risk <= 0:
            return 0.0
        qty = dollar_risk / unit_risk
        
        # 🔧 APPLY POSITION SIZE REDUCTION MULTIPLIER
        qty *= self.position_size_multiplier
        
        # Leverage cap (notional <= equity * max_leverage)
        max_qty = (self.equity * self.max_leverage) / price if price > 0 else 0.0
        
        # 🔧 APPLY MARGIN BUFFER - Use only 70% of available margin
        max_qty *= (1.0 - self.margin_buffer)
        
        return min(qty, max_qty)
