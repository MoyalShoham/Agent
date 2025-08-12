"""
EMA Ribbon Trend Strategy
 - Build ribbon of EMAs (default: 8,13,21,34,55)
 - Bullish if all fast->slow are ordered ascending and price pullback within middle band -> buy
 - Bearish mirror -> sell
"""
from __future__ import annotations
import pandas as pd
import logging

def generate_signal(df: pd.DataFrame, emas=(8,13,21,34,55), pullback_depth: float = 0.382) -> str:
    if df is None or len(df) < max(emas) + 5:
        return 'hold'
    d = df.tail(max(emas)*4).copy()
    ema_vals = {p: d['close'].ewm(span=p, adjust=False).mean() for p in emas}
    last_emas = {p: s.iloc[-1] for p, s in ema_vals.items()}
    ordered_up = all(last_emas[emas[i]] > last_emas[emas[i+1]] for i in range(len(emas)-1))
    ordered_down = all(last_emas[emas[i]] < last_emas[emas[i+1]] for i in range(len(emas)-1))
    price = d['close'].iloc[-1]
    mid_idx = len(emas)//2
    mid_ema = last_emas[emas[mid_idx]]
    fast_ema = last_emas[emas[0]]
    slow_ema = last_emas[emas[-1]]
    logging.getLogger().info(f"[EMA Ribbon] price={price:.2f} fast={fast_ema:.2f} slow={slow_ema:.2f}")
    # Pullback condition: price between fast and mid for longs when bullish trend
    if ordered_up and price <= fast_ema and price >= mid_ema * (1 - pullback_depth):
        return 'buy'
    if ordered_down and price >= fast_ema and price <= mid_ema * (1 + pullback_depth):
        return 'sell'
    return 'hold'
