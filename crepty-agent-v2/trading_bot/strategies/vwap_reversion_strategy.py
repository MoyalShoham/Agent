"""
VWAP Reversion Strategy
Idea:
 - Compute rolling session VWAP (using typical price (H+L+C)/3 weighted by volume if volume available; else simple mean of typical price).
 - Buy: price crosses back above VWAP after being below by deviation_pct.
 - Sell: price crosses back below VWAP after being above by deviation_pct.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

def generate_signal(df: pd.DataFrame, lookback: int = 96, deviation_pct: float = 0.002) -> str:
    if df is None or len(df) < max(lookback, 20):
        return 'hold'
    d = df.tail(lookback).copy()
    if 'volume' in d.columns and d['volume'].notna().any():
        tp = (d['high'] + d['low'] + d['close']) / 3.0
        vwap = (tp * d['volume']).cumsum() / d['volume'].cumsum()
    else:
        tp = (d['high'] + d['low'] + d['close']) / 3.0
        vwap = tp.expanding().mean()
    price = d['close']
    last_price = price.iloc[-1]
    last_vwap = vwap.iloc[-1]
    prev_price = price.iloc[-2]
    prev_vwap = vwap.iloc[-2]
    deviation = (last_price - last_vwap) / last_vwap
    logging.getLogger().info(f"[VWAP] price={last_price:.2f} vwap={last_vwap:.2f} dev={deviation:.4f}")
    # Cross back toward VWAP after stretch (mean reversion)
    if prev_price < prev_vwap and last_price > last_vwap and deviation > -deviation_pct:
        return 'buy'
    if prev_price > prev_vwap and last_price < last_vwap and deviation < deviation_pct:
        return 'sell'
    return 'hold'
