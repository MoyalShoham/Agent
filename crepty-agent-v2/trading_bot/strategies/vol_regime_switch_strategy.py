"""
Volatility Regime Switch Strategy
 - Classifies volatility regime using rolling std of returns.
 - Uses two sub-modes:
    * High vol: breakout bias (buy if price > recent max, sell if < recent min).
    * Low vol: mean reversion (buy if price < mean - k*std, sell if > mean + k*std).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

def generate_signal(df: pd.DataFrame, lookback: int = 40, vol_period: int = 20, high_vol_mult: float = 1.2, k: float = 1.5) -> str:
    if df is None or len(df) < max(lookback, vol_period) + 5:
        return 'hold'
    d = df.tail(lookback).copy()
    close = d['close']
    rets = close.pct_change()
    vol = rets.rolling(vol_period).std()
    last_vol = vol.iloc[-1]
    median_vol = vol.median()
    high_vol = last_vol > high_vol_mult * median_vol
    logging.getLogger().info(f"[VolRegime] vol={last_vol:.5f} median={median_vol:.5f} high={high_vol}")
    if high_vol:
        recent_max = close.rolling(vol_period).max().iloc[-2]
        recent_min = close.rolling(vol_period).min().iloc[-2]
        last = close.iloc[-1]
        if last > recent_max:
            return 'buy'
        if last < recent_min:
            return 'sell'
        return 'hold'
    else:
        mean = close.rolling(vol_period).mean().iloc[-1]
        std = close.rolling(vol_period).std().iloc[-1]
        last = close.iloc[-1]
        if last < mean - k * std:
            return 'buy'
        if last > mean + k * std:
            return 'sell'
        return 'hold'
