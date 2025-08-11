"""
Keltner Channel Breakout / Squeeze Strategy
 - EMA basis with ATR bands (multiplier default 1.5)
 - Detect squeeze using Bollinger Band width / Keltner width ratio
 - Entry when price closes outside Keltner after a squeeze period.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

def generate_signal(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, atr_mult: float = 1.5,
                    squeeze_bb_mult: float = 2.0, squeeze_lookback: int = 20, squeeze_ratio_thresh: float = 1.0) -> str:
    if df is None or len(df) < max(ema_period, atr_period, squeeze_lookback) + 5:
        return 'hold'
    d = df.copy()
    close = d['close']
    high = d['high']
    low = d['low']
    ema = close.ewm(span=ema_period, adjust=False).mean()
    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    upper_kc = ema + atr_mult * atr
    lower_kc = ema - atr_mult * atr
    # Bollinger for squeeze context
    sma = close.rolling(squeeze_lookback).mean()
    std = close.rolling(squeeze_lookback).std()
    upper_bb = sma + squeeze_bb_mult * std
    lower_bb = sma - squeeze_bb_mult * std
    bb_width = upper_bb - lower_bb
    kc_width = upper_kc - lower_kc
    width_ratio = bb_width / kc_width.replace(0, np.nan)
    recent_ratio = width_ratio.iloc[-5:]
    in_squeeze = (recent_ratio < squeeze_ratio_thresh).all()
    last_close = close.iloc[-1]
    prev_close = close.iloc[-2]
    last_upper = upper_kc.iloc[-1]
    last_lower = lower_kc.iloc[-1]
    logging.getLogger().info(f"[Keltner] close={last_close:.2f} U={last_upper:.2f} L={last_lower:.2f} squeeze={in_squeeze}")
    # Breakout logic: require prior squeeze then first close outside channel
    if in_squeeze:
        return 'hold'
    prev_outside_up = prev_close <= upper_kc.iloc[-2] and last_close > last_upper
    prev_outside_down = prev_close >= lower_kc.iloc[-2] and last_close < last_lower
    if prev_outside_up:
        return 'buy'
    if prev_outside_down:
        return 'sell'
    return 'hold'
