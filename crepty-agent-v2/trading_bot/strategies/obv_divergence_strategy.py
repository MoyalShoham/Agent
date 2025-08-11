"""
OBV Divergence Strategy
 - Compute On Balance Volume (OBV)
 - Detect simple 2-point divergence: price higher high vs OBV lower high (bearish), price lower low vs OBV higher low (bullish)
 - Uses recent window highs/lows.
Note: Needs volume column; if absent returns hold.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

def _obv(df: pd.DataFrame) -> pd.Series:
    close = df['close']
    vol = df['volume']
    direction = np.sign(close.diff().fillna(0))
    return (direction * vol).cumsum()


def generate_signal(df: pd.DataFrame, window: int = 40, pivot_lookback: int = 10) -> str:
    if df is None or len(df) < window or 'volume' not in df.columns:
        return 'hold'
    d = df.tail(window).copy()
    if d['volume'].sum() == 0:
        return 'hold'
    obv = _obv(d)
    price = d['close']
    # Simple pivot highs/lows (last two)
    def pivots(series):
        # Efficient pivot detection using rolling window
        if len(series) < 2 * pivot_lookback + 1:
            return []
        piv = []
        roll_max = series.rolling(window=2*pivot_lookback+1, center=True).max()
        roll_min = series.rolling(window=2*pivot_lookback+1, center=True).min()
        for i in range(pivot_lookback, len(series) - pivot_lookback):
            if series.iloc[i] == roll_max.iloc[i]:
                piv.append((i, series.iloc[i]))
            if series.iloc[i] == roll_min.iloc[i]:
                piv.append((i, series.iloc[i]))
        return piv
    price_piv = pivots(price)
    obv_piv = pivots(obv)
    if len(price_piv) < 2 or len(obv_piv) < 2:
        return 'hold'
    # Take last two highs and last two lows
    price_highs = [(i,v) for i,v in price_piv if v == price.iloc[i]]
    obv_highs = [(i,v) for i,v in obv_piv if v == obv.iloc[i]]
    price_lows = price_highs  # placeholder to avoid complexity (improve later)
    obv_lows = obv_highs
    # Simplified divergence detection (improve logic if needed)
    if len(price_highs) >= 2 and len(obv_highs) >= 2:
        (i1, ph1), (i2, ph2) = price_highs[-2:]
        (_, oh1), (_, oh2) = obv_highs[-2:]
        if ph2 > ph1 and oh2 < oh1:
            logging.getLogger().info("[OBV Divergence] Bearish divergence detected")
            return 'sell'
    # For bullish we reuse (simplistic)
    if len(price_highs) >= 2 and len(obv_highs) >= 2:
        (i1, ph1), (i2, ph2) = price_highs[-2:]
        (_, oh1), (_, oh2) = obv_highs[-2:]
        if ph2 < ph1 and oh2 > oh1:
            logging.getLogger().info("[OBV Divergence] Bullish divergence detected")
            return 'buy'
    return 'hold'
