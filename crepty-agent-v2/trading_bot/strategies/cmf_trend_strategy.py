"""
Chaikin Money Flow Trend Strategy
 - CMF above upper_thresh and price above EMA -> buy bias
 - CMF below lower_thresh and price below EMA -> sell bias
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

def _cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    if df is None or len(df) < period + 2:
        return pd.Series([0.0]*len(df), index=df.index) if df is not None else pd.Series(dtype=float)
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume'] if 'volume' in df else pd.Series([1.0]*len(df), index=df.index)
    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume
    cmf_raw = mfv.rolling(period, min_periods=1).sum() / volume.rolling(period, min_periods=1).sum()
    return cmf_raw.fillna(0.0)


def generate_signal(df: pd.DataFrame, period: int = 20, ema_period: int = 34, upper_thresh: float = 0.1, lower_thresh: float = -0.1) -> str:
    if df is None or len(df) < max(period, ema_period) + 5:
        return 'hold'
    d = df.tail(max(period, ema_period)*3).copy()
    cmf = _cmf(d, period)
    close = d['close']
    ema = close.ewm(span=ema_period, adjust=False).mean()
    last_cmf = float(cmf.iloc[-1]) if len(cmf) else 0.0
    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])
    logging.getLogger().info(f"[CMF] cmf={last_cmf:.3f} price={last_close:.2f} ema={last_ema:.2f}")
    if last_cmf > upper_thresh and last_close > last_ema:
        return 'buy'
    if last_cmf < lower_thresh and last_close < last_ema:
        return 'sell'
    return 'hold'
