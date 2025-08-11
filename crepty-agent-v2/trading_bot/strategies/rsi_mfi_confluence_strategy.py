"""
RSI + Money Flow Index Confluence Strategy
 - RSI oversold AND MFI oversold => buy
 - RSI overbought AND MFI overbought => sell
Default: Oversold < 40, Overbought > 60 (tighter for more activity)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.where(delta > 0, 0)).rolling(period).mean()
    down = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = tp * df.get('volume', 1)
    delta_tp = tp.diff()
    pos_mf = raw_mf.where(delta_tp > 0, 0).rolling(period).sum()
    neg_mf = raw_mf.where(delta_tp < 0, 0).rolling(period).sum()
    mfr = pos_mf / (neg_mf + 1e-9)
    return 100 - (100 / (1 + mfr))


def generate_signal(df: pd.DataFrame, period: int = 14, oversold: float = 40, overbought: float = 60) -> str:
    if df is None or len(df) < period + 5:
        return 'hold'
    d = df.tail(5*period).copy()
    rsi = _rsi(d['close'], period)
    mfi = _mfi(d, period)
    last_rsi = rsi.iloc[-1]
    last_mfi = mfi.iloc[-1]
    logging.getLogger().info(f"[RSI/MFI] RSI={last_rsi:.2f} MFI={last_mfi:.2f}")
    if last_rsi < oversold and last_mfi < oversold:
        return 'buy'
    if last_rsi > overbought and last_mfi > overbought:
        return 'sell'
    return 'hold'
