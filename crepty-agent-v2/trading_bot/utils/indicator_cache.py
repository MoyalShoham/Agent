"""Shared Indicator Cache / Enrichment
Adds commonly used indicators to a dataframe once so individual strategies
can reuse them without recomputation.

Indicators added (if enough data length):
 - ema_12, ema_26, ema_50, ema_100, ema_200
 - sma_20, sma_50
 - rsi_14
 - atr_14
 - macd, macd_signal
 - bb_upper_20_2, bb_lower_20_2, bb_mid_20
 - stoch_k_14_3, stoch_d_14_3

Design:
 enrich_dataframe(df) returns same df reference with new columns (does not copy by default)
 Uses internal simple length guards. Failures are logged, not raised.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

EMA_SET = (12,26,50,100,200)
SMA_SET = (20,50)


def _safe(series):
    return series if isinstance(series, pd.Series) else pd.Series(dtype=float)


def enrich_dataframe(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if df is None or not len(df):
        return df
    try:
        close = df['close']
        high = df.get('high')
        low = df.get('low')
    except KeyError:
        return df
    # EMAs
    for p in EMA_SET:
        col = f'ema_{p}'
        if force or col not in df.columns:
            try:
                if len(close) >= p:
                    df[col] = close.ewm(span=p, adjust=False).mean()
            except Exception:
                logging.getLogger().debug(f"indicator_cache: failed ema {p}")
    # SMAs
    for p in SMA_SET:
        col = f'sma_{p}'
        if force or col not in df.columns:
            if len(close) >= p:
                df[col] = close.rolling(p).mean()
    # RSI 14
    if force or 'rsi_14' not in df.columns:
        if len(close) >= 15:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            df['rsi_14'] = 100 - (100 / (1 + rs))
    # ATR 14
    if high is not None and low is not None and (force or 'atr_14' not in df.columns):
        if len(df) >= 15:
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean()
    # MACD 12/26 + signal 9
    if force or 'macd' not in df.columns:
        if len(close) >= 26:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            df['macd'] = macd
            if len(macd) >= 35:
                df['macd_signal'] = macd.ewm(span=9, adjust=False).mean()
    # Bollinger 20, 2
    if force or 'bb_mid_20' not in df.columns:
        if len(close) >= 20:
            mid = close.rolling(20).mean()
            std = close.rolling(20).std()
            df['bb_mid_20'] = mid
            df['bb_upper_20_2'] = mid + 2*std
            df['bb_lower_20_2'] = mid - 2*std
    # Stochastic 14,3
    if high is not None and low is not None and (force or 'stoch_k_14_3' not in df.columns):
        if len(df) >= 17:
            low_min = low.rolling(14).min()
            high_max = high.rolling(14).max()
            k = 100 * (close - low_min) / (high_max - low_min + 1e-9)
            d_line = k.rolling(3).mean()
            df['stoch_k_14_3'] = k
            df['stoch_d_14_3'] = d_line
    return df
