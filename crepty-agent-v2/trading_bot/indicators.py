"""Custom indicators module: supertrend, keltner channels, vwap, donchian width, rolling volatility.
Extend as needed. Lightweight computations relying on pandas.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def ensure_series(x):
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)

def atr(df: pd.DataFrame, period: int = 14):
    high = df['high']; low = df['low']; close = df['close']
    prev_close = close.shift(1)
    tr = (high - low).to_frame('tr1')
    tr['tr2'] = (high - prev_close).abs()
    tr['tr3'] = (low - prev_close).abs()
    tr_max = tr.max(axis=1)
    return tr_max.rolling(period).mean()

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    if len(df) < period + 2:
        return pd.Series([np.nan]*len(df), index=df.index)
    hl2 = (df['high'] + df['low']) / 2
    atr_val = atr(df, period)
    upperband = hl2 + multiplier * atr_val
    lowerband = hl2 - multiplier * atr_val
    supertrend = pd.Series(np.nan, index=df.index)
    direction = np.ones(len(df), dtype=bool)  # True=long, False=short
    # Vectorized logic: propagate direction and bands
    for i in range(period + 1, len(df)):
        prev = i - 1
        if df['close'].iloc[i] > upperband.iloc[prev]:
            direction[i] = True
        elif df['close'].iloc[i] < lowerband.iloc[prev]:
            direction[i] = False
        else:
            direction[i] = direction[prev]
        if direction[i]:
            upperband.iloc[i] = min(upperband.iloc[i], upperband.iloc[prev])
        else:
            lowerband.iloc[i] = max(lowerband.iloc[i], lowerband.iloc[prev])
        supertrend.iloc[i] = lowerband.iloc[i] if direction[i] else upperband.iloc[i]
    return supertrend

def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, atr_mult: float = 1.5):
    if len(df) < ema_period:
        return (pd.Series([np.nan]*len(df)),)*3
    typical = (df['high'] + df['low'] + df['close'])/3
    ema_mid = typical.ewm(span=ema_period, adjust=False).mean()
    atr_val = atr(df, atr_period)
    upper = ema_mid + atr_mult * atr_val
    lower = ema_mid - atr_mult * atr_val
    return upper, ema_mid, lower

def vwap(df: pd.DataFrame):
    if not {'close','high','low','volume'}.issubset(df.columns):
        return pd.Series([np.nan]*len(df))
    typical = (df['high'] + df['low'] + df['close'])/3
    cum_vol = df['volume'].cumsum()
    cum_tp_vol = (typical * df['volume']).cumsum()
    return cum_tp_vol / (cum_vol + 1e-12)

def donchian_width(df: pd.DataFrame, period: int = 20):
    if len(df) < period:
        return pd.Series([0]*len(df))
    hh = df['high'].rolling(period).max()
    ll = df['low'].rolling(period).min()
    width = (hh - ll) / (df['close'] + 1e-12)
    return width

def rolling_volatility(df: pd.DataFrame, lookback: int = 30):
    ret = df['close'].pct_change()
    return ret.rolling(lookback).std()

def efficiency_ratio(df: pd.DataFrame, lookback: int = 10):
    if len(df) < lookback + 2:
        return pd.Series([np.nan]*len(df), index=df.index)
    close = df['close']
    change = (close - close.shift(lookback)).abs()
    volatility = close.diff().abs().rolling(lookback).sum()
    er = change / (volatility + 1e-12)
    return er

def choppiness_index(df: pd.DataFrame, period: int = 14):
    if len(df) < period + 2:
        return pd.Series([np.nan]*len(df), index=df.index)
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    ci = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-12)) / np.log10(period)
    return ci

def zscore(series: pd.Series, window: int = 20):
    if len(series) < window:
        return pd.Series([np.nan]*len(series), index=series.index)
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    return (series - roll_mean) / (roll_std + 1e-12)

def cumulative_delta(df: pd.DataFrame):
    # Approximation assumes presence of 'close' only. Real implementation needs tick data.
    # We approximate by sign of return * volume if volume exists.
    if 'volume' not in df.columns:
        return pd.Series([np.nan]*len(df), index=df.index)
    ret = df['close'].pct_change().fillna(0)
    cd = (np.sign(ret) * df['volume']).cumsum()
    return cd

def orderbook_imbalance(bids: list[tuple[float,float]] | None, asks: list[tuple[float,float]] | None):
    # Placeholder simple imbalance = (bid_vol - ask_vol)/(bid_vol+ask_vol)
    if not bids or not asks:
        return 0.0
    bid_vol = sum(v for _, v in bids)
    ask_vol = sum(v for _, v in asks)
    if bid_vol + ask_vol == 0:
        return 0.0
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)
