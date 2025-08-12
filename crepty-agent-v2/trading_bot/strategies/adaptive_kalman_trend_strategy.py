"""
Adaptive Kalman Trend Strategy
 - Apply a simple Kalman filter to smooth price.
 - Buy when raw price crosses above filtered (smoothed) price after being below.
 - Sell when raw price crosses below filtered price after being above.
Note: This is a lightweight approximation (not tuned for optimal state-space parameters).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging

def _kalman(series: pd.Series, process_var: float = 1e-3, measure_var: float = 1e-2):
    # Simple 1D Kalman
    n = len(series)
    if n == 0:
        return series
    x = series.iloc[0]
    p = 1.0
    filtered = [x]
    for z in series.iloc[1:]:
        # predict
        p = p + process_var
        # update
        k = p / (p + measure_var)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered.append(x)
    return pd.Series(filtered, index=series.index)


def generate_signal(df: pd.DataFrame, lookback: int = 200, process_var: float = 1e-4, measure_var: float = 1e-2) -> str:
    if df is None or len(df) < 50:
        return 'hold'
    d = df.tail(lookback).copy()
    price = d['close']
    filt = _kalman(price, process_var=process_var, measure_var=measure_var)
    last = price.iloc[-1]
    prev = price.iloc[-2]
    last_f = filt.iloc[-1]
    prev_f = filt.iloc[-2]
    logging.getLogger().info(f"[Kalman] price={last:.2f} filt={last_f:.2f}")
    cross_up = prev <= prev_f and last > last_f
    cross_down = prev >= prev_f and last < last_f
    if cross_up:
        return 'buy'
    if cross_down:
        return 'sell'
    return 'hold'
