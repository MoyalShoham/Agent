"""
ADX / DMI Strategy
- Computes ADX plus +DI and -DI.
Signals:
  buy  when +DI crosses above -DI and ADX > adx_threshold
  sell when -DI crosses above +DI and ADX > adx_threshold
  else hold
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import logging


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    d = df[['high','low','close']].copy()
    d['prev_high'] = d['high'].shift(1)
    d['prev_low'] = d['low'].shift(1)
    d['+DM'] = np.where((d['high'] - d['prev_high']) > (d['prev_low'] - d['low']),
                        np.maximum(d['high'] - d['prev_high'], 0), 0)
    d['-DM'] = np.where((d['prev_low'] - d['low']) > (d['high'] - d['prev_high']),
                        np.maximum(d['prev_low'] - d['low'], 0), 0)
    d['TR1'] = d['high'] - d['low']
    d['TR2'] = (d['high'] - d['close'].shift(1)).abs()
    d['TR3'] = (d['low'] - d['close'].shift(1)).abs()
    d['TR'] = d[['TR1','TR2','TR3']].max(axis=1)
    # Wilder's smoothing
    tr_s = d['TR'].rolling(period).sum()
    plus_dm_s = d['+DM'].rolling(period).sum()
    minus_dm_s = d['-DM'].rolling(period).sum()
    # Replace with Wilder's smoothing after initial window
    for i in range(period, len(d)):
        tr_s.iat[i] = tr_s.iat[i-1] - (tr_s.iat[i-1] / period) + d['TR'].iat[i]
        plus_dm_s.iat[i] = plus_dm_s.iat[i-1] - (plus_dm_s.iat[i-1] / period) + d['+DM'].iat[i]
        minus_dm_s.iat[i] = minus_dm_s.iat[i-1] - (minus_dm_s.iat[i-1] / period) + d['-DM'].iat[i]
    d['+DI'] = 100 * (plus_dm_s / tr_s)
    d['-DI'] = 100 * (minus_dm_s / tr_s)
    di_diff = (d['+DI'] - d['-DI']).abs()
    di_sum = (d['+DI'] + d['-DI']).replace(0, np.nan)
    d['DX'] = 100 * (di_diff / di_sum)
    d['ADX'] = d['DX'].rolling(period).mean()
    return d


def generate_signal(df: pd.DataFrame, period: int = 14, adx_threshold: float = 20.0) -> str:
    if df is None or len(df) < period + 2:
        return 'hold'
    try:
        adx_df = _compute_adx(df.tail(5*period))  # limit calc size
        last = adx_df.iloc[-1]
        prev = adx_df.iloc[-2]
        adx = last['ADX']
        plus_di = last['+DI']
        minus_di = last['-DI']
        crossed_up = (prev['+DI'] <= prev['-DI']) and (plus_di > minus_di)
        crossed_down = (prev['-DI'] <= prev['+DI']) and (minus_di > plus_di)
        logging.getLogger().info(f"[ADX] ADX={adx:.2f} +DI={plus_di:.2f} -DI={minus_di:.2f}")
        if adx >= adx_threshold:
            if crossed_up:
                return 'buy'
            if crossed_down:
                return 'sell'
        return 'hold'
    except Exception as e:
        logging.getLogger().exception(f"adx_strategy error: {e}")
        return 'hold'
