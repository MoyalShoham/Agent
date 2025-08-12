import pandas as pd
import numpy as np

def generate_signal(df, atr_period=10, multiplier=3):
    """Supertrend strategy signal generator (vectorized, no chained assignment).

    Returns one of 'buy','sell','hold'.
    """
    if df is None or len(df) < atr_period + 2:
        return 'hold'
    df = df.copy()

    # True Range components
    high = df['high']
    low = df['low']
    close = df['close']

    hl = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    mid = (high + low) / 2.0
    upper_basic = mid + multiplier * atr
    lower_basic = mid - multiplier * atr

    supertrend = [np.nan] * len(df)
    for i in range(atr_period, len(df)):
        prev_st = supertrend[i-1]
        prev_upper = upper_basic.iloc[i-1]
        prev_lower = lower_basic.iloc[i-1]
        if close.iloc[i] > prev_upper:
            supertrend[i] = lower_basic.iloc[i]
        elif close.iloc[i] < prev_lower:
            supertrend[i] = upper_basic.iloc[i]
        else:
            supertrend[i] = prev_st
    df.loc[:, 'Supertrend'] = supertrend

    last_st = df['Supertrend'].iloc[-1]
    if np.isnan(last_st):
        return 'hold'
    if close.iloc[-1] > last_st:
        return 'buy'
    if close.iloc[-1] < last_st:
        return 'sell'
    return 'hold'
