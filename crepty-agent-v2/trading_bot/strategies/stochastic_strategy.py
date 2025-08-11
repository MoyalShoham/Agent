import pandas as pd

def generate_signal(df, k_period=14, d_period=3):
    if len(df) < k_period + d_period:
        return 'hold'
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    if k.iloc[-1] > 80 and d.iloc[-1] > 80:
        return 'sell'
    elif k.iloc[-1] < 20 and d.iloc[-1] < 20:
        return 'buy'
    else:
        return 'hold'
