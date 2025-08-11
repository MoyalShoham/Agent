import pandas as pd
import numpy as np

def generate_signal(df, atr_period=10, multiplier=3):
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(atr_period).mean()
    df['Upper Basic'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
    df['Lower Basic'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']
    df['Supertrend'] = np.nan
    for i in range(atr_period, len(df)):
        if df['close'].iloc[i] > df['Upper Basic'].iloc[i-1]:
            df['Supertrend'].iloc[i] = df['Lower Basic'].iloc[i]
        elif df['close'].iloc[i] < df['Lower Basic'].iloc[i-1]:
            df['Supertrend'].iloc[i] = df['Upper Basic'].iloc[i]
        else:
            df['Supertrend'].iloc[i] = df['Supertrend'].iloc[i-1]
    if df['close'].iloc[-1] > df['Supertrend'].iloc[-1]:
        return 'buy'
    elif df['close'].iloc[-1] < df['Supertrend'].iloc[-1]:
        return 'sell'
    else:
        return 'hold'
