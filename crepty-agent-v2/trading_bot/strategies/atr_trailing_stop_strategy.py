import pandas as pd
import numpy as np

def generate_signal(df, atr_period=14, multiplier=2):
    if len(df) < atr_period:
        return 'hold'
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(atr_period).mean()
    trailing_stop = df['close'] - multiplier * df['ATR']
    if df['close'].iloc[-1] > trailing_stop.iloc[-1]:
        return 'buy'
    elif df['close'].iloc[-1] < trailing_stop.iloc[-1]:
        return 'sell'
    else:
        return 'hold'
