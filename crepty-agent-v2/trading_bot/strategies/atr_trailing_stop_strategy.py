import pandas as pd
import numpy as np

def generate_signal(df, atr_period=14, multiplier=2):
    required_cols = {'high', 'low', 'close'}
    if len(df) < atr_period or not required_cols.issubset(df.columns):
        return 'hold'
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    # Ensure new columns are not all-NaN and have enough valid data
    h_l = df['H-L']
    h_pc = df['H-PC']
    l_pc = df['L-PC']
    if h_l.isna().all() or h_pc.isna().all() or l_pc.isna().all() or len(h_l.dropna()) < atr_period:
        return 'hold'
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    # Ensure 'TR' exists, is not all-NaN, and has enough data for rolling mean
    if 'TR' not in df.columns or df['TR'].isna().all() or len(df['TR'].dropna()) < atr_period:
        return 'hold'
    df['ATR'] = df['TR'].rolling(atr_period).mean()
    if df['ATR'].isna().all():
        return 'hold'
    trailing_stop = df['close'] - multiplier * df['ATR']
    if pd.isna(trailing_stop.iloc[-1]):
        return 'hold'
    if df['close'].iloc[-1] > trailing_stop.iloc[-1]:
        return 'buy'
    elif df['close'].iloc[-1] < trailing_stop.iloc[-1]:
        return 'sell'
    else:
        return 'hold'
