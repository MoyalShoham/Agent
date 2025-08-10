"""
Bollinger Bands Strategy - Buys when price crosses below lower band, sells when above upper band.
"""
import pandas as pd

def generate_signal(df):
    if df is None or 'close' not in df or len(df['close']) < 20:
        return 'hold'
    close = df['close']
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    last = close.iloc[-1]
    last_upper = upper.iloc[-1]
    last_lower = lower.iloc[-1]
    if last < last_lower:
        return 'buy'
    elif last > last_upper:
        return 'sell'
    return 'hold'
