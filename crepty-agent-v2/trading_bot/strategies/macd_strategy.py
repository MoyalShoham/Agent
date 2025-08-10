"""
MACD Strategy - Buys when MACD crosses above signal, sells when below.
"""
import pandas as pd

def generate_signal(df):
    if df is None or 'close' not in df or len(df['close']) < 26:
        return 'hold'
    close = df['close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        return 'buy'
    elif macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
        return 'sell'
    return 'hold'
