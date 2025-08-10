"""
Breakout Strategy - Buys if price breaks above recent resistance, sells if below support.
"""
def generate_signal(df, lookback=20):
    if df is None or 'close' not in df or len(df['close']) < lookback:
        return 'hold'
    close = df['close']
    high = close.rolling(lookback).max().iloc[-2]
    low = close.rolling(lookback).min().iloc[-2]
    last = close.iloc[-1]
    if last > high:
        return 'buy'
    elif last < low:
        return 'sell'
    return 'hold'
