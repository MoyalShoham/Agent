"""
Mean Reversion Strategy - Buys if price is below N-period mean - K*std, sells if above mean + K*std.
"""
def generate_signal(df, lookback=20, k=2):
    if df is None or 'close' not in df or len(df['close']) < lookback:
        return 'hold'
    close = df['close']
    mean = close.rolling(lookback).mean().iloc[-1]
    std = close.rolling(lookback).std().iloc[-1]
    last = close.iloc[-1]
    if last < mean - k*std:
        return 'buy'
    elif last > mean + k*std:
        return 'sell'
    return 'hold'
