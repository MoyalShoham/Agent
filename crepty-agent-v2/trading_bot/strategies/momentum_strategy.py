"""
Momentum Strategy - Buys if price is above N-period high, sells if below N-period low.
"""
def generate_signal(df, lookback=20):
    if df is None or 'close' not in df or len(df['close']) < lookback:
        return 'hold'
    close = df['close']
    if close.iloc[-1] >= close.rolling(lookback).max().iloc[-2]:
        return 'buy'
    elif close.iloc[-1] <= close.rolling(lookback).min().iloc[-2]:
        return 'sell'
    return 'hold'
