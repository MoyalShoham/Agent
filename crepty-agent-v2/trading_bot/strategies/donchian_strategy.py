import pandas as pd

def generate_signal(df, window=20):
    if len(df) < window:
        return 'hold'
    high = df['high'].rolling(window).max().iloc[-1]
    low = df['low'].rolling(window).min().iloc[-1]
    close = df['close'].iloc[-1]
    if close >= high:
        return 'buy'
    elif close <= low:
        return 'sell'
    else:
        return 'hold'
