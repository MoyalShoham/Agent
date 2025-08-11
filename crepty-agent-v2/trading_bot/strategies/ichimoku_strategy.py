import pandas as pd

def generate_signal(df):
    if len(df) < 52:
        return 'hold'
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)
    close = df['close']
    if close.iloc[-1] > max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]):
        return 'buy'
    elif close.iloc[-1] < min(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]):
        return 'sell'
    else:
        return 'hold'
