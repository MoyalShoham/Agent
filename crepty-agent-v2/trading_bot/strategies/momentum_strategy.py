"""
Momentum Strategy - Buys if price is above N-period high, sells if below N-period low.
"""
def generate_signal(df, lookback=20):
    if df is None or 'close' not in df or len(df['close']) < lookback:
        return 'hold'
    close = df['close']
    import logging
    max_val = close.rolling(lookback).max().iloc[-2]
    min_val = close.rolling(lookback).min().iloc[-2]
    logging.getLogger().info(f"[Momentum] last={close.iloc[-1]}, max={max_val}, min={min_val}")
    # More aggressive: buy if price > rolling mean, sell if price < rolling mean
    mean_val = close.rolling(lookback).mean().iloc[-2]
    if close.iloc[-1] > mean_val:
        return 'buy'
    elif close.iloc[-1] < mean_val:
        return 'sell'
    return 'hold'
