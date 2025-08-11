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
    import logging
    logging.getLogger().info(f"[MACD] macd={macd.iloc[-1]}, signal={signal.iloc[-1]}")
    # More aggressive: buy if macd > 0, sell if macd < 0
    if macd.iloc[-1] > 0:
        return 'buy'
    elif macd.iloc[-1] < 0:
        return 'sell'
    return 'hold'
