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
    last_sma = sma.iloc[-1]
    import logging
    logging.getLogger().info(f"[Bollinger] last={last}, upper={last_upper}, lower={last_lower}, sma={last_sma}")
    # More aggressive: buy if price < sma, sell if price > sma
    if last < last_sma:
        return 'buy'
    elif last > last_sma:
        return 'sell'
    return 'hold'
