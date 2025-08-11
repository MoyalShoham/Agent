"""
RSI Strategy - Buys when RSI < 30, sells when RSI > 70.
"""

import pandas as pd
from trading_bot.utils.ml_signals import generate_ml_signal

def generate_signal(df, rsi_buy=30, rsi_sell=70):
    if df is None or 'close' not in df or len(df['close']) < 15:
        return 'hold'
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    last_rsi = rsi.iloc[-1]

    import logging
    logging.getLogger().info(f"[RSI] last_rsi={last_rsi}")

    # More aggressive: buy if RSI < 45, sell if RSI > 60
    if last_rsi < 45:
        return 'buy'
    elif last_rsi > 60:
        return 'sell'
    return 'hold'
