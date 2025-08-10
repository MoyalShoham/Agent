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

    # ML signal integration for risk control and confirmation
    ml_signal = generate_ml_signal(df)
    if ml_signal == 'high_risk':
        return 'hold'  # Block trade if ML says high risk

    if last_rsi < rsi_buy and (ml_signal in ['buy', 'hold']):
        return 'buy'
    elif last_rsi > rsi_sell and (ml_signal in ['sell', 'hold']):
        return 'sell'
    return 'hold'
