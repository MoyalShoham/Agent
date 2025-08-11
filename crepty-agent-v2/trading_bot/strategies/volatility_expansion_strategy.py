"""
Volatility Expansion Strategy - Buys on volatility breakout, sells on volatility contraction.
"""
def generate_signal(df, lookback=20, vol_thresh=0.02):
    if df is None or 'close' not in df or len(df['close']) < lookback+1:
        return 'hold'
    close = df['close']
    vol = close.pct_change().rolling(lookback).std()
    if vol.iloc[-2] < vol_thresh and vol.iloc[-1] > vol_thresh:
        return 'buy'
    elif vol.iloc[-2] > vol_thresh and vol.iloc[-1] < vol_thresh:
        return 'sell'
    return 'hold'
