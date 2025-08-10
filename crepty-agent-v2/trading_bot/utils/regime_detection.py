"""
Regime Detection Utility - Classifies market as bull, bear, or sideways.
"""
import pandas as pd

def detect_regime(df, fast=20, slow=50, vol_window=20, vol_thresh=0.01):
    if df is None or 'close' not in df or len(df['close']) < slow:
        return 'sideways'
    close = df['close']
    sma_fast = close.rolling(fast).mean().iloc[-1]
    sma_slow = close.rolling(slow).mean().iloc[-1]
    # Volatility as %
    vol = close.pct_change().rolling(vol_window).std().iloc[-1]
    if sma_fast > sma_slow and vol > vol_thresh:
        return 'bull'
    elif sma_fast < sma_slow and vol > vol_thresh:
        return 'bear'
    else:
        return 'sideways'
