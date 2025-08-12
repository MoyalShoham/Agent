"""
Regime Detection Utility - Classifies market as bull, bear, or sideways.
"""
import pandas as pd

def detect_regime(df, fast=20, slow=50, vol_window=20, vol_thresh=0.01):
    if df is None or 'close' not in df or len(df['close']) < slow:
        return 'sideways'
    close = df['close']
    # Only use the last needed values for rolling calculations for speed
    sma_fast = close.tail(fast*2).rolling(fast).mean().iloc[-1]
    sma_slow = close.tail(slow*2).rolling(slow).mean().iloc[-1]
    vol = close.tail(vol_window*2).pct_change().rolling(vol_window).std().iloc[-1]
    if sma_fast > sma_slow and vol > vol_thresh:
        return 'bull'
    elif sma_fast < sma_slow and vol > vol_thresh:
        return 'bear'
    else:
        return 'sideways'
