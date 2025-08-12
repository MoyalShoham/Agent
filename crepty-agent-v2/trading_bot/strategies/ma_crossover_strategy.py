"""
Moving Average Crossover Strategy - Buys when fast MA crosses above slow MA, sells when below.
"""
import pandas as pd
def generate_signal(df, fast=10, slow=30):
    if df is None or 'close' not in df or len(df['close']) < slow:
        return 'hold'
    close = df['close']
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    # Ensure rolling means are valid (not all NaN)
    if pd.isna(fast_ma.iloc[-1]) or pd.isna(slow_ma.iloc[-1]):
        return 'hold'
    import logging
    logging.getLogger().info(f"[MA Crossover] fast_ma={fast_ma.iloc[-1]}, slow_ma={slow_ma.iloc[-1]}")
    # More aggressive: buy if fast_ma > slow_ma, sell if fast_ma < slow_ma
    if fast_ma.iloc[-1] > slow_ma.iloc[-1]:
        return 'buy'
    elif fast_ma.iloc[-1] < slow_ma.iloc[-1]:
        return 'sell'
    return 'hold'
