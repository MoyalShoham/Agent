"""
Moving Average Crossover Strategy - Buys when fast MA crosses above slow MA, sells when below.
"""
def generate_signal(df, fast=10, slow=30):
    if df is None or 'close' not in df or len(df['close']) < slow:
        return 'hold'
    close = df['close']
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    if fast_ma.iloc[-2] < slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]:
        return 'buy'
    elif fast_ma.iloc[-2] > slow_ma.iloc[-2] and fast_ma.iloc[-1] < slow_ma.iloc[-1]:
        return 'sell'
    return 'hold'
