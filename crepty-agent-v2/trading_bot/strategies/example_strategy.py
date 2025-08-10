"""
Example Strategy Module - All strategies should implement generate_signal(df).
"""
def generate_signal(df):
    # Aggressive: Use 3-period SMA and EMA, fallback to analysis agent if flat
    if df is None or 'close' not in df or len(df['close']) < 3:
        return 'hold'
    close = df['close']
    sma3 = close.rolling(3).mean().iloc[-1]
    ema3 = close.ewm(span=3, adjust=False).mean().iloc[-1]
    last = close.iloc[-1]
    # If price is above both, strong buy; below both, strong sell
    if last > sma3 and last > ema3:
        return 'buy'
    elif last < sma3 and last < ema3:
        return 'sell'
    # If price is above one, below the other, be more active: alternate buy/sell
    elif last > sma3 or last > ema3:
        return 'buy'
    elif last < sma3 or last < ema3:
        return 'sell'
    return 'hold'
