"""
Example Strategy Module - All strategies should implement generate_signal(df).
"""
def generate_signal(df):
    # Simple SMA crossover example
    if df is None or 'close' not in df or len(df['close']) < 5:
        return 'hold'
    if df['close'].iloc[-1] > df['close'].rolling(5).mean().iloc[-1]:
        return 'buy'
    elif df['close'].iloc[-1] < df['close'].rolling(5).mean().iloc[-1]:
        return 'sell'
    return 'hold'
