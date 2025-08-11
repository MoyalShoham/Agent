"""
Regime Switcher Module
Detects market regime and switches strategies or risk levels accordingly.
"""
def detect_regime(df):
    # Example: simple regime detection using moving averages
    if df is None or len(df) < 50:
        return 'sideways'
    ma_short = df['close'].rolling(window=20).mean().iloc[-1]
    ma_long = df['close'].rolling(window=50).mean().iloc[-1]
    if ma_short > ma_long * 1.01:
        return 'bull'
    elif ma_short < ma_long * 0.99:
        return 'bear'
    else:
        return 'sideways'
