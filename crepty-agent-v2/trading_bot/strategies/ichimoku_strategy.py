import pandas as pd

def generate_signal(df):
    required_cols = {'high', 'low', 'close'}
    # Must have at least 52 rows for all rolling windows, and no NaNs in last 52 rows
    if (
        df is None or
        len(df) < 52 or
        not required_cols.issubset(df.columns) or
        df[['high','low','close']].tail(52).isnull().any().any()
    ):
        return 'hold'
    try:
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        close = df['close']
        # Ensure last values are not NaN
        if pd.isna(senkou_span_a.iloc[-1]) or pd.isna(senkou_span_b.iloc[-1]) or pd.isna(close.iloc[-1]):
            return 'hold'
        if close.iloc[-1] > max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]):
            return 'buy'
        elif close.iloc[-1] < min(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]):
            return 'sell'
        else:
            return 'hold'
    except Exception:
        return 'hold'
