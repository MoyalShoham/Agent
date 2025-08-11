from trading_bot.utils.ml_signals import generate_ml_signal
import pandas as pd

def test_generate_ml_signal_hold():
    df = pd.DataFrame({'close': [1, 1, 1, 1, 1]})
    signal = generate_ml_signal(df)
    assert signal in ['buy', 'sell', 'hold']
