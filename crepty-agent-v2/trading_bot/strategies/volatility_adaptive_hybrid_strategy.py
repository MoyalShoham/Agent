"""
Volatility Adaptive Hybrid Strategy
Combines breakout (trend) and mean reversion (range) logic based on volatility & regime.
Signals:
- Breakout BUY: close > prior N high + k*ATR and volume_ratio > v_min and squeeze just released
- Breakout SELL: close < prior N low - k*ATR and volume_ratio > v_min
- Mean Reversion BUY: RSI < rsi_low AND price < lower Bollinger AND regime sideways
- Mean Reversion SELL: RSI > rsi_high AND price > upper Bollinger AND regime sideways
Priority: Breakout signals override mean reversion. If conflicting, choose higher strength.
"""
import pandas as pd
import numpy as np
from loguru import logger


def _atr(df: pd.DataFrame, period: int = 14):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def generate_signal(
    df: pd.DataFrame,
    regime: str | None = None,
    lookback: int = 20,
    breakout_k: float = 0.75,
    vol_squeeze_window: int = 20,
    rsi_period: int = 14,
    rsi_low: float = 35,
    rsi_high: float = 65,
    volume_window: int = 20,
    volume_min_ratio: float = 1.4,
):
    try:
        if df is None or not {'close','high','low','volume'}.issubset(df.columns) or len(df) < max(lookback, 60):
            return 'hold'
        work = df.tail(lookback+50).copy()
        close = work['close']
        high = work['high']
        low = work['low']
        volume = work['volume'] if 'volume' in work.columns else pd.Series(np.ones(len(work)), index=work.index)

        # Core indicators
        atr = _atr(work, 14)
        rolling_high = high.rolling(lookback).max()
        rolling_low = low.rolling(lookback).min()

        # Bollinger
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        # Volume ratio
        vol_sma = volume.rolling(volume_window).mean()
        volume_ratio = volume / vol_sma

        # Volatility squeeze detection (Bollinger band width vs its own SMA)
        band_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)
        width_sma = band_width.rolling(vol_squeeze_window).mean()
        squeeze = (band_width < width_sma * 0.9)  # compressed
        squeeze_released = squeeze.shift(1) & (~squeeze)

        last_close = close.iloc[-1]
        last_atr = atr.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_volume_ratio = volume_ratio.iloc[-1]
        last_rolling_high = rolling_high.iloc[-2]  # prior fully formed
        last_rolling_low = rolling_low.iloc[-2]
        last_bb_upper = bb_upper.iloc[-1]
        last_bb_lower = bb_lower.iloc[-1]
        last_squeeze_released = bool(squeeze_released.iloc[-1]) if squeeze_released.notna().any() else False

        breakout_buy = (
            last_close > last_rolling_high + breakout_k * last_atr * 0.5 and
            last_volume_ratio > volume_min_ratio and
            last_squeeze_released
        )
        breakout_sell = (
            last_close < last_rolling_low - breakout_k * last_atr * 0.5 and
            last_volume_ratio > volume_min_ratio and
            last_squeeze_released
        )

        # Mean reversion (only if sideways or unknown regime)
        is_sideways = (regime == 'sideways' or regime is None)
        mean_rev_buy = is_sideways and last_rsi < rsi_low and last_close < last_bb_lower
        mean_rev_sell = is_sideways and last_rsi > rsi_high and last_close > last_bb_upper

        # Priority logic
        if breakout_buy and not breakout_sell:
            return 'buy'
        if breakout_sell and not breakout_buy:
            return 'sell'
        if breakout_buy and breakout_sell:
            # Extremely unlikely; choose direction of larger deviation
            dev_high = (last_close - (last_rolling_high + breakout_k * last_atr * 0.5)) / (last_atr + 1e-9)
            dev_low = ((last_rolling_low - breakout_k * last_atr * 0.5) - last_close) / (last_atr + 1e-9)
            return 'buy' if dev_high > dev_low else 'sell'
        # Fallback to mean reversion
        if mean_rev_buy and not mean_rev_sell:
            return 'buy'
        if mean_rev_sell and not mean_rev_buy:
            return 'sell'
        return 'hold'
    except Exception as e:
        logger.error(f"volatility_adaptive_hybrid_strategy error: {e}")
        return 'hold'
