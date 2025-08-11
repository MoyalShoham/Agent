"""Volatility Compression Breakout Strategy (Squeeze)
Logic:
 - Monitor Donchian channel width and Bollinger Band width contraction.
 - When width below percentile threshold and price breaks above recent high -> buy; below recent low -> sell.
"""
from __future__ import annotations
import pandas as pd
from loguru import logger

NAME = 'volatility_compression_breakout_strategy'

def generate_signal(df: pd.DataFrame, lookback: int = 40, squeeze_window: int = 20, pct_threshold: float = 0.25):
    try:
        if len(df) < max(lookback, squeeze_window, 25) + 5:
            return 'hold'
        closes = df['close']
        highs = df['high']
        lows = df['low']
        recent = df.tail(lookback)
        # Bollinger width
        mid = recent['close'].rolling(20).mean()
        std = recent['close'].rolling(20).std()
        bw = (2*std) / (mid + 1e-9)
        don_h = highs.rolling(squeeze_window).max()
        don_l = lows.rolling(squeeze_window).min()
        don_width = (don_h - don_l) / (closes + 1e-9)
        current_width = don_width.iloc[-1]
        # Only compute quantile if enough non-NaN values
        if bw.notna().sum() < 10:
            perc = False
        else:
            qval = bw.quantile(pct_threshold)
            perc = (bw.iloc[-1] <= qval)
        if perc and current_width == current_width:  # not NaN
            breakout_high = highs.iloc[-2] < don_h.iloc[-2] and closes.iloc[-1] > don_h.iloc[-2]
            breakdown_low = lows.iloc[-2] > don_l.iloc[-2] and closes.iloc[-1] < don_l.iloc[-2]
            if breakout_high:
                return 'buy'
            if breakdown_low:
                return 'sell'
        return 'hold'
    except Exception as e:
        logger.debug(f"{NAME} error: {e}")
        return 'hold'
