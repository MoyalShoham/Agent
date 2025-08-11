"""Regime Adaptive Switch Strategy
Chooses momentum bias in bull, mean reversion in sideways, protective in bear.
Simplified: Uses moving averages already enriched.
"""
from __future__ import annotations
import pandas as pd
from loguru import logger
from trading_bot.utils.regime_detection import detect_regime

NAME = 'regime_adaptive_switch_strategy'

def generate_signal(df: pd.DataFrame, fast: int = 10, slow: int = 40):
    try:
        if len(df) < slow + 2:
            return 'hold'
        regime = detect_regime(df)
        close = df['close']
        sma_fast = close.rolling(fast).mean().iloc[-1]
        sma_slow = close.rolling(slow).mean().iloc[-1]
        if regime == 'bull':
            if sma_fast > sma_slow:
                return 'buy'
        elif regime == 'bear':
            if sma_fast < sma_slow:
                return 'sell'
        else:  # sideways mean reversion attempt
            price = close.iloc[-1]
            mid = close.rolling(20).mean().iloc[-1]
            if price < mid * 0.995:
                return 'buy'
            if price > mid * 1.005:
                return 'sell'
        return 'hold'
    except Exception as e:
        logger.debug(f"{NAME} error: {e}")
        return 'hold'
