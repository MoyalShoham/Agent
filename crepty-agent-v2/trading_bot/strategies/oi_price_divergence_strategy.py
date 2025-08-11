"""Open Interest + Price Divergence Strategy
Needs research agent metrics: open_interest_change_pct in StrategyManager.research_state
If price up strongly but OI falling -> potential short squeeze unwind (sell). If price down strongly but OI rising -> potential breakdown (sell) else contrarian buy.
Currently placeholder until research metrics wired.
"""
from __future__ import annotations
import pandas as pd
from loguru import logger

NAME = 'oi_price_divergence_strategy'

def generate_signal(df: pd.DataFrame, momentum_window: int = 12, mom_thresh: float = 0.01, oi_div_thresh: float = 5.0):
    try:
        if len(df) < momentum_window + 2:
            return 'hold'
        close = df['close']
        mom = (close.iloc[-1] - close.iloc[-momentum_window]) / (close.iloc[-momentum_window] + 1e-9)
        # Without OI data, stay neutral
        return 'hold'
    except Exception as e:
        logger.debug(f"{NAME} error: {e}")
        return 'hold'
