"""Funding Rate Fade Strategy
Concept: If funding rate extreme positive and price momentum up -> expect mean reversion (sell). Vice versa.
Requires research agent to inject latest funding rate & z-score into StrategyManager.research_state
research_state keys expected: funding_rate, funding_z
"""
from __future__ import annotations
import pandas as pd
from loguru import logger

NAME = 'funding_rate_fade_strategy'

def generate_signal(df: pd.DataFrame, funding_extreme: float = 0.02, z_threshold: float = 2.0):
    try:
        if len(df) < 30:
            return 'hold'
        # Expect research state merged externally; import lazily to avoid cycles
        from trading_bot.strategies.strategy_manager import StrategyManager  # type: ignore
    except Exception:
        pass
    # Fallback: no global access; cannot read research state safely -> hold
    # (In future we could design dependency injection for research metrics.)
    try:
        # Compute simple momentum
        close = df['close']
        mom = (close.iloc[-1] - close.iloc[-6]) / (close.iloc[-6] + 1e-9)
        # Without direct research state, we return hold to avoid false action.
        return 'hold'
    except Exception as e:
        logger.debug(f"{NAME} error: {e}")
        return 'hold'
