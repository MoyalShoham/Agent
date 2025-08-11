"""Supertrend + VWAP Pullback Strategy
Entry long: price above supertrend & pulls back to near VWAP (within tolerance) while higher timeframe trend up.
Entry short: mirror logic.
Currently simplified (no HTF) due to limited context.
"""
from __future__ import annotations
import pandas as pd
from loguru import logger
from trading_bot.indicators import supertrend, vwap

NAME = 'supertrend_vwap_pullback_strategy'

def generate_signal(df: pd.DataFrame, st_period: int = 10, st_mult: float = 2.5, vwap_look: int = 40, tolerance: float = 0.002):
    try:
        if len(df) < max(st_period + 5, vwap_look):
            return 'hold'
        st = supertrend(df, period=st_period, multiplier=st_mult)
        vw = vwap(df.tail(vwap_look))
        close = df['close'].iloc[-1]
        st_val = st.iloc[-1]
        vw_val = vw.iloc[-1]
        if pd.isna(st_val) or pd.isna(vw_val):
            return 'hold'
        dist = (close - vw_val) / vw_val
        # Trend up if close > supertrend
        if close > st_val and abs(dist) <= tolerance and close > vw_val:
            return 'buy'
        if close < st_val and abs(dist) <= tolerance and close < vw_val:
            return 'sell'
        return 'hold'
    except Exception as e:
        logger.debug(f"{NAME} error: {e}")
        return 'hold'
