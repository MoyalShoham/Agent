"""Futures market metrics collection.
Fetches:
 - Open interest
 - Top long/short account ratio
 - Premium index (basis via markPrice vs indexPrice)
 - Liquidation estimates (placeholder)

Provides caching & periodic refresh via async background task.
"""
from __future__ import annotations
import aiohttp
import asyncio
import time
from typing import Dict, Any
from loguru import logger
from trading_bot.config.settings import settings

BINANCE_FAPI = 'https://fapi.binance.com'

class FuturesMetricsCache:
    def __init__(self):
        self.data: Dict[str, Dict[str, Any]] = {}
        self.last_update: float = 0
        self.interval = settings.OI_METRICS_INTERVAL
        self._lock = asyncio.Lock()

    async def _fetch_json(self, session, url, params=None):
        try:
            async with session.get(url, params=params, timeout=8) as r:
                if r.status != 200:
                    return None
                return await r.json()
        except Exception as e:
            logger.debug(f"futures metrics fetch error {url}: {e}")
            return None

    async def _update_symbol(self, session, symbol: str):
        metrics = {}
        # Open Interest
        oi = await self._fetch_json(session, f"{BINANCE_FAPI}/futures/data/openInterestHist", params={"symbol": symbol, "period": "5m", "limit": 2})
        try:
            if isinstance(oi, list) and len(oi) >= 2:
                latest = float(oi[-1]['sumOpenInterest'])
                prev = float(oi[-2]['sumOpenInterest'])
                change = (latest - prev) / prev if prev else 0
                metrics['open_interest'] = latest
                metrics['open_interest_change_pct'] = change
        except Exception:
            pass
        # Long/Short Account Ratio
        lsr = await self._fetch_json(session, f"{BINANCE_FAPI}/futures/data/topLongShortAccountRatio", params={"symbol": symbol, "period": "5m", "limit": 1})
        try:
            if isinstance(lsr, list) and lsr:
                ratio = float(lsr[-1]['longShortRatio'])
                metrics['long_short_ratio'] = ratio
        except Exception:
            pass
        # Premium Index (basis) -> markPrice - indexPrice
        prem = await self._fetch_json(session, f"{BINANCE_FAPI}/fapi/v1/premiumIndex", params={"symbol": symbol})
        try:
            if isinstance(prem, dict):
                mark_p = float(prem.get('markPrice', 0))
                index_p = float(prem.get('indexPrice', 0))
                if index_p:
                    basis = (mark_p - index_p) / index_p
                    metrics['basis'] = basis
                    metrics['funding_rate'] = float(prem.get('lastFundingRate', 0))
        except Exception:
            pass
        # Liquidations placeholder
        metrics['liquidation_pressure'] = None
        return metrics

    async def refresh(self, symbols):
        async with self._lock:
            now = time.time()
            if now - self.last_update < self.interval:
                return
            try:
                async with aiohttp.ClientSession() as session:
                    tasks = [self._update_symbol(session, s) for s in symbols]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                for sym, res in zip(symbols, results):
                    if isinstance(res, dict):
                        self.data[sym] = res
                self.last_update = now
                logger.debug(f"Futures metrics refreshed for {len(symbols)} symbols")
            except Exception as e:
                logger.error(f"Futures metrics refresh error: {e}")

    def get(self, symbol: str) -> Dict[str, Any]:
        return self.data.get(symbol, {})

futures_metrics_cache = FuturesMetricsCache()
