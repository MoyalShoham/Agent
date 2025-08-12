"""External data integrations for secondary crypto data sources and derivatives metrics.
Provides:
 - CoinGecko spot market data (price, market cap, volume, categories)
 - Binance futures funding rate utilities (batch)
 - Helper to merge into internal research snapshots
"""
from __future__ import annotations
import aiohttp
from loguru import logger
from typing import Dict, Any, List
from trading_bot.config.settings import settings
import asyncio

COINGECKO_BASE = settings.COINGECKO_API_BASE.rstrip('/')

async def _fetch_json(session: aiohttp.ClientSession, url: str, params=None) -> Any:
    try:
        async with session.get(url, params=params, timeout=8) as r:
            if r.status != 200:
                logger.debug(f"ext data non-200 {r.status} {url}")
                return None
            return await r.json()
    except Exception as e:
        logger.debug(f"ext data fetch error {url}: {e}")
        return None

async def fetch_coingecko_market(symbols: List[str]) -> Dict[str, Any]:
    """Fetch market data for a list of symbols (expects symbols like BTCUSDT -> btc)."""
    if not symbols:
        return {}
    # Map symbols to CoinGecko ids heuristically (user can extend mapping)
    id_map = {}
    for s in symbols:
        base = s.replace('USDT','').replace('USDC','').lower()
        id_map[s] = base
    vs_currency = 'usd'
    ids = ','.join(set(id_map.values()))
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {"vs_currency": vs_currency, "ids": ids, "price_change_percentage": "1h,24h,7d"}
    async with aiohttp.ClientSession() as session:
        data = await _fetch_json(session, url, params=params)
    out: Dict[str, Any] = {}
    if isinstance(data, list):
        for coin in data:
            cid = coin.get('id')
            for sym, mapped in id_map.items():
                if mapped == cid:
                    out[sym] = {
                        'cg_price': coin.get('current_price'),
                        'cg_mcap': coin.get('market_cap'),
                        'cg_vol24h': coin.get('total_volume'),
                        'cg_change_1h': coin.get('price_change_percentage_1h_in_currency'),
                        'cg_change_24h': coin.get('price_change_percentage_24h_in_currency'),
                        'cg_change_7d': coin.get('price_change_percentage_7d_in_currency'),
                        'cg_category': coin.get('category')
                    }
    return out

async def fetch_futures_funding(symbols: List[str]) -> Dict[str, Any]:
    """Fetch latest funding rates using Binance public endpoint (no key)."""
    if not symbols:
        return {}
    url = 'https://fapi.binance.com/fapi/v1/premiumIndex'
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=8) as r:
                if r.status != 200:
                    return {}
                data = await r.json()
        except Exception as e:
            logger.debug(f"funding fetch error: {e}")
            return {}
    out: Dict[str, Any] = {}
    for item in data:
        sym = item.get('symbol')
        if sym in symbols:
            try:
                fr = float(item.get('lastFundingRate', 0.0))
            except Exception:
                fr = None
            out[sym] = {
                'funding_rate': fr,
                'next_funding_time': item.get('nextFundingTime')
            }
    return out

async def enrich_research_with_external(base_snapshot: Dict[str, Any], symbols: List[str]) -> Dict[str, Any]:
    try:
        coingecko_task = fetch_coingecko_market(symbols)
        funding_task = fetch_futures_funding(symbols)
        cg, fr = await asyncio.gather(coingecko_task, funding_task)
        # Merge CG data into each symbol entry keyed maybe by symbol
        base_snapshot = base_snapshot.copy()
        base_snapshot['coingecko'] = cg
        base_snapshot['funding'] = fr
        return base_snapshot
    except Exception as e:
        logger.error(f"enrich external error: {e}")
        return base_snapshot

async def funding_position_scale(symbol: str, funding_rate: float | None) -> float:
    """Return a scale modifier based on funding rate extremes."""
    if funding_rate is None:
        return 1.0
    try:
        if funding_rate > settings.FUNDING_POSITIVE_THRESHOLD:
            return settings.FUNDING_SIZE_REDUCTION
        if funding_rate < settings.FUNDING_NEGATIVE_THRESHOLD:
            return settings.FUNDING_SIZE_BONUS
    except Exception:
        pass
    return 1.0
