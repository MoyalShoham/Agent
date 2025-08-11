"""Research Data Sources - asynchronous fetchers for external metrics.
Placeholders included; replace with real API calls and add keys to settings/env.
"""
from __future__ import annotations
import asyncio
import aiohttp
from loguru import logger
from typing import Dict, Any

DEFAULT_TIMEOUT = 8

async def _fetch_json(session: aiohttp.ClientSession, url: str, headers=None) -> Any:
    try:
        async with session.get(url, timeout=DEFAULT_TIMEOUT, headers=headers) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception as e:
        logger.debug(f"fetch error {url}: {e}")
        return None

async def fetch_fear_greed(session):
    data = await _fetch_json(session, 'https://api.alternative.me/fng/?limit=1')
    try:
        if data and 'data' in data:
            v = data['data'][0]
            return {'fear_greed_value': float(v['value'])}
    except Exception:
        pass
    return {'fear_greed_value': None}

async def fetch_coindesk_headlines(session):
    # Placeholder: would parse RSS -> sentiment.
    return {'news_headlines': []}

async def fetch_combined_research() -> Dict[str, Any]:
    tasks = []
    async with aiohttp.ClientSession() as session:
        tasks.append(fetch_fear_greed(session))
        tasks.append(fetch_coindesk_headlines(session))
        results = await asyncio.gather(*tasks, return_exceptions=True)
    combined: Dict[str, Any] = {}
    for r in results:
        if isinstance(r, dict):
            combined.update(r)
    return combined
