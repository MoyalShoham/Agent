#!/usr/bin/env python3
"""Asynchronous Binance Futures order book + trades collector.
Stores top N levels snapshots aggregated to 1 second, later resampled to 5m.
CSV storage to stay consistent with project preference.

Environment variables used:
  FUTURES_SYMBOLS (comma separated)
  OB_LEVELS (default 30)
  OB_DATA_DIR (default data/orderbook)
  BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_API_SECRET (only needed if private endpoints added later)

Usage:
  python orderbook_collector.py --symbols BTCUSDT,ETHUSDT --levels 30

Outputs per symbol:
  data/orderbook/{symbol}_raw.csv (append-only 1s snapshots)

We keep in-memory latest book and build 1 second snapshots combining depth diffs and recent trades.
"""
import os
import csv
import json
import time
import asyncio
import argparse
import logging
from datetime import datetime, timezone
from collections import deque
from typing import Dict, List, Tuple, Any

import websockets  # type: ignore
import requests  # new for REST snapshot

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger("orderbook_collector")

BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws"
REST_DEPTH_SNAPSHOT = "https://fapi.binance.com/fapi/v1/depth"

# Helper ---------------------------------------------------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

# Data structures -------------------------------------------------------------

class OrderBook:
    def __init__(self, symbol: str, levels: int):
        self.symbol = symbol.lower()
        self.levels = levels
        self.last_update_id = 0
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_trade: Dict[str, Any] = {"px": None, "qty": None, "side": None, "ts": None}

    def apply_snapshot(self, snapshot: Dict[str, Any]):
        self.last_update_id = snapshot['lastUpdateId']
        self.bids = {float(p): float(q) for p, q in snapshot['bids'][: self.levels] if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in snapshot['asks'][: self.levels] if float(q) > 0}

    def apply_diff(self, diff: Dict[str, Any]):
        if diff['u'] <= self.last_update_id:
            return  # ignore stale
        if diff['U'] <= self.last_update_id + 1 <= diff['u']:
            # process
            for p, q in diff['b']:
                price = float(p); qty = float(q)
                if qty == 0:
                    self.bids.pop(price, None)
                else:
                    self.bids[price] = qty
            for p, q in diff['a']:
                price = float(p); qty = float(q)
                if qty == 0:
                    self.asks.pop(price, None)
                else:
                    self.asks[price] = qty
            self.last_update_id = diff['u']
            # trim to top N
            self._trim()

    def _trim(self):
        # Keep best N levels
        if len(self.bids) > self.levels:
            for p in sorted(self.bids.keys(), reverse=True)[self.levels:]:
                self.bids.pop(p, None)
        if len(self.asks) > self.levels:
            for p in sorted(self.asks.keys())[: -(self.levels)]:
                self.asks.pop(p, None)

    def snapshot_row(self, ts_ms: int) -> Dict[str, Any]:
        if not self.bids or not self.asks:
            return {}
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        # depth imbalance
        bid_levels = sorted(self.bids.keys(), reverse=True)[: self.levels]
        ask_levels = sorted(self.asks.keys())[: self.levels]
        bid_vol = sum(self.bids[p] for p in bid_levels)
        ask_vol = sum(self.asks[p] for p in ask_levels)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
        trade_px = self.last_trade['px']
        trade_side = self.last_trade['side']
        trade_qty = self.last_trade['qty']
        row: Dict[str, Any] = {
            'timestamp': datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
            'symbol': self.symbol.upper(),
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid,
            'spread': spread,
            'spread_bps': (spread / mid) * 10000 if mid else 0,
            'bid_vol_sum': bid_vol,
            'ask_vol_sum': ask_vol,
            'depth_imbalance': imbalance,
            'last_trade_price': trade_px,
            'last_trade_qty': trade_qty,
            'last_trade_side': trade_side,
        }
        # Flatten top 5 levels each side for richer microstructure
        for i, p in enumerate(bid_levels[:5]):
            row[f'bid_px_{i+1}'] = p
            row[f'bid_sz_{i+1}'] = self.bids[p]
        for i, p in enumerate(ask_levels[:5]):
            row[f'ask_px_{i+1}'] = p
            row[f'ask_sz_{i+1}'] = self.asks[p]
        return row

# Collector ------------------------------------------------------------------

async def collect_symbol(symbol: str, levels: int, out_file: str, stop_event: asyncio.Event):
    book = OrderBook(symbol, levels)
    symbol_lower = symbol.lower()
    # Initial REST snapshot (Binance sequencing requirement)
    snapshot_params = {"symbol": symbol.upper(), "limit": max(1000, levels)}
    try:
        r = requests.get(REST_DEPTH_SNAPSHOT, params=snapshot_params, timeout=5)
        r.raise_for_status()
        snap = r.json()
        book.apply_snapshot(snap)
        logger.info(f"Loaded initial snapshot {symbol} lastUpdateId={book.last_update_id}")
    except Exception as e:
        logger.error(f"Failed to load initial snapshot {symbol}: {e}")
    # Streams: depth diffs + trades
    stream = f"{symbol_lower}@depth@100ms/{symbol_lower}@trade"
    url = f"wss://fstream.binance.com/stream?streams={stream}"
    logger.info(f"Connecting {symbol} -> {url}")

    # Ensure file has header
    file_exists = os.path.exists(out_file)
    with open(out_file, 'a', newline='') as f:
        if not file_exists and book.bids and book.asks:
            row = book.snapshot_row(_utc_ms())
            if row:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
                logger.info(f"Wrote initial snapshot row for {symbol}")
    last_written_second = int(time.time()) if (book.bids and book.asks) else None
    buffered_diffs: List[Dict[str, Any]] = []
    snapshot_acknowledged = book.last_update_id != 0

    async for ws in _reconnecting_session(url):
        try:
            async for message in ws:
                data = json.loads(message)
                payload = data.get('data') or data
                stream_type = payload.get('e')
                if stream_type == 'depthUpdate':
                    # Buffer until snapshot applied & sequence satisfied
                    if not snapshot_acknowledged:
                        buffered_diffs.append(payload)
                        if len(buffered_diffs) > 50:
                            buffered_diffs.pop(0)
                    else:
                        book.apply_diff({'U': payload['U'], 'u': payload['u'], 'b': payload['b'], 'a': payload['a']})
                elif stream_type == 'trade':
                    book.last_trade = {
                        'px': float(payload['p']),
                        'qty': float(payload['q']),
                        'side': 'BUY' if payload['m'] is False else 'SELL',
                        'ts': payload['T']
                    }
                # If snapshot loaded and we have buffered diffs not yet applied, process those whose sequence covers last_update_id+1
                if snapshot_acknowledged and buffered_diffs:
                    for diff in list(buffered_diffs):
                        if diff['u'] <= book.last_update_id:
                            buffered_diffs.remove(diff)
                            continue
                        if diff['U'] <= book.last_update_id + 1 <= diff['u']:
                            book.apply_diff({'U': diff['U'], 'u': diff['u'], 'b': diff['b'], 'a': diff['a']})
                            buffered_diffs.remove(diff)
                        else:
                            # Wait for correct diff
                            break
                # Attempt to acknowledge snapshot if not yet and we have enough buffered diffs
                if not snapshot_acknowledged and book.last_update_id != 0 and buffered_diffs:
                    # Find first diff where U <= lastUpdateId+1 <= u
                    for diff in buffered_diffs:
                        if diff['U'] <= book.last_update_id + 1 <= diff['u']:
                            snapshot_acknowledged = True
                            logger.info(f"Snapshot synchronized for {symbol} applying buffered diffs")
                            break
                # 1 second aggregation
                now_sec = int(time.time())
                if last_written_second is None:
                    last_written_second = now_sec
                if now_sec != last_written_second and book.bids and book.asks:
                    row = book.snapshot_row(_utc_ms())
                    if row:
                        write_header = not os.path.exists(out_file) or os.path.getsize(out_file) == 0
                        with open(out_file, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                            if write_header:
                                writer.writeheader()
                            writer.writerow(row)
                    last_written_second = now_sec
                if stop_event.is_set():
                    logger.info(f"Stop signal received for {symbol}")
                    return
        except Exception as e:
            logger.warning(f"Stream error {symbol}: {e}; reconnecting")
            await asyncio.sleep(1)

async def _reconnecting_session(url: str):
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                yield ws
        except Exception as e:
            logger.error(f"Websocket connection failed {url}: {e}; retrying in 2s")
            await asyncio.sleep(2)

async def main_async(symbols: List[str], levels: int, data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    stop_event = asyncio.Event()
    tasks = [asyncio.create_task(collect_symbol(sym, levels, os.path.join(data_dir, f"{sym}_raw.csv"), stop_event)) for sym in symbols]
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt; shutting down")
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', type=str, default=os.environ.get('FUTURES_SYMBOLS', 'BTCUSDT'))
    ap.add_argument('--levels', type=int, default=int(os.environ.get('OB_LEVELS', '30')))
    ap.add_argument('--data-dir', type=str, default=os.environ.get('OB_DATA_DIR', 'data/orderbook'))
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    asyncio.run(main_async(symbols, args.levels, args.data_dir))
