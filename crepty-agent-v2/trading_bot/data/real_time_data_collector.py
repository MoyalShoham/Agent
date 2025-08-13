"""
Real-time Crypto Data Collector
Handles WebSocket streams, REST API calls, and data aggregation from multiple sources
"""

import asyncio
import json
import websockets
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from loguru import logger
import gzip
from pathlib import Path
import sqlite3
from collections import defaultdict, deque
import time

@dataclass
class MarketData:
    """Unified market data structure"""
    symbol: str
    timestamp: int
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: Optional[str] = None
    
@dataclass
class OrderBookData:
    """Order book data structure"""
    symbol: str
    timestamp: int
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]
    
@dataclass
class FundingData:
    """Funding rate data"""
    symbol: str
    timestamp: int
    funding_rate: float
    next_funding_time: int
    
@dataclass
class OpenInterestData:
    """Open interest data"""
    symbol: str
    timestamp: int
    open_interest: float
    
class BinanceWSDataCollector:
    """Binance WebSocket data collector for futures"""
    
    def __init__(self, symbols: List[str], callbacks: Dict[str, Callable] = None):
        self.symbols = [s.lower() for s in symbols]
        self.callbacks = callbacks or {}
        self.ws_url = "wss://fstream.binance.com/ws"
        self.connections = {}
        self.running = False
        
        # Data buffers
        self.trade_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.orderbook_buffer = defaultdict(lambda: deque(maxlen=100))
        self.funding_buffer = defaultdict(lambda: deque(maxlen=24))  # 24 hours
        self.liquidation_buffer = defaultdict(lambda: deque(maxlen=500))
        
    async def start(self):
        """Start all WebSocket connections"""
        self.running = True
        logger.info(f"Starting Binance WS data collector for {len(self.symbols)} symbols")
        
        # Create tasks for different stream types
        tasks = [
            self._connect_aggTrade_stream(),
            self._connect_depth_stream(), 
            self._connect_mark_price_stream(),
            self._connect_liquidation_stream(),
            self._connect_funding_stream()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _connect_aggTrade_stream(self):
        """Connect to aggregated trade stream"""
        streams = [f"{symbol}@aggTrade" for symbol in self.symbols]
        stream_url = f"{self.ws_url}/{'/'.join(streams)}"
        
        try:
            async with websockets.connect(stream_url) as websocket:
                logger.info("✅ Connected to Binance aggTrade stream")
                async for message in websocket:
                    if not self.running:
                        break
                    await self._handle_trade_message(json.loads(message))
        except Exception as e:
            logger.error(f"❌ Trade stream error: {e}")
            
    async def _connect_depth_stream(self):
        """Connect to order book depth stream"""
        streams = [f"{symbol}@depth@100ms" for symbol in self.symbols]
        stream_url = f"{self.ws_url}/{'/'.join(streams)}"
        
        try:
            async with websockets.connect(stream_url) as websocket:
                logger.info("✅ Connected to Binance depth stream")
                async for message in websocket:
                    if not self.running:
                        break
                    await self._handle_depth_message(json.loads(message))
        except Exception as e:
            logger.error(f"❌ Depth stream error: {e}")
            
    async def _connect_mark_price_stream(self):
        """Connect to mark price stream"""
        stream_url = f"{self.ws_url}/!markPrice@arr@1s"
        
        try:
            async with websockets.connect(stream_url) as websocket:
                logger.info("✅ Connected to Binance mark price stream")
                async for message in websocket:
                    if not self.running:
                        break
                    await self._handle_mark_price_message(json.loads(message))
        except Exception as e:
            logger.error(f"❌ Mark price stream error: {e}")
            
    async def _connect_liquidation_stream(self):
        """Connect to liquidation stream"""
        stream_url = f"{self.ws_url}/!forceOrder@arr"
        
        try:
            async with websockets.connect(stream_url) as websocket:
                logger.info("✅ Connected to Binance liquidation stream")
                async for message in websocket:
                    if not self.running:
                        break
                    await self._handle_liquidation_message(json.loads(message))
        except Exception as e:
            logger.error(f"❌ Liquidation stream error: {e}")
            
    async def _connect_funding_stream(self):
        """Connect to funding rate updates"""
        # This is typically updated via REST API every 8 hours
        while self.running:
            try:
                await self._fetch_funding_rates()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"❌ Funding rate fetch error: {e}")
                await asyncio.sleep(60)
                
    async def _handle_trade_message(self, data):
        """Handle aggregated trade messages"""
        if 'stream' in data:
            stream_data = data['data']
            symbol = stream_data['s']
            
            trade = MarketData(
                symbol=symbol,
                timestamp=stream_data['T'],
                price=float(stream_data['p']),
                volume=float(stream_data['q']),
                side='buy' if stream_data['m'] else 'sell',
                trade_id=str(stream_data['a'])
            )
            
            self.trade_buffer[symbol].append(trade)
            
            # Call registered callbacks
            if 'trade' in self.callbacks:
                await self.callbacks['trade'](trade)
                
    async def _handle_depth_message(self, data):
        """Handle order book depth messages"""
        if 'stream' in data:
            stream_data = data['data']
            symbol = stream_data['s']
            
            orderbook = OrderBookData(
                symbol=symbol,
                timestamp=stream_data['T'],
                bids=[[float(p), float(q)] for p, q in stream_data['b']],
                asks=[[float(p), float(q)] for p, q in stream_data['a']]
            )
            
            self.orderbook_buffer[symbol].append(orderbook)
            
            if 'orderbook' in self.callbacks:
                await self.callbacks['orderbook'](orderbook)
                
    async def _handle_mark_price_message(self, data):
        """Handle mark price messages"""
        if isinstance(data, list):
            for item in data:
                symbol = item['s']
                if symbol.lower() in self.symbols:
                    mark_price_data = {
                        'symbol': symbol,
                        'timestamp': item['T'],
                        'mark_price': float(item['p']),
                        'funding_rate': float(item['r']),
                        'next_funding_time': item['T']
                    }
                    
                    if 'mark_price' in self.callbacks:
                        await self.callbacks['mark_price'](mark_price_data)
                        
    async def _handle_liquidation_message(self, data):
        """Handle liquidation messages"""
        if isinstance(data, list):
            for item in data:
                liquidation_data = {
                    'symbol': item['s'],
                    'side': item['S'],
                    'order_type': item['o'],
                    'time_in_force': item['f'],
                    'quantity': float(item['q']),
                    'price': float(item['p']),
                    'average_price': float(item['ap']),
                    'order_status': item['X'],
                    'order_last_filled_quantity': float(item['l']),
                    'order_filled_accumulated_quantity': float(item['z']),
                    'order_trade_time': item['T']
                }
                
                if 'liquidation' in self.callbacks:
                    await self.callbacks['liquidation'](liquidation_data)
                    
    async def _fetch_funding_rates(self):
        """Fetch current funding rates via REST API"""
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        for item in data:
                            symbol = item['symbol']
                            if symbol.lower() in self.symbols:
                                funding = FundingData(
                                    symbol=symbol,
                                    timestamp=int(item['time']),
                                    funding_rate=float(item['lastFundingRate']),
                                    next_funding_time=int(item['nextFundingTime'])
                                )
                                
                                self.funding_buffer[symbol].append(funding)
                                
                                if 'funding' in self.callbacks:
                                    await self.callbacks['funding'](funding)
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            
    def get_latest_trade(self, symbol: str) -> Optional[MarketData]:
        """Get latest trade for symbol"""
        buffer = self.trade_buffer.get(symbol.upper())
        return buffer[-1] if buffer else None
        
    def get_latest_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """Get latest order book for symbol"""
        buffer = self.orderbook_buffer.get(symbol.upper())
        return buffer[-1] if buffer else None
        
    def get_trade_history(self, symbol: str, count: int = 100) -> List[MarketData]:
        """Get recent trade history"""
        buffer = self.trade_buffer.get(symbol.upper(), deque())
        return list(buffer)[-count:]
        
    def stop(self):
        """Stop data collection"""
        self.running = False
        logger.info("🛑 Stopping Binance WS data collector")

class AggregatorDataCollector:
    """Data collector for aggregator APIs (CoinGecko, CryptoCompare, etc.)"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def fetch_coingecko_prices(self, coin_ids: List[str]) -> Dict:
        """Fetch prices from CoinGecko API"""
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"CoinGecko API error: {e}")
            
        return {}
        
    async def fetch_cryptocompare_data(self, symbols: List[str]) -> Dict:
        """Fetch data from CryptoCompare"""
        url = "https://min-api.cryptocompare.com/data/pricemultifull"
        params = {
            'fsyms': ','.join(symbols),
            'tsyms': 'USD'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"CryptoCompare API error: {e}")
            
        return {}

class DataStorage:
    """SQLite-based data storage for historical analysis"""
    
    def __init__(self, db_path: str = "crypto_data.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    side TEXT NOT NULL,
                    trade_id TEXT
                )
            ''')
            
            # Order book snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    best_bid REAL NOT NULL,
                    best_ask REAL NOT NULL,
                    bid_volume REAL NOT NULL,
                    ask_volume REAL NOT NULL,
                    spread REAL NOT NULL
                )
            ''')
            
            # Funding rates
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS funding_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    funding_rate REAL NOT NULL,
                    next_funding_time INTEGER NOT NULL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_timestamp ON orderbook_snapshots(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_funding_symbol_timestamp ON funding_rates(symbol, timestamp)')
            
            conn.commit()
            
    def store_trade(self, trade: MarketData):
        """Store trade data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades (symbol, timestamp, price, volume, side, trade_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (trade.symbol, trade.timestamp, trade.price, trade.volume, trade.side, trade.trade_id))
            conn.commit()
            
    def store_orderbook_snapshot(self, orderbook: OrderBookData):
        """Store order book snapshot"""
        if not orderbook.bids or not orderbook.asks:
            return
            
        best_bid = orderbook.bids[0]
        best_ask = orderbook.asks[0]
        spread = (best_ask[0] - best_bid[0]) / best_bid[0]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO orderbook_snapshots 
                (symbol, timestamp, best_bid, best_ask, bid_volume, ask_volume, spread)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (orderbook.symbol, orderbook.timestamp, best_bid[0], best_ask[0], 
                  best_bid[1], best_ask[1], spread))
            conn.commit()
            
    def store_funding_rate(self, funding: FundingData):
        """Store funding rate data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO funding_rates (symbol, timestamp, funding_rate, next_funding_time)
                VALUES (?, ?, ?, ?)
            ''', (funding.symbol, funding.timestamp, funding.funding_rate, funding.next_funding_time))
            conn.commit()
            
    def get_recent_trades(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get recent trades as DataFrame"""
        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
        
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT symbol, timestamp, price, volume, side
                FROM trades
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            return pd.read_sql_query(query, conn, params=(symbol, start_time))

class RealTimeDataManager:
    """Main coordinator for all data collection"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.storage = DataStorage()
        self.binance_collector = None
        self.running = False
        
        # Set up callbacks
        self.callbacks = {
            'trade': self._on_trade,
            'orderbook': self._on_orderbook,
            'funding': self._on_funding,
            'mark_price': self._on_mark_price,
            'liquidation': self._on_liquidation
        }
        
    async def start(self):
        """Start all data collection"""
        self.running = True
        logger.info("🚀 Starting Real-Time Data Manager")
        
        # Initialize Binance collector
        self.binance_collector = BinanceWSDataCollector(self.symbols, self.callbacks)
        
        # Start collection tasks
        tasks = [
            self.binance_collector.start(),
            self._periodic_aggregator_fetch()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _on_trade(self, trade: MarketData):
        """Handle incoming trade data"""
        self.storage.store_trade(trade)
        # Add real-time analysis here
        
    async def _on_orderbook(self, orderbook: OrderBookData):
        """Handle incoming order book data"""
        self.storage.store_orderbook_snapshot(orderbook)
        # Add order flow analysis here
        
    async def _on_funding(self, funding: FundingData):
        """Handle funding rate updates"""
        self.storage.store_funding_rate(funding)
        # Add funding analysis here
        
    async def _on_mark_price(self, mark_price_data: Dict):
        """Handle mark price updates"""
        # Add mark price analysis here
        pass
        
    async def _on_liquidation(self, liquidation_data: Dict):
        """Handle liquidation events"""
        # Add liquidation analysis here
        logger.info(f"Liquidation: {liquidation_data['symbol']} {liquidation_data['side']} ${liquidation_data['quantity']:.2f}")
        
    async def _periodic_aggregator_fetch(self):
        """Periodically fetch data from aggregators"""
        while self.running:
            try:
                async with AggregatorDataCollector() as collector:
                    # Fetch CoinGecko data
                    coin_ids = ['bitcoin', 'ethereum', 'binancecoin']  # Map symbols to CoinGecko IDs
                    coingecko_data = await collector.fetch_coingecko_prices(coin_ids)
                    
                    # Fetch CryptoCompare data
                    base_symbols = [s.replace('USDT', '') for s in self.symbols]
                    cryptocompare_data = await collector.fetch_cryptocompare_data(base_symbols)
                    
                    logger.debug(f"Fetched aggregator data: CG={len(coingecko_data)}, CC={bool(cryptocompare_data)}")
                    
            except Exception as e:
                logger.error(f"Aggregator fetch error: {e}")
                
            await asyncio.sleep(300)  # Every 5 minutes
            
    def stop(self):
        """Stop all data collection"""
        self.running = False
        if self.binance_collector:
            self.binance_collector.stop()
        logger.info("🛑 Real-Time Data Manager stopped")
        
    def get_latest_data(self, symbol: str) -> Dict:
        """Get latest data for a symbol"""
        if not self.binance_collector:
            return {}
            
        return {
            'trade': self.binance_collector.get_latest_trade(symbol),
            'orderbook': self.binance_collector.get_latest_orderbook(symbol)
        }
