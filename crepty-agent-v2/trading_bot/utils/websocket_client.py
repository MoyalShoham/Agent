"""
WebSocket Market Data Client - Real-time price feeds from Binance
Provides low-latency market data for improved execution quality.
"""
import asyncio
import json
import websockets
import threading
from typing import Dict, List, Callable, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from loguru import logger

@dataclass
class TickData:
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None

@dataclass
class OrderBookLevel:
    price: float
    quantity: float

@dataclass
class OrderBookSnapshot:
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime

class BinanceWebSocketClient:
    """Real-time WebSocket client for Binance market data"""
    
    def __init__(self, symbols: List[str], max_reconnect_attempts: int = 5):
        self.symbols = [s.upper() for s in symbols]
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Data storage
        self.latest_prices: Dict[str, float] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        self.tick_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # WebSocket connections
        self.price_ws = None
        self.depth_ws = None
        self.is_running = False
        self.reconnect_count = 0
        
        # Callbacks
        self.price_callbacks: List[Callable[[str, float], None]] = []
        self.orderbook_callbacks: List[Callable[[OrderBookSnapshot], None]] = []
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        
        # Performance metrics
        self.message_count = 0
        self.last_message_time = time.time()
        self.latency_samples = deque(maxlen=1000)
        
        logger.info(f"WebSocket client initialized for {len(self.symbols)} symbols")
    
    def add_price_callback(self, callback: Callable[[str, float], None]):
        """Add callback for price updates"""
        self.price_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable[[OrderBookSnapshot], None]):
        """Add callback for order book updates"""
        self.orderbook_callbacks.append(callback)
    
    def add_tick_callback(self, callback: Callable[[TickData], None]):
        """Add callback for tick data updates"""
        self.tick_callbacks.append(callback)
    
    async def _handle_ticker_message(self, message: str):
        """Handle ticker message from WebSocket"""
        try:
            data = json.loads(message)
            
            if 'c' in data:  # 24hr ticker
                symbol = data['s']
                price = float(data['c'])
                volume = float(data['v'])
                
                # Update latest price
                self.latest_prices[symbol] = price
                self.price_history[symbol].append(price)
                
                # Create tick data
                tick = TickData(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    timestamp=datetime.now()
                )
                
                self.tick_data[symbol].append(tick)
                
                # Call callbacks
                for callback in self.price_callbacks:
                    try:
                        callback(symbol, price)
                    except Exception as e:
                        logger.error(f"Price callback error: {e}")
                
                for callback in self.tick_callbacks:
                    try:
                        callback(tick)
                    except Exception as e:
                        logger.error(f"Tick callback error: {e}")
                
                self.message_count += 1
                
        except Exception as e:
            logger.error(f"Error handling ticker message: {e}")
    
    async def _handle_depth_message(self, message: str):
        """Handle order book depth message"""
        try:
            data = json.loads(message)
            
            if 'bids' in data and 'asks' in data:
                symbol = data['s'] if 's' in data else 'UNKNOWN'
                
                bids = [OrderBookLevel(float(level[0]), float(level[1])) 
                       for level in data['bids'][:10]]  # Top 10 levels
                asks = [OrderBookLevel(float(level[0]), float(level[1])) 
                       for level in data['asks'][:10]]
                
                orderbook = OrderBookSnapshot(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.now()
                )
                
                self.order_books[symbol] = orderbook
                
                # Call callbacks
                for callback in self.orderbook_callbacks:
                    try:
                        callback(orderbook)
                    except Exception as e:
                        logger.error(f"Orderbook callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Error handling depth message: {e}")
    
    async def _connect_ticker_stream(self):
        """Connect to ticker WebSocket stream"""
        streams = [f"{symbol.lower()}@ticker" for symbol in self.symbols]
        url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
        
        try:
            async with websockets.connect(url) as websocket:
                self.price_ws = websocket
                logger.info(f"Connected to ticker stream for {len(self.symbols)} symbols")
                
                async for message in websocket:
                    if not self.is_running:
                        break
                    await self._handle_ticker_message(message)
                    
        except Exception as e:
            logger.error(f"Ticker WebSocket error: {e}")
            if self.is_running and self.reconnect_count < self.max_reconnect_attempts:
                self.reconnect_count += 1
                logger.info(f"Attempting to reconnect ticker stream (attempt {self.reconnect_count})")
                await asyncio.sleep(2 ** self.reconnect_count)  # Exponential backoff
                await self._connect_ticker_stream()
    
    async def _connect_depth_stream(self):
        """Connect to order book depth WebSocket stream"""
        streams = [f"{symbol.lower()}@depth10@100ms" for symbol in self.symbols]
        url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
        
        try:
            async with websockets.connect(url) as websocket:
                self.depth_ws = websocket
                logger.info(f"Connected to depth stream for {len(self.symbols)} symbols")
                
                async for message in websocket:
                    if not self.is_running:
                        break
                    await self._handle_depth_message(message)
                    
        except Exception as e:
            logger.error(f"Depth WebSocket error: {e}")
            if self.is_running and self.reconnect_count < self.max_reconnect_attempts:
                self.reconnect_count += 1
                logger.info(f"Attempting to reconnect depth stream (attempt {self.reconnect_count})")
                await asyncio.sleep(2 ** self.reconnect_count)
                await self._connect_depth_stream()
    
    async def start(self):
        """Start WebSocket connections"""
        self.is_running = True
        self.reconnect_count = 0
        
        # Start both streams concurrently
        tasks = [
            asyncio.create_task(self._connect_ticker_stream()),
            asyncio.create_task(self._connect_depth_stream())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"WebSocket start error: {e}")
    
    def start_background(self):
        """Start WebSocket client in background thread"""
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start())
        
        thread = threading.Thread(target=run_websocket, daemon=True)
        thread.start()
        logger.info("WebSocket client started in background")
    
    def stop(self):
        """Stop WebSocket connections"""
        self.is_running = False
        logger.info("WebSocket client stopping...")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.latest_prices.get(symbol.upper())
    
    def get_price_history(self, symbol: str, limit: int = None) -> List[float]:
        """Get price history for symbol"""
        history = list(self.price_history.get(symbol.upper(), []))
        if limit:
            return history[-limit:]
        return history
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get latest order book for symbol"""
        return self.order_books.get(symbol.upper())
    
    def get_bid_ask_spread(self, symbol: str) -> Optional[float]:
        """Get bid-ask spread for symbol"""
        orderbook = self.get_orderbook(symbol)
        if orderbook and orderbook.bids and orderbook.asks:
            best_bid = orderbook.bids[0].price
            best_ask = orderbook.asks[0].price
            return best_ask - best_bid
        return None
    
    def get_mid_price(self, symbol: str) -> Optional[float]:
        """Get mid price (average of best bid and ask)"""
        orderbook = self.get_orderbook(symbol)
        if orderbook and orderbook.bids and orderbook.asks:
            best_bid = orderbook.bids[0].price
            best_ask = orderbook.asks[0].price
            return (best_bid + best_ask) / 2
        return None
    
    def calculate_vwap(self, symbol: str, period_minutes: int = 60) -> Optional[float]:
        """Calculate Volume Weighted Average Price"""
        ticks = list(self.tick_data.get(symbol.upper(), []))
        if not ticks:
            return None
        
        # Filter ticks within the specified period
        cutoff_time = datetime.now() - timedelta(minutes=period_minutes)
        recent_ticks = [tick for tick in ticks if tick.timestamp > cutoff_time]
        
        if not recent_ticks:
            return None
        
        total_value = sum(tick.price * tick.volume for tick in recent_ticks)
        total_volume = sum(tick.volume for tick in recent_ticks)
        
        if total_volume == 0:
            return None
        
        return total_value / total_volume
    
    def get_price_volatility(self, symbol: str, period_minutes: int = 60) -> Optional[float]:
        """Calculate price volatility over specified period"""
        import numpy as np
        
        prices = self.get_price_history(symbol)
        if len(prices) < 2:
            return None
        
        # Calculate returns
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
        
        if len(returns) == 0:
            return None
        
        return float(np.std(returns))
    
    def get_market_impact_estimate(self, symbol: str, order_size: float) -> Optional[float]:
        """Estimate market impact of an order"""
        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            return None
        
        # Simple market impact estimation based on order book depth
        cumulative_size = 0
        weighted_price = 0
        
        # For buy orders, walk through asks
        if order_size > 0:
            levels = orderbook.asks
        else:
            levels = orderbook.bids
            order_size = abs(order_size)
        
        for level in levels:
            if cumulative_size >= order_size:
                break
            
            level_size = min(level.quantity, order_size - cumulative_size)
            weighted_price += level.price * level_size
            cumulative_size += level_size
        
        if cumulative_size == 0:
            return None
        
        average_fill_price = weighted_price / cumulative_size
        mid_price = self.get_mid_price(symbol)
        
        if mid_price is None:
            return None
        
        # Return impact as percentage
        return abs(average_fill_price - mid_price) / mid_price
    
    def get_statistics(self) -> Dict:
        """Get WebSocket performance statistics"""
        current_time = time.time()
        uptime = current_time - self.last_message_time if self.last_message_time else 0
        
        return {
            'is_running': self.is_running,
            'symbols_count': len(self.symbols),
            'messages_received': self.message_count,
            'uptime_seconds': uptime,
            'reconnect_count': self.reconnect_count,
            'active_symbols': len(self.latest_prices),
            'orderbook_symbols': len(self.order_books),
            'average_latency_ms': sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0
        }

# Global WebSocket client instance
_websocket_client: Optional[BinanceWebSocketClient] = None

def initialize_websocket_client(symbols: List[str]) -> BinanceWebSocketClient:
    """Initialize global WebSocket client"""
    global _websocket_client
    _websocket_client = BinanceWebSocketClient(symbols)
    return _websocket_client

def get_websocket_client() -> Optional[BinanceWebSocketClient]:
    """Get global WebSocket client instance"""
    return _websocket_client

def start_websocket_feeds(symbols: List[str]):
    """Start WebSocket feeds for specified symbols"""
    client = initialize_websocket_client(symbols)
    client.start_background()
    return client
