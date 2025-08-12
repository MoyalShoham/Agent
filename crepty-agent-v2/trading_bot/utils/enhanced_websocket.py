"""
Enhanced WebSocket Client with Optimized Real-Time Data Processing
Implements high-performance WebSocket connections with smart reconnection,
data buffering, and intelligent symbol management.
"""
import asyncio
import websockets
import json
import gzip
from typing import Dict, List, Callable, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time
import numpy as np
from loguru import logger
import threading
from concurrent.futures import ThreadPoolExecutor
import ssl
import certifi

@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/"
    max_connections: int = 5
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10
    ping_interval: int = 20
    ping_timeout: int = 10
    message_queue_size: int = 1000
    data_buffer_size: int = 100
    compression_enabled: bool = True
    heartbeat_interval: int = 30

@dataclass
class MarketDataPoint:
    """Market data structure"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    change_24h: float = 0.0
    volume_24h: float = 0.0
    trades_count: int = 0

@dataclass
class ConnectionStats:
    """Connection statistics"""
    connection_id: str
    connected_at: datetime
    messages_received: int = 0
    messages_processed: int = 0
    reconnections: int = 0
    last_message_time: Optional[datetime] = None
    latency_ms: float = 0.0
    status: str = "disconnected"

class EnhancedWebSocketClient:
    """
    High-performance WebSocket client with advanced features:
    - Smart connection pooling
    - Intelligent symbol batching
    - Real-time data buffering
    - Automatic reconnection with backoff
    - Performance monitoring
    """
    
    def __init__(self, config: WebSocketConfig = None):
        self.config = config or WebSocketConfig()
        
        # Connection management
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.connection_stats: Dict[str, ConnectionStats] = {}
        self.symbol_to_connection: Dict[str, str] = {}
        
        # Data management
        self.data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.data_buffer_size))
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.message_queue_size)
        self.subscribed_symbols: Set[str] = set()
        
        # Callbacks
        self.data_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        self.connection_callbacks: List[Callable] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_messages': 0,
            'messages_per_second': 0.0,
            'average_latency': 0.0,
            'connection_uptime': 0.0,
            'data_loss_events': 0,
            'reconnection_events': 0
        }
        
        # Processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_tasks: List[asyncio.Task] = []
        
        # State
        self.running = False
        self.last_performance_update = time.time()
        
        logger.info("Enhanced WebSocket Client initialized")

    async def start(self):
        """Start the WebSocket client"""
        if self.running:
            logger.warning("WebSocket client already running")
            return
        
        self.running = True
        logger.info("Starting Enhanced WebSocket Client")
        
        # Start processing tasks
        self.processing_tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._connection_health_monitor())
        ]
        
        logger.info("WebSocket client started successfully")

    async def stop(self):
        """Stop the WebSocket client"""
        if not self.running:
            return
        
        logger.info("Stopping Enhanced WebSocket Client")
        self.running = False
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Close all connections
        await self._close_all_connections()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("WebSocket client stopped")

    async def subscribe_to_symbols(self, symbols: List[str], stream_types: List[str] = None):
        """Subscribe to market data for specific symbols"""
        if stream_types is None:
            stream_types = ['ticker', 'trade', 'depth@100ms']
        
        try:
            # Group symbols for efficient connection usage
            symbol_groups = self._group_symbols_for_connections(symbols)
            
            for group_id, symbol_group in enumerate(symbol_groups):
                connection_id = f"conn_{group_id}"
                
                # Create streams for this group
                streams = []
                for symbol in symbol_group:
                    symbol_lower = symbol.lower()
                    for stream_type in stream_types:
                        if stream_type == 'ticker':
                            streams.append(f"{symbol_lower}@ticker")
                        elif stream_type == 'trade':
                            streams.append(f"{symbol_lower}@trade")
                        elif stream_type.startswith('depth'):
                            streams.append(f"{symbol_lower}@{stream_type}")
                        elif stream_type == 'kline_1m':
                            streams.append(f"{symbol_lower}@kline_1m")
                
                # Connect and subscribe
                if streams:
                    await self._create_connection(connection_id, streams)
                    
                    # Map symbols to connections
                    for symbol in symbol_group:
                        self.symbol_to_connection[symbol] = connection_id
                        self.subscribed_symbols.add(symbol)
            
            logger.info(f"Successfully subscribed to {len(symbols)} symbols across {len(symbol_groups)} connections")
            
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
            await self._notify_error_callbacks(f"Subscription error: {str(e)}")

    async def unsubscribe_from_symbols(self, symbols: List[str]):
        """Unsubscribe from specific symbols"""
        try:
            for symbol in symbols:
                if symbol in self.symbol_to_connection:
                    connection_id = self.symbol_to_connection[symbol]
                    
                    # Remove from tracking
                    del self.symbol_to_connection[symbol]
                    self.subscribed_symbols.discard(symbol)
                    
                    # Clear data buffer
                    if symbol in self.data_buffer:
                        self.data_buffer[symbol].clear()
            
            # Check if any connections are no longer needed
            active_connections = set(self.symbol_to_connection.values())
            for connection_id in list(self.connections.keys()):
                if connection_id not in active_connections:
                    await self._close_connection(connection_id)
            
            logger.info(f"Unsubscribed from {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from symbols: {e}")

    def _group_symbols_for_connections(self, symbols: List[str]) -> List[List[str]]:
        """Group symbols efficiently across connections to maximize throughput"""
        # Binance allows up to 1024 streams per connection
        # We'll use a more conservative limit for stability
        max_streams_per_connection = 200  # Each symbol can have 3-4 streams
        max_symbols_per_connection = max_streams_per_connection // 4  # Conservative estimate
        
        groups = []
        current_group = []
        
        for symbol in symbols:
            if len(current_group) >= max_symbols_per_connection:
                groups.append(current_group)
                current_group = []
            
            current_group.append(symbol)
        
        if current_group:
            groups.append(current_group)
        
        return groups

    async def _create_connection(self, connection_id: str, streams: List[str]):
        """Create a new WebSocket connection"""
        try:
            # Close existing connection if it exists
            if connection_id in self.connections:
                await self._close_connection(connection_id)
            
            # Create SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Build WebSocket URL
            stream_names = "/".join(streams)
            ws_url = f"{self.config.binance_ws_url}{stream_names}"
            
            logger.info(f"Connecting to {connection_id} with {len(streams)} streams")
            
            # Connect with retry logic
            for attempt in range(self.config.max_reconnect_attempts):
                try:
                    websocket = await websockets.connect(
                        ws_url,
                        ssl=ssl_context,
                        ping_interval=self.config.ping_interval,
                        ping_timeout=self.config.ping_timeout,
                        compression='deflate' if self.config.compression_enabled else None
                    )
                    
                    # Store connection
                    self.connections[connection_id] = websocket
                    self.connection_stats[connection_id] = ConnectionStats(
                        connection_id=connection_id,
                        connected_at=datetime.now(),
                        status="connected"
                    )
                    
                    # Start message handler
                    asyncio.create_task(self._handle_connection_messages(connection_id, websocket))
                    
                    logger.info(f"Successfully connected {connection_id}")
                    await self._notify_connection_callbacks(connection_id, "connected")
                    
                    break
                    
                except Exception as e:
                    logger.warning(f"Connection attempt {attempt + 1} failed for {connection_id}: {e}")
                    if attempt < self.config.max_reconnect_attempts - 1:
                        await asyncio.sleep(self.config.reconnect_delay * (attempt + 1))
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Failed to create connection {connection_id}: {e}")
            await self._notify_error_callbacks(f"Connection failed: {str(e)}")

    async def _handle_connection_messages(self, connection_id: str, websocket):
        """Handle messages from a specific connection"""
        try:
            stats = self.connection_stats[connection_id]
            
            async for message in websocket:
                try:
                    # Record message timing
                    receive_time = time.time()
                    stats.messages_received += 1
                    stats.last_message_time = datetime.now()
                    
                    # Parse message
                    if isinstance(message, bytes):
                        message = gzip.decompress(message).decode('utf-8')
                    
                    data = json.loads(message)
                    
                    # Calculate latency (if timestamp available)
                    if 'E' in data:  # Event time
                        event_time = data['E'] / 1000  # Convert to seconds
                        latency = (receive_time - event_time) * 1000  # Convert to ms
                        stats.latency_ms = latency
                    
                    # Queue message for processing
                    if not self.message_queue.full():
                        await self.message_queue.put((connection_id, data, receive_time))
                        stats.messages_processed += 1
                    else:
                        self.performance_metrics['data_loss_events'] += 1
                        logger.warning(f"Message queue full - dropping message from {connection_id}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from {connection_id}: {e}")
                except Exception as e:
                    logger.error(f"Error processing message from {connection_id}: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Connection {connection_id} closed")
            stats.status = "disconnected"
            
            # Attempt reconnection if still running
            if self.running:
                await self._reconnect_connection(connection_id)
        
        except Exception as e:
            logger.error(f"Connection {connection_id} error: {e}")
            stats.status = "error"
            
            if self.running:
                await self._reconnect_connection(connection_id)

    async def _reconnect_connection(self, connection_id: str):
        """Reconnect a failed connection"""
        try:
            logger.info(f"Attempting to reconnect {connection_id}")
            
            # Get symbols for this connection
            symbols_for_connection = [
                symbol for symbol, conn_id in self.symbol_to_connection.items()
                if conn_id == connection_id
            ]
            
            if symbols_for_connection:
                # Update stats
                if connection_id in self.connection_stats:
                    self.connection_stats[connection_id].reconnections += 1
                
                self.performance_metrics['reconnection_events'] += 1
                
                # Create streams
                streams = []
                for symbol in symbols_for_connection:
                    symbol_lower = symbol.lower()
                    streams.extend([
                        f"{symbol_lower}@ticker",
                        f"{symbol_lower}@trade",
                        f"{symbol_lower}@depth@100ms"
                    ])
                
                # Recreate connection
                await self._create_connection(connection_id, streams)
                
                logger.info(f"Successfully reconnected {connection_id}")
                
        except Exception as e:
            logger.error(f"Failed to reconnect {connection_id}: {e}")
            
            # Schedule retry after delay
            await asyncio.sleep(self.config.reconnect_delay)
            if self.running:
                asyncio.create_task(self._reconnect_connection(connection_id))

    async def _message_processor(self):
        """Process incoming messages"""
        while self.running:
            try:
                # Get message with timeout
                try:
                    connection_id, data, receive_time = await asyncio.wait_for(
                        self.message_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process message
                await self._process_market_data(data, receive_time)
                self.performance_metrics['total_messages'] += 1
                
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(0.1)

    async def _process_market_data(self, data: Dict, receive_time: float):
        """Process and normalize market data"""
        try:
            if 'stream' not in data or 'data' not in data:
                return
            
            stream = data['stream']
            market_data = data['data']
            
            # Extract symbol from stream
            symbol = self._extract_symbol_from_stream(stream)
            if not symbol:
                return
            
            # Process different stream types
            if '@ticker' in stream:
                await self._process_ticker_data(symbol, market_data, receive_time)
            elif '@trade' in stream:
                await self._process_trade_data(symbol, market_data, receive_time)
            elif '@depth' in stream:
                await self._process_depth_data(symbol, market_data, receive_time)
            elif '@kline' in stream:
                await self._process_kline_data(symbol, market_data, receive_time)
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")

    async def _process_ticker_data(self, symbol: str, data: Dict, receive_time: float):
        """Process ticker data"""
        try:
            market_point = MarketDataPoint(
                symbol=symbol,
                price=float(data.get('c', 0)),  # Current price
                volume=float(data.get('v', 0)),  # 24h volume
                timestamp=datetime.fromtimestamp(receive_time),
                bid=float(data.get('b', 0)),
                ask=float(data.get('a', 0)),
                high_24h=float(data.get('h', 0)),
                low_24h=float(data.get('l', 0)),
                change_24h=float(data.get('P', 0)) / 100,  # Percentage change
                volume_24h=float(data.get('q', 0)),  # Quote volume
                trades_count=int(data.get('n', 0))
            )
            
            # Add to buffer
            self.data_buffer[symbol].append(market_point)
            
            # Notify callbacks
            await self._notify_data_callbacks(symbol, market_point, 'ticker')
            
        except Exception as e:
            logger.error(f"Error processing ticker data for {symbol}: {e}")

    async def _process_trade_data(self, symbol: str, data: Dict, receive_time: float):
        """Process trade data"""
        try:
            market_point = MarketDataPoint(
                symbol=symbol,
                price=float(data.get('p', 0)),
                volume=float(data.get('q', 0)),
                timestamp=datetime.fromtimestamp(receive_time)
            )
            
            # Notify callbacks for real-time trade data
            await self._notify_data_callbacks(symbol, market_point, 'trade')
            
        except Exception as e:
            logger.error(f"Error processing trade data for {symbol}: {e}")

    async def _process_depth_data(self, symbol: str, data: Dict, receive_time: float):
        """Process order book depth data"""
        try:
            # Extract best bid/ask from order book
            bids = data.get('b', [])
            asks = data.get('a', [])
            
            bid_price = float(bids[0][0]) if bids else 0.0
            ask_price = float(asks[0][0]) if asks else 0.0
            
            market_point = MarketDataPoint(
                symbol=symbol,
                price=(bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else 0,
                volume=0,  # Not applicable for depth
                timestamp=datetime.fromtimestamp(receive_time),
                bid=bid_price,
                ask=ask_price
            )
            
            # Notify callbacks
            await self._notify_data_callbacks(symbol, market_point, 'depth')
            
        except Exception as e:
            logger.error(f"Error processing depth data for {symbol}: {e}")

    async def _process_kline_data(self, symbol: str, data: Dict, receive_time: float):
        """Process kline/candlestick data"""
        try:
            kline = data.get('k', {})
            
            market_point = MarketDataPoint(
                symbol=symbol,
                price=float(kline.get('c', 0)),  # Close price
                volume=float(kline.get('v', 0)),  # Volume
                timestamp=datetime.fromtimestamp(receive_time),
                high_24h=float(kline.get('h', 0)),
                low_24h=float(kline.get('l', 0))
            )
            
            # Only notify on closed klines
            if kline.get('x', False):  # Kline closed
                await self._notify_data_callbacks(symbol, market_point, 'kline')
            
        except Exception as e:
            logger.error(f"Error processing kline data for {symbol}: {e}")

    def _extract_symbol_from_stream(self, stream: str) -> str:
        """Extract symbol from stream name"""
        try:
            # Stream format: "btcusdt@ticker" -> "BTCUSDT"
            symbol_part = stream.split('@')[0]
            return symbol_part.upper()
        except:
            return ""

    async def _performance_monitor(self):
        """Monitor and update performance metrics"""
        while self.running:
            try:
                current_time = time.time()
                time_diff = current_time - self.last_performance_update
                
                if time_diff >= 10:  # Update every 10 seconds
                    # Calculate messages per second
                    recent_messages = self.performance_metrics['total_messages']
                    self.performance_metrics['messages_per_second'] = recent_messages / time_diff
                    
                    # Calculate average latency
                    latencies = [
                        stats.latency_ms for stats in self.connection_stats.values()
                        if stats.latency_ms > 0
                    ]
                    
                    if latencies:
                        self.performance_metrics['average_latency'] = np.mean(latencies)
                    
                    # Calculate connection uptime
                    if self.connection_stats:
                        uptimes = []
                        for stats in self.connection_stats.values():
                            if stats.status == "connected":
                                uptime = (datetime.now() - stats.connected_at).total_seconds()
                                uptimes.append(uptime)
                        
                        if uptimes:
                            self.performance_metrics['connection_uptime'] = np.mean(uptimes)
                    
                    self.last_performance_update = current_time
                    
                    # Log performance summary
                    if self.performance_metrics['messages_per_second'] > 0:
                        logger.debug(
                            f"WebSocket Performance: "
                            f"{self.performance_metrics['messages_per_second']:.1f} msg/s, "
                            f"{self.performance_metrics['average_latency']:.1f}ms latency, "
                            f"{len(self.connections)} connections"
                        )
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(5)

    async def _connection_health_monitor(self):
        """Monitor connection health and trigger reconnections if needed"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for connection_id, stats in self.connection_stats.items():
                    # Check for stale connections
                    if stats.last_message_time:
                        time_since_last_message = current_time - stats.last_message_time
                        
                        if time_since_last_message > timedelta(minutes=2):
                            logger.warning(f"Stale connection detected: {connection_id}")
                            if connection_id in self.connections:
                                await self._reconnect_connection(connection_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(30)

    async def _close_connection(self, connection_id: str):
        """Close a specific connection"""
        try:
            if connection_id in self.connections:
                websocket = self.connections[connection_id]
                await websocket.close()
                del self.connections[connection_id]
                
                # Update stats
                if connection_id in self.connection_stats:
                    self.connection_stats[connection_id].status = "disconnected"
                
                logger.info(f"Closed connection {connection_id}")
                
        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {e}")

    async def _close_all_connections(self):
        """Close all connections"""
        for connection_id in list(self.connections.keys()):
            await self._close_connection(connection_id)

    # Callback management
    def add_data_callback(self, callback: Callable):
        """Add callback for market data updates"""
        self.data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable):
        """Add callback for error events"""
        self.error_callbacks.append(callback)

    def add_connection_callback(self, callback: Callable):
        """Add callback for connection events"""
        self.connection_callbacks.append(callback)

    async def _notify_data_callbacks(self, symbol: str, data: MarketDataPoint, data_type: str):
        """Notify all data callbacks"""
        for callback in self.data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, data, data_type)
                else:
                    callback(symbol, data, data_type)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")

    async def _notify_error_callbacks(self, error_message: str):
        """Notify all error callbacks"""
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_message)
                else:
                    callback(error_message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def _notify_connection_callbacks(self, connection_id: str, status: str):
        """Notify all connection callbacks"""
        for callback in self.connection_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(connection_id, status)
                else:
                    callback(connection_id, status)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")

    # Data access methods
    def get_latest_data(self, symbol: str, count: int = 1) -> List[MarketDataPoint]:
        """Get latest market data for a symbol"""
        if symbol in self.data_buffer:
            buffer = self.data_buffer[symbol]
            return list(buffer)[-count:] if len(buffer) >= count else list(buffer)
        return []

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        return {
            **self.performance_metrics,
            'active_connections': len(self.connections),
            'subscribed_symbols': len(self.subscribed_symbols),
            'buffer_usage': {
                symbol: len(buffer) for symbol, buffer in self.data_buffer.items()
            },
            'connection_stats': {
                conn_id: {
                    'status': stats.status,
                    'messages_received': stats.messages_received,
                    'reconnections': stats.reconnections,
                    'latency_ms': stats.latency_ms
                }
                for conn_id, stats in self.connection_stats.items()
            }
        }

    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols"""
        return list(self.subscribed_symbols)

# Global instance
enhanced_ws_client = EnhancedWebSocketClient()
