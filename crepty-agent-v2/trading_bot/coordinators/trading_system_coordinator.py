"""
Real-Time Crypto Trading System Coordinator
Integrates data collection, strategy experts, AI agents, and meta-learning
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger

from trading_bot.data.real_time_data_collector import RealTimeDataManager
from trading_bot.strategies.strategy_experts import StrategyManager, create_default_strategy_manager
from trading_bot.ai_models.ai_agents import AIAgentOrchestrator, create_ai_agent_orchestrator
from trading_bot.ai_models.meta_learner import MetaLearner, create_meta_learner, MetaSignal, MetaSignalType
from trading_bot.utils.binance_client import BinanceClient
from trading_bot.risk.advanced_risk_manager import AdvancedRiskManager

class TradingSystemCoordinator:
    """Main coordinator for the complete trading system"""
    
    def __init__(self, symbols: List[str], config: Dict[str, Any] = None):
        self.symbols = symbols
        self.config = config or {}
        self.running = False
        
        # Core components
        self.data_manager: Optional[RealTimeDataManager] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.ai_orchestrator: Optional[AIAgentOrchestrator] = None
        self.meta_learner: Optional[MetaLearner] = None
        self.risk_manager: Optional[AdvancedRiskManager] = None
        self.binance_client: Optional[BinanceClient] = None
        
        # State tracking
        self.current_positions = {}
        self.pending_orders = {}
        self.last_signals = {}
        self.performance_tracker = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Configuration
        self.signal_cooldown = timedelta(minutes=self.config.get('signal_cooldown_minutes', 5))
        self.max_concurrent_positions = self.config.get('max_concurrent_positions', 5)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        self.position_update_interval = self.config.get('position_update_interval', 30)  # seconds
        
    async def initialize(self):
        """Initialize all system components"""
        logger.info("🚀 Initializing Real-Time Crypto Trading System...")
        
        try:
            # Initialize data manager
            self.data_manager = RealTimeDataManager(self.symbols)
            logger.info("✅ Data manager initialized")
            
            # Initialize strategy manager
            self.strategy_manager = create_default_strategy_manager()
            logger.info(f"✅ Strategy manager initialized with {len(self.strategy_manager.strategies)} strategies")
            
            # Initialize AI orchestrator
            self.ai_orchestrator = create_ai_agent_orchestrator()
            logger.info("✅ AI agent orchestrator initialized")
            
            # Initialize meta-learner
            self.meta_learner = create_meta_learner()
            await self.meta_learner.initialize(len(self.strategy_manager.strategies))
            logger.info("✅ Meta-learner initialized")
            
            # Initialize risk manager
            self.risk_manager = AdvancedRiskManager()
            logger.info("✅ Risk manager initialized")
            
            # Initialize Binance client
            self.binance_client = BinanceClient()
            logger.info("✅ Binance client initialized")
            
            logger.info("🎯 All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            raise
            
    async def run(self):
        """Main execution loop"""
        if not all([self.data_manager, self.strategy_manager, self.ai_orchestrator, self.meta_learner]):
            await self.initialize()
            
        self.running = True
        logger.info("🎮 Starting trading system execution loop...")
        
        # Start data collection
        data_task = asyncio.create_task(self.data_manager.start())
        
        # Start main trading loop
        trading_task = asyncio.create_task(self._trading_loop())
        
        # Start position monitoring
        monitoring_task = asyncio.create_task(self._position_monitoring_loop())
        
        # Start performance tracking
        performance_task = asyncio.create_task(self._performance_tracking_loop())
        
        try:
            await asyncio.gather(
                data_task,
                trading_task,
                monitoring_task,
                performance_task,
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"❌ Execution loop error: {e}")
        finally:
            await self.shutdown()
            
    async def _trading_loop(self):
        """Main trading decision loop"""
        logger.info("📊 Starting trading decision loop...")
        
        while self.running:
            try:
                # Process each symbol
                for symbol in self.symbols:
                    await self._process_symbol(symbol)
                    
                # Brief pause between symbols
                await asyncio.sleep(1)
                
                # Longer pause between full cycles
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(30)  # Longer pause on error
                
    async def _process_symbol(self, symbol: str):
        """Process trading signals for a single symbol"""
        try:
            # Check cooldown
            if symbol in self.last_signals:
                time_since_last = datetime.now() - self.last_signals[symbol]['timestamp']
                if time_since_last < self.signal_cooldown:
                    return
                    
            # Get market data
            market_data = await self._collect_market_data(symbol)
            if not market_data:
                return
                
            # Generate strategy signals
            strategy_signals = await self.strategy_manager.generate_all_signals(market_data)
            if not strategy_signals:
                return
                
            # Get AI agent insights
            ai_insights = await self._get_ai_insights(market_data, strategy_signals)
            
            # Generate meta-signal
            meta_signal = await self.meta_learner.generate_meta_signal(
                symbol, market_data, strategy_signals, ai_insights
            )
            
            if not meta_signal:
                return
                
            # Risk management check
            risk_approved = await self._risk_management_check(meta_signal, market_data)
            if not risk_approved:
                logger.info(f"⚠️ Risk management rejected signal for {symbol}")
                return
                
            # Execute trading decision
            await self._execute_trading_decision(meta_signal, market_data)
            
            # Update last signal
            self.last_signals[symbol] = {
                'timestamp': datetime.now(),
                'signal': meta_signal
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            
    async def _collect_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Collect comprehensive market data for a symbol"""
        try:
            # Get real-time data from data manager
            latest_data = self.data_manager.get_latest_data(symbol)
            
            # Get OHLCV data from Binance
            ohlcv_data = await self._get_ohlcv_data(symbol)
            
            # Get funding and open interest data
            funding_data = await self._get_funding_data(symbol)
            oi_data = await self._get_open_interest_data(symbol)
            
            # Compile market data
            market_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'ohlcv': ohlcv_data,
                'latest_trade': latest_data.get('trade'),
                'latest_orderbook': latest_data.get('orderbook'),
                'funding_rate': funding_data.get('funding_rate', 0),
                'funding_history': funding_data.get('history', []),
                'open_interest': oi_data.get('open_interest', 0),
                'oi_history': oi_data.get('history', []),
                'current_price': ohlcv_data['close'].iloc[-1] if len(ohlcv_data) > 0 else 0
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Market data collection error for {symbol}: {e}")
            return None
            
    async def _get_ohlcv_data(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Get OHLCV data from Binance"""
        try:
            # Use 5-minute timeframe for real-time analysis
            klines = await asyncio.to_thread(
                self.binance_client.client.get_klines,
                symbol=symbol,
                interval='5m',
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"OHLCV data error for {symbol}: {e}")
            return pd.DataFrame()
            
    async def _get_funding_data(self, symbol: str) -> Dict[str, Any]:
        """Get funding rate data"""
        try:
            # Current funding rate
            funding_info = await asyncio.to_thread(
                self.binance_client.client.futures_funding_rate,
                symbol=symbol,
                limit=1
            )
            
            current_rate = float(funding_info[0]['fundingRate']) if funding_info else 0
            
            # Historical funding rates
            funding_history = await asyncio.to_thread(
                self.binance_client.client.futures_funding_rate,
                symbol=symbol,
                limit=10
            )
            
            history = [
                {
                    'timestamp': int(item['fundingTime']),
                    'funding_rate': float(item['fundingRate'])
                }
                for item in funding_history
            ]
            
            return {
                'funding_rate': current_rate,
                'history': history
            }
            
        except Exception as e:
            logger.error(f"Funding data error for {symbol}: {e}")
            return {'funding_rate': 0, 'history': []}
            
    async def _get_open_interest_data(self, symbol: str) -> Dict[str, Any]:
        """Get open interest data"""
        try:
            # Current open interest
            oi_info = await asyncio.to_thread(
                self.binance_client.client.futures_open_interest,
                symbol=symbol
            )
            
            current_oi = float(oi_info['openInterest']) if oi_info else 0
            
            # For historical OI, we'd need to store it ourselves
            # For now, just return current value
            return {
                'open_interest': current_oi,
                'history': [{'timestamp': int(datetime.now().timestamp() * 1000), 'open_interest': current_oi}]
            }
            
        except Exception as e:
            logger.error(f"Open interest data error for {symbol}: {e}")
            return {'open_interest': 0, 'history': []}
            
    async def _get_ai_insights(self, market_data: Dict[str, Any], strategy_signals: List) -> Dict[str, Any]:
        """Get insights from AI agents"""
        try:
            # Get market analysis
            market_analysis = await self.ai_orchestrator.get_market_analysis(market_data)
            
            # Get trading decision insights
            signal_data = {
                'signals': [
                    {
                        'strategy': s.strategy_name,
                        'signal': s.signal.name,
                        'confidence': s.confidence,
                        'position_size': s.position_size
                    }
                    for s in strategy_signals
                ]
            }
            
            portfolio_data = {
                'current_positions': self.current_positions,
                'total_equity': await self._get_total_equity(),
                'daily_pnl': self.performance_tracker['daily_pnl']
            }
            
            trading_insights = await self.ai_orchestrator.get_trading_decision(signal_data, portfolio_data)
            
            return {
                'financial_analysis': market_analysis.get('financial_analysis'),
                'sentiment_analysis': market_analysis.get('sentiment_analysis'),
                'broker_decision': trading_insights.get('broker_decision'),
                'risk_assessment': trading_insights.get('risk_assessment')
            }
            
        except Exception as e:
            logger.error(f"AI insights error: {e}")
            return {}
            
    async def _risk_management_check(self, meta_signal: MetaSignal, market_data: Dict[str, Any]) -> bool:
        """Comprehensive risk management check"""
        try:
            # Check confidence threshold
            if meta_signal.confidence < self.min_confidence_threshold:
                return False
                
            # Check maximum concurrent positions
            if len(self.current_positions) >= self.max_concurrent_positions:
                return False
                
            # Check if we already have a position in this symbol
            if meta_signal.symbol in self.current_positions:
                current_pos = self.current_positions[meta_signal.symbol]
                # Only allow position adjustments, not reversals
                if (current_pos['side'] == 'long' and meta_signal.signal_type in [MetaSignalType.SELL, MetaSignalType.STRONG_SELL]) or \
                   (current_pos['side'] == 'short' and meta_signal.signal_type in [MetaSignalType.BUY, MetaSignalType.STRONG_BUY]):
                    return False
                    
            # Risk manager check
            risk_approved = await asyncio.to_thread(
                self.risk_manager.validate_trade,
                meta_signal.symbol,
                meta_signal.position_size,
                meta_signal.entry_price or market_data.get('current_price', 0)
            )
            
            if not risk_approved:
                return False
                
            # Check daily loss limits
            if self.performance_tracker['daily_pnl'] < -0.05:  # -5% daily loss limit
                logger.warning("Daily loss limit reached, blocking new trades")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Risk management check error: {e}")
            return False
            
    async def _execute_trading_decision(self, meta_signal: MetaSignal, market_data: Dict[str, Any]):
        """Execute the trading decision"""
        try:
            symbol = meta_signal.symbol
            signal_type = meta_signal.signal_type
            
            # Determine action
            if signal_type in [MetaSignalType.BUY, MetaSignalType.STRONG_BUY]:
                await self._open_long_position(meta_signal, market_data)
            elif signal_type in [MetaSignalType.SELL, MetaSignalType.STRONG_SELL]:
                await self._open_short_position(meta_signal, market_data)
            else:
                # HOLD signal - check if we should close existing positions
                if symbol in self.current_positions:
                    await self._evaluate_position_close(symbol, meta_signal, market_data)
                    
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            
    async def _open_long_position(self, meta_signal: MetaSignal, market_data: Dict[str, Any]):
        """Open a long position"""
        try:
            symbol = meta_signal.symbol
            entry_price = market_data.get('current_price', meta_signal.entry_price)
            
            # Calculate position size in USDT
            total_equity = await self._get_total_equity()
            position_value = total_equity * meta_signal.position_size
            quantity = position_value / entry_price
            
            # Place market buy order
            order = await asyncio.to_thread(
                self.binance_client.client.futures_create_order,
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=quantity
            )
            
            # Set stop loss and take profit
            if meta_signal.stop_loss:
                stop_order = await asyncio.to_thread(
                    self.binance_client.client.futures_create_order,
                    symbol=symbol,
                    side='SELL',
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=meta_signal.stop_loss
                )
                
            if meta_signal.take_profit:
                tp_order = await asyncio.to_thread(
                    self.binance_client.client.futures_create_order,
                    symbol=symbol,
                    side='SELL',
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=meta_signal.take_profit
                )
                
            # Track position
            self.current_positions[symbol] = {
                'side': 'long',
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'meta_signal': meta_signal,
                'order_id': order['orderId']
            }
            
            logger.info(f"🟢 Opened LONG position: {symbol} | Qty: {quantity:.6f} | Price: ${entry_price:.4f}")
            
        except Exception as e:
            logger.error(f"Long position error: {e}")
            
    async def _open_short_position(self, meta_signal: MetaSignal, market_data: Dict[str, Any]):
        """Open a short position"""
        try:
            symbol = meta_signal.symbol
            entry_price = market_data.get('current_price', meta_signal.entry_price)
            
            # Calculate position size in USDT
            total_equity = await self._get_total_equity()
            position_value = total_equity * meta_signal.position_size
            quantity = position_value / entry_price
            
            # Place market sell order
            order = await asyncio.to_thread(
                self.binance_client.client.futures_create_order,
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            
            # Set stop loss and take profit
            if meta_signal.stop_loss:
                stop_order = await asyncio.to_thread(
                    self.binance_client.client.futures_create_order,
                    symbol=symbol,
                    side='BUY',
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=meta_signal.stop_loss
                )
                
            if meta_signal.take_profit:
                tp_order = await asyncio.to_thread(
                    self.binance_client.client.futures_create_order,
                    symbol=symbol,
                    side='BUY',
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=meta_signal.take_profit
                )
                
            # Track position
            self.current_positions[symbol] = {
                'side': 'short',
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'meta_signal': meta_signal,
                'order_id': order['orderId']
            }
            
            logger.info(f"🔴 Opened SHORT position: {symbol} | Qty: {quantity:.6f} | Price: ${entry_price:.4f}")
            
        except Exception as e:
            logger.error(f"Short position error: {e}")
            
    async def _evaluate_position_close(self, symbol: str, meta_signal: MetaSignal, market_data: Dict[str, Any]):
        """Evaluate whether to close an existing position"""
        if symbol not in self.current_positions:
            return
            
        position = self.current_positions[symbol]
        current_price = market_data.get('current_price', 0)
        
        # Calculate current PnL
        if position['side'] == 'long':
            pnl = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl = (position['entry_price'] - current_price) / position['entry_price']
            
        # Close conditions
        should_close = False
        close_reason = ""
        
        # Time-based close (position held too long)
        position_age = datetime.now() - position['entry_time']
        if position_age > timedelta(hours=24):
            should_close = True
            close_reason = "time_limit"
            
        # Profit taking
        if pnl > 0.05:  # 5% profit
            should_close = True
            close_reason = "profit_target"
            
        # Stop loss
        if pnl < -0.03:  # 3% loss
            should_close = True
            close_reason = "stop_loss"
            
        # Low confidence signal suggests exit
        if meta_signal.confidence < 0.3:
            should_close = True
            close_reason = "low_confidence"
            
        if should_close:
            await self._close_position(symbol, close_reason)
            
    async def _close_position(self, symbol: str, reason: str = "manual"):
        """Close an existing position"""
        try:
            if symbol not in self.current_positions:
                return
                
            position = self.current_positions[symbol]
            
            # Determine close side
            close_side = 'SELL' if position['side'] == 'long' else 'BUY'
            
            # Close position
            order = await asyncio.to_thread(
                self.binance_client.client.futures_create_order,
                symbol=symbol,
                side=close_side,
                type='MARKET',
                quantity=position['quantity']
            )
            
            # Calculate final PnL
            current_price = float(order['price']) if 'price' in order else position['entry_price']
            if position['side'] == 'long':
                pnl = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl = (position['entry_price'] - current_price) / position['entry_price']
                
            # Update performance
            self.performance_tracker['total_trades'] += 1
            if pnl > 0:
                self.performance_tracker['winning_trades'] += 1
            self.performance_tracker['total_pnl'] += pnl
            self.performance_tracker['daily_pnl'] += pnl
            
            # Update meta-learner
            trade_duration = (datetime.now() - position['entry_time']).total_seconds() / 3600  # hours
            await self.meta_learner.update_performance(position['meta_signal'], pnl, trade_duration)
            
            # Remove from positions
            del self.current_positions[symbol]
            
            logger.info(f"❌ Closed {position['side'].upper()} position: {symbol} | "
                       f"PnL: {pnl:.3%} | Reason: {reason}")
            
        except Exception as e:
            logger.error(f"Position close error: {e}")
            
    async def _position_monitoring_loop(self):
        """Monitor existing positions"""
        logger.info("👁️ Starting position monitoring loop...")
        
        while self.running:
            try:
                for symbol in list(self.current_positions.keys()):
                    # Get current market data
                    market_data = await self._collect_market_data(symbol)
                    if market_data:
                        # Check if position should be closed
                        await self._evaluate_position_close(symbol, 
                                                           self.current_positions[symbol]['meta_signal'], 
                                                           market_data)
                        
                await asyncio.sleep(self.position_update_interval)
                
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _performance_tracking_loop(self):
        """Track and log performance metrics"""
        logger.info("📈 Starting performance tracking loop...")
        
        while self.running:
            try:
                # Reset daily PnL at midnight
                now = datetime.now()
                if now.hour == 0 and now.minute == 0:
                    self.performance_tracker['daily_pnl'] = 0.0
                    
                # Log performance every hour
                if now.minute == 0:
                    await self._log_performance()
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(300)
                
    async def _log_performance(self):
        """Log current performance metrics"""
        try:
            total_equity = await self._get_total_equity()
            win_rate = (self.performance_tracker['winning_trades'] / 
                       max(self.performance_tracker['total_trades'], 1))
            
            # Get meta-learner performance
            meta_performance = self.meta_learner.get_performance_metrics()
            
            logger.info(f"📊 Performance Update:")
            logger.info(f"  Total Equity: ${total_equity:.2f}")
            logger.info(f"  Active Positions: {len(self.current_positions)}")
            logger.info(f"  Total Trades: {self.performance_tracker['total_trades']}")
            logger.info(f"  Win Rate: {win_rate:.2%}")
            logger.info(f"  Total PnL: {self.performance_tracker['total_pnl']:.3%}")
            logger.info(f"  Daily PnL: {self.performance_tracker['daily_pnl']:.3%}")
            logger.info(f"  Meta-learner Win Rate: {meta_performance.get('win_rate', 0):.2%}")
            
        except Exception as e:
            logger.error(f"Performance logging error: {e}")
            
    async def _get_total_equity(self) -> float:
        """Get total account equity"""
        try:
            account_info = await asyncio.to_thread(
                self.binance_client.client.futures_account
            )
            return float(account_info['totalWalletBalance'])
        except Exception as e:
            logger.error(f"Equity retrieval error: {e}")
            return 10000.0  # Default fallback
            
    async def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("🛑 Shutting down trading system...")
        
        self.running = False
        
        # Close all positions
        for symbol in list(self.current_positions.keys()):
            await self._close_position(symbol, "shutdown")
            
        # Stop data collection
        if self.data_manager:
            self.data_manager.stop()
            
        logger.info("✅ Trading system shutdown complete")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'running': self.running,
            'symbols': self.symbols,
            'active_positions': len(self.current_positions),
            'positions': self.current_positions,
            'performance': self.performance_tracker,
            'strategy_performance': self.strategy_manager.get_strategy_performance() if self.strategy_manager else {},
            'ai_agent_status': self.ai_orchestrator.get_agent_status() if self.ai_orchestrator else {},
            'meta_learner_performance': self.meta_learner.get_performance_metrics() if self.meta_learner else {},
            'last_update': datetime.now()
        }

# Factory function
def create_trading_system(symbols: List[str], config: Dict[str, Any] = None) -> TradingSystemCoordinator:
    """Create a new trading system instance"""
    return TradingSystemCoordinator(symbols, config)
