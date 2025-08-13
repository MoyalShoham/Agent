"""
Real-Time Crypto Analysis Integration
Main integration point for the comprehensive crypto trading system
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

# Import all our new components
from trading_bot.coordinators.trading_system_coordinator import create_trading_system
from trading_bot.data.real_time_data_collector import RealTimeDataManager
from trading_bot.strategies.strategy_experts import create_default_strategy_manager
from trading_bot.ai_models.ai_agents import create_ai_agent_orchestrator
from trading_bot.ai_models.meta_learner import create_meta_learner

class RealTimeCryptoAnalysisSystem:
    """
    Main interface for the Real-Time Crypto Analysis System
    Integrates with your existing trading bot architecture
    """
    
    def __init__(self, symbols: List[str] = None, config: Dict[str, Any] = None):
        # Default symbols from your .env if not provided
        self.symbols = symbols or os.getenv('FUTURES_SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT').split(',')
        self.config = config or self._load_default_config()
        
        # Main coordinator
        self.coordinator = None
        
        # Individual components for direct access
        self.data_manager = None
        self.strategy_manager = None
        self.ai_orchestrator = None
        self.meta_learner = None
        
        self.running = False
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            'signal_cooldown_minutes': int(os.getenv('MIN_TRADE_INTERVAL_MINUTES', 3)),
            'max_concurrent_positions': int(os.getenv('MAX_CONCURRENT_POSITIONS', 7)),
            'min_confidence_threshold': float(os.getenv('AI_CONFIDENCE_THRESHOLD', 0.45)),
            'position_update_interval': 30,
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 0.004)),
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', 0.12)),
            'stop_loss_percentage': float(os.getenv('STOP_LOSS_PERCENTAGE', 2.0)),
            'take_profit_percentage': float(os.getenv('TAKE_PROFIT_PERCENTAGE', 4.0)),
            'max_daily_losses': float(os.getenv('MAX_DAILY_LOSSES', 0.05))
        }
        
    async def initialize(self):
        """Initialize the complete system"""
        logger.info("🚀 Initializing Real-Time Crypto Analysis System...")
        
        try:
            # Create main coordinator
            self.coordinator = create_trading_system(self.symbols, self.config)
            await self.coordinator.initialize()
            
            # Store references to individual components
            self.data_manager = self.coordinator.data_manager
            self.strategy_manager = self.coordinator.strategy_manager
            self.ai_orchestrator = self.coordinator.ai_orchestrator
            self.meta_learner = self.coordinator.meta_learner
            
            logger.info("✅ Real-Time Crypto Analysis System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
            
    async def start(self):
        """Start the trading system"""
        if not self.coordinator:
            await self.initialize()
            
        self.running = True
        logger.info("🎮 Starting Real-Time Crypto Analysis System...")
        
        try:
            await self.coordinator.run()
        except Exception as e:
            logger.error(f"❌ System execution error: {e}")
            raise
        finally:
            await self.stop()
            
    async def stop(self):
        """Stop the trading system gracefully"""
        self.running = False
        if self.coordinator:
            await self.coordinator.shutdown()
        logger.info("🛑 Real-Time Crypto Analysis System stopped")
        
    async def get_market_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market analysis for a symbol"""
        if not self.coordinator:
            await self.initialize()
            
        try:
            # Collect market data
            market_data = await self.coordinator._collect_market_data(symbol)
            if not market_data:
                return None
                
            # Get strategy signals
            strategy_signals = await self.strategy_manager.generate_all_signals(market_data)
            
            # Get AI insights
            ai_insights = await self.coordinator._get_ai_insights(market_data, strategy_signals)
            
            # Generate meta-signal
            meta_signal = await self.meta_learner.generate_meta_signal(
                symbol, market_data, strategy_signals, ai_insights
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'market_data': market_data,
                'strategy_signals': [
                    {
                        'strategy': s.strategy_name,
                        'signal': s.signal.name,
                        'confidence': s.confidence,
                        'position_size': s.position_size,
                        'entry_price': s.entry_price,
                        'stop_loss': s.stop_loss,
                        'take_profit': s.take_profit
                    }
                    for s in strategy_signals
                ],
                'ai_insights': ai_insights,
                'meta_signal': {
                    'signal_type': meta_signal.signal_type.name if meta_signal else 'HOLD',
                    'confidence': meta_signal.confidence if meta_signal else 0,
                    'position_size': meta_signal.position_size if meta_signal else 0,
                    'risk_score': meta_signal.risk_score if meta_signal else 0.5,
                    'expected_return': meta_signal.expected_return if meta_signal else 0,
                    'entry_price': meta_signal.entry_price if meta_signal else 0,
                    'stop_loss': meta_signal.stop_loss if meta_signal else 0,
                    'take_profit': meta_signal.take_profit if meta_signal else 0
                } if meta_signal else None
            }
            
        except Exception as e:
            logger.error(f"Market analysis error for {symbol}: {e}")
            return None
            
    async def get_portfolio_analysis(self) -> Dict[str, Any]:
        """Get comprehensive portfolio analysis"""
        if not self.coordinator:
            await self.initialize()
            
        try:
            portfolio_data = {
                'current_positions': self.coordinator.current_positions,
                'total_equity': await self.coordinator._get_total_equity(),
                'performance': self.coordinator.performance_tracker
            }
            
            # Get portfolio optimization from AI
            optimization = await self.ai_orchestrator.get_portfolio_optimization(portfolio_data)
            
            return {
                'timestamp': datetime.now(),
                'portfolio_data': portfolio_data,
                'optimization_recommendations': optimization,
                'system_status': self.get_system_status()
            }
            
        except Exception as e:
            logger.error(f"Portfolio analysis error: {e}")
            return {}
            
    async def emergency_risk_assessment(self) -> Dict[str, Any]:
        """Perform emergency risk assessment"""
        if not self.coordinator:
            await self.initialize()
            
        try:
            portfolio_data = {
                'current_positions': self.coordinator.current_positions,
                'total_equity': await self.coordinator._get_total_equity(),
                'daily_pnl': self.coordinator.performance_tracker['daily_pnl']
            }
            
            emergency_assessment = await self.ai_orchestrator.emergency_risk_check(portfolio_data)
            
            return emergency_assessment
            
        except Exception as e:
            logger.error(f"Emergency risk assessment error: {e}")
            return {}
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        if not self.coordinator:
            return {'status': 'not_initialized'}
            
        return self.coordinator.get_system_status()
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.coordinator:
            return {}
            
        metrics = {
            'coordinator_performance': self.coordinator.performance_tracker,
            'strategy_performance': self.strategy_manager.get_strategy_performance() if self.strategy_manager else {},
            'meta_learner_performance': self.meta_learner.get_performance_metrics() if self.meta_learner else {},
            'ai_agent_status': self.ai_orchestrator.get_agent_status() if self.ai_orchestrator else {}
        }
        
        return metrics
        
    async def manual_trade_signal(self, symbol: str, action: str, confidence: float = 0.8) -> Dict[str, Any]:
        """Manually trigger a trade signal for testing"""
        if not self.coordinator:
            await self.initialize()
            
        try:
            # Get current market data
            market_data = await self.coordinator._collect_market_data(symbol)
            
            # Create a manual meta-signal
            from trading_bot.ai_models.meta_learner import MetaSignal, MetaSignalType
            
            signal_type_map = {
                'buy': MetaSignalType.BUY,
                'sell': MetaSignalType.SELL,
                'strong_buy': MetaSignalType.STRONG_BUY,
                'strong_sell': MetaSignalType.STRONG_SELL,
                'hold': MetaSignalType.HOLD
            }
            
            if action.lower() not in signal_type_map:
                raise ValueError(f"Invalid action: {action}. Must be one of {list(signal_type_map.keys())}")
                
            manual_signal = MetaSignal(
                symbol=symbol,
                signal_type=signal_type_map[action.lower()],
                confidence=confidence,
                position_size=0.05,  # Small test position
                entry_price=market_data.get('current_price'),
                stop_loss=None,
                take_profit=None,
                contributing_strategies=['manual'],
                ai_agent_insights={},
                risk_score=0.5,
                expected_return=0.02,
                expected_volatility=0.02,
                timestamp=datetime.now(),
                metadata={'manual': True}
            )
            
            # Execute the signal
            await self.coordinator._execute_trading_decision(manual_signal, market_data)
            
            return {
                'status': 'executed',
                'signal': manual_signal,
                'market_data': market_data
            }
            
        except Exception as e:
            logger.error(f"Manual trade signal error: {e}")
            return {'status': 'failed', 'error': str(e)}

# Integration functions for your existing system

async def get_ai_enhanced_signals(symbols: List[str]) -> Dict[str, Any]:
    """
    Get AI-enhanced trading signals for the provided symbols
    This can be called from your existing trading bot
    """
    system = RealTimeCryptoAnalysisSystem(symbols)
    
    try:
        await system.initialize()
        
        results = {}
        for symbol in symbols:
            analysis = await system.get_market_analysis(symbol)
            if analysis:
                results[symbol] = analysis
                
        return results
        
    except Exception as e:
        logger.error(f"AI enhanced signals error: {e}")
        return {}
    finally:
        await system.stop()

async def get_portfolio_optimization_recommendations() -> Dict[str, Any]:
    """
    Get portfolio optimization recommendations
    """
    # Get symbols from environment
    symbols = os.getenv('FUTURES_SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT').split(',')
    system = RealTimeCryptoAnalysisSystem(symbols)
    
    try:
        await system.initialize()
        portfolio_analysis = await system.get_portfolio_analysis()
        return portfolio_analysis
        
    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        return {}
    finally:
        await system.stop()

async def emergency_risk_check() -> Dict[str, Any]:
    """
    Perform emergency risk assessment
    """
    symbols = os.getenv('FUTURES_SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT').split(',')
    system = RealTimeCryptoAnalysisSystem(symbols)
    
    try:
        await system.initialize()
        risk_assessment = await system.emergency_risk_assessment()
        return risk_assessment
        
    except Exception as e:
        logger.error(f"Emergency risk check error: {e}")
        return {}
    finally:
        await system.stop()

# Factory function for easy integration
def create_real_time_analysis_system(symbols: List[str] = None, config: Dict[str, Any] = None) -> RealTimeCryptoAnalysisSystem:
    """
    Create a new Real-Time Crypto Analysis System
    
    Args:
        symbols: List of trading symbols (default: from FUTURES_SYMBOLS env var)
        config: Configuration dictionary (default: from environment variables)
        
    Returns:
        RealTimeCryptoAnalysisSystem instance
    """
    return RealTimeCryptoAnalysisSystem(symbols, config)

# Main execution for standalone use
async def main():
    """Main execution function for standalone use"""
    # Get symbols from environment
    symbols = os.getenv('FUTURES_SYMBOLS', 'BTCUSDT,ETHUSDT,BNBUSDT').split(',')
    
    # Create and start system
    system = create_real_time_analysis_system(symbols)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await system.stop()

if __name__ == "__main__":
    asyncio.run(main())
