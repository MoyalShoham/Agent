
import os
import logging
from dotenv import load_dotenv
from loguru import logger
from trading_bot.agents.manager_agent import ManagerAgent
from trading_bot.config.settings import settings
from trading_bot.risk.advanced_risk_manager import AdvancedRiskManager

# --- Start auto symbol agent in background ---
try:
    from auto_symbol_agent import start_agent as start_symbol_agent
    start_symbol_agent()
    logger.info("✅ Auto symbol agent started (background)")
except Exception as e:
    logger.error(f"❌ Failed to start auto symbol agent: {e}")

# Replace duplicate instantiations with singleton imports
# (These modules already create global instances at import time)
try:
    from trading_bot.utils.ai_integration import ai_integrator  # singleton
    from trading_bot.risk.emergency_controls import emergency_controls  # singleton
    from trading_bot.utils.enhanced_websocket import enhanced_ws_client  # singleton
    from trading_bot.utils.ml_signals import _ml_generator as ml_signal_generator  # global instance
except Exception as _imp_err:
    logger.error(f"Singleton import error: {_imp_err}")

load_dotenv()
logging.basicConfig(level=getattr(settings, 'LOG_LEVEL', 'INFO'))

def main():
    logger.info("🚀 Starting Enhanced Real-Time Crypto Trading Agent System...")
    logger.info("🔧 Initializing components (singleton-aware)...")

    # Initialize only what lacks a singleton
    risk_manager = None
    try:
        risk_manager = AdvancedRiskManager()
        logger.info("✅ Advanced Risk Manager initialized")
    except Exception as e:
        logger.error(f"❌ Advanced Risk Manager init failed: {e}")

    # Log readiness of existing singletons (already constructed on import)
    try:
        if 'ai_integrator' in globals():
            logger.info("✅ AI Model Integrator ready (singleton)")
        if 'emergency_controls' in globals():
            logger.info("✅ Emergency Risk Controls ready (singleton)")
        if 'enhanced_ws_client' in globals():
            logger.info("✅ Enhanced WebSocket Client ready (singleton)")
        if 'ml_signal_generator' in globals():
            logger.info("✅ Enhanced ML Signal Generator ready (singleton)")
        logger.info("🎯 All core components available")
    except Exception as e:
        logger.error(f"❌ Component readiness logging failed: {e}")

    paper_trading = getattr(settings, 'PAPER_TRADING', True)
    mode = os.getenv('MODE', 'live' if not paper_trading else 'sim')

    if settings.FUTURES_ENABLED:
        try:
            from trading_bot.utils.binance_client import BinanceClient
            from trading_bot.utils import order_execution
            spot_client = BinanceClient()
            equity = spot_client.get_total_usdt_value()
            order_execution.initialize(external_equity=equity)
            logger.info(f"💰 Futures mode enabled. Initialized execution with equity={equity:.2f} USDT")
        except Exception as e:
            logger.exception(f"❌ Futures initialization failed: {e}")

    manager = ManagerAgent()

    if mode == 'backtest':
        from trading_bot.utils.binance_client import BinanceClient
        allocation = getattr(settings, 'PORTFOLIO_ALLOCATION', {
            'BTCUSDT': 0.2,
            'ETHUSDT': 0.2,
            'BNBUSDT': 0.1,
            'SOLUSDT': 0.1,
            'ADAUSDT': 0.1,
            'USDCUSDT': 0.15,
            'DAIUSDT': 0.15
        })
        binance_client = BinanceClient(paper_trading=True)
        logger.info('📊 Running full portfolio backtest...')
        result_file = binance_client.backtest_portfolio_allocation(
            asset_allocation=allocation,
            interval='1h',
            start_balance=10000,
            limit=200,
            filename='portfolio_backtest_results.csv'
        )
        logger.info(f'✅ Backtest complete. Results saved to {result_file}')
    else:
        try:
            import asyncio
            logger.info("🎮 Starting trading system...")
            asyncio.run(manager.run())
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down system.")
        except Exception as e:
            logger.exception(f"💥 Fatal error: {e}")

if __name__ == "__main__":
    main()
