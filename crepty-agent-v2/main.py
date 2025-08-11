import os
import logging
from dotenv import load_dotenv
from loguru import logger
from trading_bot.agents.manager_agent import ManagerAgent
from trading_bot.config.settings import settings

load_dotenv()
logging.basicConfig(level=getattr(settings, 'LOG_LEVEL', 'INFO'))

def main():
    logger.info("Starting Real-Time Crypto Trading Agent System...")
    paper_trading = getattr(settings, 'PAPER_TRADING', True)
    mode = os.getenv('MODE', 'live' if not paper_trading else 'sim')
    if settings.FUTURES_ENABLED:
        try:
            from trading_bot.utils.binance_client import BinanceClient
            from trading_bot.utils import order_execution
            spot_client = BinanceClient()
            equity = spot_client.get_total_usdt_value()
            order_execution.initialize(external_equity=equity)
            logger.info(f"Futures mode enabled. Initialized execution with equity={equity:.2f} USDT")
        except Exception as e:
            logger.exception(f"Futures initialization failed: {e}")
    manager = ManagerAgent()
    # Optionally, allow for backtest mode using the new backtest_portfolio_allocation
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
        logger.info('Running full portfolio backtest...')
        result_file = binance_client.backtest_portfolio_allocation(
            asset_allocation=allocation,
            interval='1h',
            start_balance=10000,
            limit=200,
            filename='portfolio_backtest_results.csv'
        )
        logger.info(f'Backtest complete. Results saved to {result_file}')
    else:
        try:
            import asyncio
            asyncio.run(manager.run())
        except KeyboardInterrupt:
            logger.info("Shutting down system.")
        except Exception as e:
            logger.exception(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
