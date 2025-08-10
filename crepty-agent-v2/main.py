import asyncio
from loguru import logger
from trading_bot.agents.manager_agent import ManagerAgent

if __name__ == "__main__":
    logger.info("Starting Real-Time Crypto Trading Agent System...")
    manager = ManagerAgent()
    try:
        asyncio.run(manager.run())
    except KeyboardInterrupt:
        logger.info("Shutting down system.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
