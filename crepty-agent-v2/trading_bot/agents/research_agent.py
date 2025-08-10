import asyncio
from loguru import logger
from trading_bot.utils.binance_client import BinanceClient
from trading_bot.utils.data_models import MarketData
from datetime import datetime

class ResearchAgent:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.binance = BinanceClient()
        logger.info(f"ResearchAgent initialized for {symbol}")

    async def fetch_market_data(self) -> MarketData:
        try:
            ticker = self.binance.get_ticker(self.symbol)
            price = float(ticker.get('price', 0.0))
            if price == 0.0:
                price = float(ticker.get('lastPrice', 0.0))
                if price == 0.0:
                    logger.warning(f"Ticker for {self.symbol} missing both 'price' and 'lastPrice'. Ticker: {ticker}")
            volume = float(ticker.get('volume', 0.0)) if 'volume' in ticker else 0.0
            quote_volume = float(ticker.get('quoteVolume', 0.0)) if 'quoteVolume' in ticker else 0.0
            fee = self.binance.get_trade_fee(self.symbol)
            data = MarketData(
                symbol=self.symbol,
                price=price,
                volume=volume,
                market_cap=quote_volume,
                timestamp=datetime.utcnow(),
                indicators={"fee": fee}
            )
            logger.info(f"Fetched market data: {data}")
            return data
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise

    async def receive_message(self, message):
        # Example: log the received message
        from loguru import logger
        logger.info(f"ResearchAgent received message: {message}")
