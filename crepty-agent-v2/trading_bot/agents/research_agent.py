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
            # Add SMA indicator
            closes = self.binance.get_historical_prices(self.symbol, interval='1h', limit=30)
            sma_14 = self.binance.simple_moving_average(closes, window=14) if closes else None
            ema_14 = self.binance.exponential_moving_average(closes, window=14) if closes else None
            rsi_14 = self.binance.relative_strength_index(closes, window=14) if closes else None
            bb_upper, bb_mid, bb_lower = self.binance.bollinger_bands(closes, window=20) if closes else (None, None, None)
            indicators = {
                "fee": fee,
                "sma_14": sma_14,
                "ema_14": ema_14,
                "rsi_14": rsi_14,
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower
            }
            data = MarketData(
                symbol=self.symbol,
                price=price,
                volume=volume,
                market_cap=quote_volume,
                timestamp=datetime.utcnow(),
                indicators=indicators
            )
            logger.info(f"Fetched market data: {data}")
            return data
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise


    async def research(self, market_data: MarketData) -> int:
        """
        Placeholder research method. Returns 1 (buy), 0 (hold), or -1 (sell).
        Extend with real research logic as needed.
        """
        logger.info(f"[RESEARCH] ResearchAgent analyzing market data for {market_data.symbol}")
        # Example: always return 1 (buy)
        return 1

    async def receive_message(self, message):
        # Example: log the received message
        from loguru import logger
        logger.info(f"ResearchAgent received message: {message}")
