from loguru import logger
from trading_bot.utils.openai_client import OpenAIClient
from trading_bot.utils.data_models import MarketData

class AnalysisAgent:
    def __init__(self):
        self.openai = OpenAIClient()
        logger.info("AnalysisAgent initialized.")

    async def analyze(self, market_data: MarketData) -> dict:
        prompt = f"""
        Given the following market data:
        Symbol: {market_data.symbol}
        Price: {market_data.price}
        Volume: {market_data.volume}
        Market Cap: {market_data.market_cap}
        Indicators: {market_data.indicators}
        Provide a trading recommendation (buy/sell/hold), entry/exit points, and confidence score.
        """
        try:
            response = self.openai.ask(prompt)
            logger.info(f"OpenAI analysis response: {response}")
            return {"recommendation": response}
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            return {"error": str(e)}

    async def receive_message(self, message):
        # Handle incoming messages from ManagerAgent
        pass
