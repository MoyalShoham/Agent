from loguru import logger
from trading_bot.utils.openai_client import OpenAIClient
from trading_bot.utils.data_models import MarketData

class AnalysisAgent:
    def __init__(self):
        self.openai = OpenAIClient()
        logger.info("AnalysisAgent initialized.")

    async def analyze(self, market_data: MarketData) -> dict:
        # Smarter prompt: include SMA and more context
        prompt = f"""
        You are a crypto trading expert. Analyze the following data and provide a trading recommendation (buy/sell/hold), entry/exit points, and confidence score.
        Symbol: {market_data.symbol}
        Price: {market_data.price}
        Volume: {market_data.volume}
        Market Cap: {market_data.market_cap}
        Fee: {market_data.indicators.get('fee')}
        SMA-14: {market_data.indicators.get('sma_14')}
        EMA-14: {market_data.indicators.get('ema_14')}
        RSI-14: {market_data.indicators.get('rsi_14')}
        Bollinger Bands: Upper={market_data.indicators.get('bb_upper')}, Mid={market_data.indicators.get('bb_mid')}, Lower={market_data.indicators.get('bb_lower')}
        Timestamp: {market_data.timestamp}
        If SMA-14 is above price, consider it a bearish sign. If price is above SMA-14, consider it bullish. If RSI-14 > 70, consider overbought; if < 30, oversold. If price is near Bollinger Upper, consider overbought; near Lower, oversold. Use all data to justify your answer.
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
