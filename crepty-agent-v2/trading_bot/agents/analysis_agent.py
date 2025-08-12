from loguru import logger
from trading_bot.utils.openai_client import OpenAIClient
from trading_bot.utils.data_models import MarketData
from typing import Dict, Any
import json

class AnalysisAgent:
    def __init__(self):
        self.openai = OpenAIClient()
        logger.info("AnalysisAgent initialized.")

    def _build_prompt(self, market_data: MarketData, context: Dict[str, Any]) -> str:
        indicators = market_data.indicators or {}
        position_ctx = context.get('position', {})
        risk_ctx = context.get('risk', {})
        micro_ctx = context.get('microstructure', {})
        prompt = f"""
You are an expert Binance futures broker and quantitative trading analyst. Return ONLY a JSON object (no markdown) following the specified schema.

Market Snapshot:
Symbol: {market_data.symbol}
Price: {market_data.price}
Volume: {market_data.volume}
MarketCapProxy: {market_data.market_cap}
TimestampUTC: {market_data.timestamp}
Indicators:
  fee: {indicators.get('fee')}
  sma_14: {indicators.get('sma_14')}
  ema_14: {indicators.get('ema_14')}
  rsi_14: {indicators.get('rsi_14')}
  bb_upper: {indicators.get('bb_upper')}
  bb_mid: {indicators.get('bb_mid')}
  bb_lower: {indicators.get('bb_lower')}
Microstructure (may be empty): {json.dumps(micro_ctx)}
Current Position: {json.dumps(position_ctx)}
Risk Limits: {json.dumps(risk_ctx)}

Interpretation rules:
- If price > sma_14 and RSI between 45-65 -> mild bullish continuation.
- RSI > 70 = overbought; < 30 = oversold.
- Price near bb_upper -> potential mean reversion; near bb_lower -> potential bounce.
- Avoid initiating if daily loss limit reached or risk flag true.

Output JSON Schema (keys & types):
{{
  "action": "buy|sell|hold",
  "entry": float or null,
  "stop": float or null,
  "targets": [float, ...],
  "size_pct": float,  # percent of allowed max position size (0-100)
  "confidence": float,  # 0-1
  "reasoning": str,
  "risk_tags": [str]
}}
Constraints:
- action must be hold if risk_ctx.daily_loss_exceeded true.
- size_pct <= 100.
- Provide 1-3 targets if action != hold.
Return ONLY JSON.
"""
        return prompt

    def _validate(self, data: Dict[str, Any]) -> bool:
        try:
            if data.get('action') not in ['buy','sell','hold']:
                return False
            if not 0 <= float(data.get('confidence', 0)) <= 1:
                return False
            if not 0 <= float(data.get('size_pct', 0)) <= 100:
                return False
            if 'targets' in data and not isinstance(data['targets'], list):
                return False
            return True
        except Exception:
            return False

    async def analyze(self, market_data: MarketData, context: Dict[str, Any] | None = None) -> dict:
        context = context or {}
        prompt = self._build_prompt(market_data, context)
        try:
            result = self.openai.ask_json(prompt, schema_validator=self._validate)
            logger.info(f"AnalysisAgent structured response: {result}")
            return result
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            return {"error": str(e)}

    async def receive_message(self, message):
        # Future: handle inter-agent routing
        logger.debug(f"AnalysisAgent received message: {message}")
        return None
