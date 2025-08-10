import asyncio
import logging
from loguru import logger
from trading_bot.config.settings import settings
from trading_bot.utils.data_models import AgentMessage

class ManagerAgent:
    def __init__(self):
        self.agents = {}
        self.loop = asyncio.get_event_loop()
        logger.info("ManagerAgent initialized.")

    def register_agent(self, name, agent):
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")

    async def send_message(self, message: AgentMessage):
        recipient = self.agents.get(message.recipient)
        if recipient:
            await recipient.receive_message(message)
        else:
            logger.error(f"Recipient agent {message.recipient} not found.")

    async def run(self):
        logger.info("ManagerAgent running main event loop.")
        # Example: Register and interact with ResearchAgent
        from trading_bot.agents.research_agent import ResearchAgent
        from trading_bot.agents.analysis_agent import AnalysisAgent
        from trading_bot.utils.binance_client import BinanceClient
        import datetime
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]
        analysis_agent = AnalysisAgent()
        binance = BinanceClient()
        self.register_agent("analysis", analysis_agent)

        while True:
            logger.info(f"ManagerAgent heartbeat: {datetime.datetime.utcnow().isoformat()}")
            for symbol in symbols:
                research_agent = ResearchAgent(symbol)
                self.register_agent(f"research_{symbol}", research_agent)
                try:
                    market_data = await research_agent.fetch_market_data()
                    logger.info(f"Fetched market data: {market_data}")
                    recommendation = await analysis_agent.analyze(market_data)
                    logger.info(f"AnalysisAgent recommendation: {recommendation}")
                    rec_text = recommendation.get("recommendation", "").lower()
                    fee = market_data.indicators.get("fee", 0.001)
                    if "buy" in rec_text:
                        usdt_balance = binance.get_balance("USDT")
                        price = market_data.price
                        if price == 0:
                            logger.warning(f"Skipping {symbol} buy: price is zero.")
                            continue
                        min_qty, step_size = binance.get_lot_size(symbol)
                        qty = (usdt_balance * 0.4) / price if price > 0 else 0
                        # Adjust qty to step size
                        if step_size > 0:
                            qty = qty - (qty % step_size)
                        qty = round(qty, 6)
                        # If not enough USDT, auto-convert largest non-USDT asset to USDT
                        min_usdt_needed = price * min_qty
                        if usdt_balance < min_usdt_needed:
                            try:
                                assets = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE"]
                                asset_balances = {a: binance.get_balance(a) for a in assets}
                                asset_balances = {a: b for a, b in asset_balances.items() if a != "USDT" and b > 0}
                                if asset_balances:
                                    largest_asset = max(asset_balances, key=asset_balances.get)
                                    largest_qty = asset_balances[largest_asset]
                                    sell_pair = f"{largest_asset}USDT"
                                    min_sell_qty, sell_step = binance.get_lot_size(sell_pair)
                                    sell_qty = largest_qty * 0.4
                                    if sell_step > 0:
                                        sell_qty = sell_qty - (sell_qty % sell_step)
                                    sell_qty = round(sell_qty, 6)
                                    asset_price = binance.get_price(sell_pair)
                                    if asset_price > 0 and sell_qty >= min_sell_qty:
                                        try:
                                            sell_order = binance.create_order(sell_pair, "SELL", sell_qty)
                                            logger.info(f"Auto-converted {sell_qty} {largest_asset} to USDT: {sell_order}")
                                            binance.log_trade("CONVERT", sell_pair, sell_qty, asset_price, 0, "SUCCESS", str(sell_order))
                                            usdt_balance += sell_qty * asset_price
                                            qty = (usdt_balance * 0.4) / price
                                            if step_size > 0:
                                                qty = qty - (qty % step_size)
                                            qty = round(qty, 6)
                                        except Exception as ce:
                                            logger.error(f"Auto-convert failed: {ce}")
                                            binance.log_trade("CONVERT", sell_pair, sell_qty, asset_price, 0, "FAILED", str(ce))
                                else:
                                    logger.warning("No non-USDT assets to convert for buy.")
                            except Exception as ce:
                                logger.error(f"Error during auto-convert: {ce}")
                        est_fee = qty * price * fee
                        logger.info(f"Estimated BUY fee for {symbol}: {est_fee} USDT")
                        if qty >= min_qty and est_fee < (qty * price * 0.01):
                            try:
                                order = binance.create_order(symbol, "BUY", qty)
                                logger.info(f"BUY order placed: {order}")
                                binance.log_trade("BUY", symbol, qty, price, est_fee, "SUCCESS", str(order))
                            except Exception as oe:
                                logger.error(f"BUY order failed: {oe}")
                                binance.log_trade("BUY", symbol, qty, price, est_fee, "FAILED", str(oe))
                        else:
                            logger.info(f"BUY skipped for {symbol} due to high fee, zero qty, or below min lot size.")
                            binance.log_trade("BUY", symbol, qty, price, est_fee, "SKIPPED", "High fee, zero qty, or below min lot size")
                    elif "sell" in rec_text:
                        base_asset = symbol.replace("USDT", "")
                        asset_balance = binance.get_balance(base_asset)
                        if market_data.price == 0:
                            logger.warning(f"Skipping {symbol} sell: price is zero.")
                            continue
                        min_qty, step_size = binance.get_lot_size(symbol)
                        qty = asset_balance * 0.4
                        if step_size > 0:
                            qty = qty - (qty % step_size)
                        qty = round(qty, 6)
                        est_fee = qty * market_data.price * fee
                        logger.info(f"Estimated SELL fee for {symbol}: {est_fee} USDT")
                        if qty >= min_qty and est_fee < (qty * market_data.price * 0.01):
                            try:
                                order = binance.create_order(symbol, "SELL", qty)
                                logger.info(f"SELL order placed: {order}")
                                binance.log_trade("SELL", symbol, qty, market_data.price, est_fee, "SUCCESS", str(order))
                            except Exception as oe:
                                logger.error(f"SELL order failed: {oe}")
                                binance.log_trade("SELL", symbol, qty, market_data.price, est_fee, "FAILED", str(oe))
                        else:
                            logger.info(f"SELL skipped for {symbol} due to high fee, zero qty, or below min lot size.")
                            binance.log_trade("SELL", symbol, qty, market_data.price, est_fee, "SKIPPED", "High fee, zero qty, or below min lot size")
                    elif "sell" in rec_text:
                        base_asset = symbol.replace("USDT", "")
                        asset_balance = binance.get_balance(base_asset)
                        if market_data.price == 0:
                            logger.warning(f"Skipping {symbol} sell: price is zero.")
                            continue
                        qty = round(asset_balance * 0.4, 6)
                        est_fee = qty * market_data.price * fee
                        logger.info(f"Estimated SELL fee for {symbol}: {est_fee} USDT")
                        if qty > 0 and est_fee < (qty * market_data.price * 0.01):
                            try:
                                order = binance.create_order(symbol, "SELL", qty)
                                logger.info(f"SELL order placed: {order}")
                                binance.log_trade("SELL", symbol, qty, market_data.price, est_fee, "SUCCESS", str(order))
                            except Exception as oe:
                                logger.error(f"SELL order failed: {oe}")
                                binance.log_trade("SELL", symbol, qty, market_data.price, est_fee, "FAILED", str(oe))
                        else:
                            logger.info(f"SELL skipped for {symbol} due to high fee or zero qty.")
                            binance.log_trade("SELL", symbol, qty, market_data.price, est_fee, "SKIPPED", "High fee or zero qty")
                except Exception as e:
                    logger.error(f"Error in Research/AnalysisAgent for {symbol}: {e}")
            await asyncio.sleep(30)
            # Health checks, circuit breakers, etc.
