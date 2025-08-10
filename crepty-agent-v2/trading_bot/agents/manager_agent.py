import asyncio
import logging
from loguru import logger
from trading_bot.config.settings import settings
from trading_bot.utils.data_models import AgentMessage

class ManagerAgent:
    def __init__(self):
        self.agents = {}
        self.loop = asyncio.get_event_loop()
        self.daily_loss = 0.0
        self.max_daily_loss = float(getattr(settings, 'MAX_DAILY_LOSS', 1000))
        logger.info("ManagerAgent initialized.")
    def __init__(self):
        self.agents = {}
        self.loop = asyncio.get_event_loop()
        self.daily_loss = 0.0
        self.max_daily_loss = float(getattr(settings, 'MAX_DAILY_LOSS', 1000))
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

    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]

    async def research_and_add_hot_coin(self, binance, min_change=None, min_volume=None):
        """Research and add a new hot coin to the symbols list if not already present. Logs analytics on top movers."""
        import os
        min_change = min_change if min_change is not None else float(os.getenv('HOT_COIN_MIN_CHANGE', 5))
        min_volume = min_volume if min_volume is not None else float(os.getenv('HOT_COIN_MIN_VOLUME', 1000000))
        try:
            tickers = binance.client.get_ticker()
            # Exclude stablecoins and already tracked symbols
            stablecoins = ["USDT", "BUSD", "USDC", "TUSD", "DAI"]
            def is_stable(symbol):
                return any(stable in symbol for stable in stablecoins)
            usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT') and t['symbol'] not in self.symbols and not is_stable(t['symbol'].replace('USDT',''))]
            # Filter by min volume and price change
            filtered = [t for t in usdt_tickers if float(t.get('quoteVolume', 0)) > min_volume and abs(float(t.get('priceChangePercent', 0))) > min_change]
            # Sort by priceChangePercent descending
            filtered.sort(key=lambda x: float(x.get('priceChangePercent', 0)), reverse=True)
            # Log analytics: top 3 gainers/losers by % and volume
            top_gainers = sorted(usdt_tickers, key=lambda x: float(x.get('priceChangePercent', 0)), reverse=True)[:3]
            top_losers = sorted(usdt_tickers, key=lambda x: float(x.get('priceChangePercent', 0)))[:3]
            top_volume = sorted(usdt_tickers, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)[:3]
            gainers_str = [f"{t['symbol']} {t['priceChangePercent']}%" for t in top_gainers]
            losers_str = [f"{t['symbol']} {t['priceChangePercent']}%" for t in top_losers]
            volume_str = [f"{t['symbol']} {t['quoteVolume']}" for t in top_volume]
            logger.info(f"Top 3 gainers: {gainers_str}")
            logger.info(f"Top 3 losers: {losers_str}")
            logger.info(f"Top 3 by volume: {volume_str}")
            if filtered:
                chosen = filtered[0]
                self.symbols.append(chosen['symbol'])
                logger.info(f"Added new hot coin to symbols: {chosen['symbol']} (change: {chosen['priceChangePercent']}%, volume: {chosen['quoteVolume']})")
            else:
                logger.info("No new hot coin met the criteria this cycle.")
        except Exception as e:
            logger.error(f"Error researching hot coin: {e}")

    async def run(self):
        logger.info("ManagerAgent running main event loop.")
        from trading_bot.agents.research_agent import ResearchAgent
        from trading_bot.agents.analysis_agent import AnalysisAgent
        from trading_bot.utils.binance_client import BinanceClient
        import datetime
        analysis_agent = AnalysisAgent()
        binance = BinanceClient()
        self.register_agent("analysis", analysis_agent)

        last_pnl_check = datetime.datetime.utcnow().date()
        hot_coin_check_counter = 0
        summary_counter = 0
        trades_this_hour = 0
        last_summary_time = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        unique_coins_traded = set()
        trade_results = []  # (status, qty)
        while True:
            logger.info(f"ManagerAgent heartbeat: {datetime.datetime.utcnow().isoformat()}")
            # Reset daily loss at UTC midnight
            now = datetime.datetime.utcnow().date()
            if now != last_pnl_check:
                self.daily_loss = 0.0
                last_pnl_check = now

            # Research and add a new hot coin every 10 cycles (about every 5 minutes if sleep=30s)
            hot_coin_check_counter += 1
            if hot_coin_check_counter >= 10:
                await self.research_and_add_hot_coin(binance)
                hot_coin_check_counter = 0

            # Hourly analytics summary
            now_time = datetime.datetime.utcnow()
            if (now_time - last_summary_time).total_seconds() >= 3600:
                try:
                    total_usdt = binance.get_total_usdt_value()
                    win_trades = [r for r in trade_results if r[0] == 'SUCCESS']
                    total_trades = len(trade_results)
                    win_rate = round(len(win_trades) / total_trades, 3) if total_trades else 0
                    avg_trade_size = round(sum(q for _, q in trade_results) / total_trades, 4) if total_trades else 0
                    with open('analytics_summary.csv', 'a', newline='') as f:
                        import csv
                        writer = csv.writer(f)
                        if f.tell() == 0:
                            writer.writerow(["timestamp", "total_usdt", "trades_this_hour", "unique_coins_traded", "win_rate", "avg_trade_size", "daily_loss"])
                        writer.writerow([
                            now_time.isoformat(),
                            round(total_usdt, 4),
                            trades_this_hour,
                            len(unique_coins_traded),
                            win_rate,
                            avg_trade_size,
                            round(self.daily_loss, 4)
                        ])
                    logger.info(f"Hourly summary: total_usdt={total_usdt}, trades_this_hour={trades_this_hour}, unique_coins_traded={len(unique_coins_traded)}, win_rate={win_rate}, avg_trade_size={avg_trade_size}, daily_loss={self.daily_loss}")
                except Exception as e:
                    logger.error(f"Error writing analytics summary: {e}")
                trades_this_hour = 0
                unique_coins_traded.clear()
                trade_results.clear()
                last_summary_time = now_time.replace(minute=0, second=0, microsecond=0)

            for symbol in self.symbols:
                research_agent = ResearchAgent(symbol)
                self.register_agent(f"research_{symbol}", research_agent)
                try:
                    market_data = await research_agent.fetch_market_data()
                    logger.info(f"Fetched market data: {market_data}")
                    recommendation = await analysis_agent.analyze(market_data)
                    logger.info(f"AnalysisAgent recommendation: {recommendation}")
                    rec_text = recommendation.get("recommendation", "").lower()
                    fee = market_data.indicators.get("fee", 0.001)
                    # Risk management: stop trading if daily loss exceeded
                    if self.daily_loss <= -self.max_daily_loss:
                        logger.warning(f"Max daily loss reached: {self.daily_loss}. Trading paused for today.")
                        continue
                    if "buy" in rec_text or "sell" in rec_text:
                        trades_this_hour += 1
                        unique_coins_traded.add(symbol)
                    if "buy" in rec_text:
                        usdt_balance = binance.get_balance("USDT")
                        price = market_data.price
                        if price == 0:
                            logger.warning(f"Skipping {symbol} buy: price is zero.")
                            continue
                        min_qty, step_size = binance.get_lot_size(symbol)
                        min_notional = binance.get_min_notional(symbol)
                        qty = (usdt_balance * 0.4) / price if price > 0 else 0
                        # Adjust qty to step size
                        if step_size > 0:
                            qty = qty - (qty % step_size)
                        qty = round(qty, 8)
                        notional = qty * price
                        fmt_qty = binance.format_quantity(qty, step_size)
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
                        if qty >= min_qty and notional >= min_notional and est_fee < (qty * price * 0.01):
                            try:
                                order = binance.create_order(symbol, "BUY", float(fmt_qty))
                                logger.info(f"BUY order placed: {order}")
                                binance.log_trade("BUY", symbol, float(fmt_qty), price, est_fee, "SUCCESS", str(order))
                                self.daily_loss -= qty * price + est_fee
                                trade_results.append(("SUCCESS", qty))
                            except Exception as oe:
                                logger.error(f"BUY order failed: {oe}")
                                binance.log_trade("BUY", symbol, float(fmt_qty), price, est_fee, "FAILED", str(oe))
                                trade_results.append(("FAILED", qty))
                        else:
                            logger.info(f"BUY skipped for {symbol} due to high fee, zero qty, below min lot size, or notional.")
                            binance.log_trade("BUY", symbol, float(fmt_qty), price, est_fee, "SKIPPED", "High fee, zero qty, below min lot size, or notional")
                            trade_results.append(("SKIPPED", qty))
                    elif "sell" in rec_text:
                        base_asset = symbol.replace("USDT", "")
                        asset_balance = binance.get_balance(base_asset)
                        if market_data.price == 0:
                            logger.warning(f"Skipping {symbol} sell: price is zero.")
                            continue
                        min_qty, step_size = binance.get_lot_size(symbol)
                        min_notional = binance.get_min_notional(symbol)
                        qty = asset_balance * 0.4
                        if step_size > 0:
                            qty = qty - (qty % step_size)
                        qty = round(qty, 8)
                        notional = qty * market_data.price
                        fmt_qty = binance.format_quantity(qty, step_size)
                        est_fee = qty * market_data.price * fee
                        logger.info(f"Estimated SELL fee for {symbol}: {est_fee} USDT")
                        if qty >= min_qty and notional >= min_notional and est_fee < (qty * market_data.price * 0.01):
                            try:
                                order = binance.create_order(symbol, "SELL", float(fmt_qty))
                                logger.info(f"SELL order placed: {order}")
                                binance.log_trade("SELL", symbol, float(fmt_qty), market_data.price, est_fee, "SUCCESS", str(order))
                                self.daily_loss += qty * market_data.price - est_fee
                                trade_results.append(("SUCCESS", qty))
                            except Exception as oe:
                                logger.error(f"SELL order failed: {oe}")
                                binance.log_trade("SELL", symbol, float(fmt_qty), market_data.price, est_fee, "FAILED", str(oe))
                                trade_results.append(("FAILED", qty))
                        else:
                            logger.info(f"SELL skipped for {symbol} due to high fee, zero qty, below min lot size, or notional.")
                            binance.log_trade("SELL", symbol, float(fmt_qty), market_data.price, est_fee, "SKIPPED", "High fee, zero qty, below min lot size, or notional")
                            trade_results.append(("SKIPPED", qty))
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
