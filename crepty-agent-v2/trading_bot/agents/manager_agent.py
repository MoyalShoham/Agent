from trading_bot.utils.risk_monitor import RiskMonitor

import asyncio
import logging
from loguru import logger
from trading_bot.config.settings import settings
from trading_bot.utils.data_models import AgentMessage
from trading_bot.utils.smart_order_router import SmartOrderRouter
from trading_bot.utils.sentiment import fetch_sentiment
from trading_bot.utils.notifications import send_notification
from trading_bot.utils.ml_signals import generate_ml_signal
import importlib
import os


class ManagerAgent:
    def __init__(self):
        self.agents = {}
        self.loop = asyncio.get_event_loop()
        self.daily_loss = 0.0
        self.max_daily_loss = float(getattr(settings, 'MAX_DAILY_LOSS', 1000))
        self.smart_router = SmartOrderRouter()
        self.trail_perc = float(os.getenv('TRAILING_STOP_PERC', 1.5))
        self.take_profit_perc = float(os.getenv('TAKE_PROFIT_PERC', 2.0))
        self.position_sizing_mode = os.getenv('POSITION_SIZING_MODE', 'fixed')
        self.risk_monitor = RiskMonitor()
        # Advanced modules
        from trading_bot.utils import adaptive_position_sizing, portfolio_optimizer, multi_timeframe, continuous_learning, alpha_signal_stacker, regime_switcher, order_execution, onchain_social_analytics, risk_controls, performance_monitor, backtest_simulator
        self.adaptive_position_sizing = adaptive_position_sizing
        self.portfolio_optimizer = portfolio_optimizer
        self.multi_timeframe = multi_timeframe
        self.continuous_learning = continuous_learning
        self.alpha_signal_stacker = alpha_signal_stacker
        self.regime_switcher = regime_switcher
        self.order_execution = order_execution
        self.onchain_social_analytics = onchain_social_analytics
        self.risk_controls = risk_controls
        self.performance_monitor = performance_monitor
        self.backtest_simulator = backtest_simulator
        logger.info("ManagerAgent initialized with super-broker and advanced modules.")

    def register_agent(self, name, agent):
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")

    async def send_message(self, message: AgentMessage):
        recipient = self.agents.get(message.recipient)
        if recipient:
            await recipient.receive_message(message)
        else:
            logger.error(f"Recipient agent {message.recipient} not found.")


    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "SHIBUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT", "UNIUSDT", "XLMUSDT"]

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
        logger.info("ManagerAgent running main event loop with advanced modules.")
        from trading_bot.agents.research_agent import ResearchAgent
        from trading_bot.agents.analysis_agent import AnalysisAgent
        from trading_bot.utils.binance_client import BinanceClient
        import datetime
        from trading_bot.strategies.strategy_manager import StrategyManager
        strategy_manager = StrategyManager()
        strategy_manager.reload_meta_learner_periodically()
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

        # Periodic meta-learner retraining (continuous learning)
        import threading
        def retrain_loop():
            while True:
                try:
                    self.continuous_learning.retrain_meta_learner('trade_log.csv', 'trading_bot/utils/meta_learner_model.pkl')
                except Exception as e:
                    logger.error(f"Meta-learner retrain failed: {e}")
                import time; time.sleep(3600)
        threading.Thread(target=retrain_loop, daemon=True).start()

        while True:
            logger.info(f"ManagerAgent heartbeat: {datetime.datetime.utcnow().isoformat()}")
            now = datetime.datetime.utcnow().date()
            if now != last_pnl_check:
                self.daily_loss = 0.0
                last_pnl_check = now

            # Research and add a new hot coin every 10 cycles
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

            # Portfolio optimization (capital allocation)
            try:
                # Example: allocate capital using optimizer (placeholder: equal weights)
                returns = [0.01 for _ in self.symbols]  # Replace with real returns
                cov_matrix = [[0.01 for _ in self.symbols] for _ in self.symbols]  # Replace with real cov
                weights = self.portfolio_optimizer.optimize_portfolio(returns, cov_matrix)
            except Exception as e:
                logger.error(f"Portfolio optimization failed: {e}")
                weights = [1/len(self.symbols)] * len(self.symbols)

            # 70% allocation: advanced logic
            for i, symbol in enumerate(self.symbols):
                try:
                    research_agent = ResearchAgent(symbol)
                    self.register_agent(f"research_{symbol}", research_agent)
                    market_data = await research_agent.fetch_market_data()
                    logger.info(f"Fetched market data: {market_data}")
                    sentiment = fetch_sentiment(symbol)
                    logger.info(f"Sentiment for {symbol}: {sentiment['sentiment_score']}")
                    # On-chain/social analytics
                    onchain_score = self.onchain_social_analytics.fetch_onchain_sentiment(symbol)
                    social_score = self.onchain_social_analytics.fetch_social_sentiment(symbol)
                    logger.info(f"On-chain score: {onchain_score}, Social score: {social_score}")
                    df = getattr(market_data, 'df', None)
                    # Multi-timeframe analysis
                    multi_signal = None
                    if df is not None:
                        df_dict = {'1h': df}  # Add more timeframes as needed
                        multi_signal = self.multi_timeframe.get_multi_timeframe_signals(df_dict)
                        ml_signal = generate_ml_signal(df)
                        strategy_signal = strategy_manager.consensus_signal(df)
                        regime = self.regime_switcher.detect_regime(df)
                    else:
                        ml_signal = 'hold'
                        strategy_signal = 'hold'
                        regime = 'sideways'
                    logger.info(f"ML signal: {ml_signal}, Strategy signal: {strategy_signal}, Multi-timeframe: {multi_signal}, Regime: {regime}")
                    # Alpha stacking
                    alpha_score = self.alpha_signal_stacker.stack_alpha_signals({
                        'technical': strategy_signal,
                        'ml': ml_signal,
                        'sentiment': sentiment['sentiment_score'],
                        'onchain': onchain_score,
                        'social': social_score,
                        'multi': multi_signal
                    })
                    logger.info(f"Alpha score: {alpha_score}")
                    recommendation = await analysis_agent.analyze(market_data)
                    logger.info(f"AnalysisAgent recommendation: {recommendation}")
                    rec_val = ""
                    if isinstance(recommendation, dict):
                        rec_val = recommendation.get("recommendation", "")
                    # Final decision: combine all signals
                    rec_text = str(strategy_signal or ml_signal or rec_val or "hold").lower()
                    if abs(alpha_score) < 1:
                        rec_text = 'hold'
                    fee = market_data.indicators.get("fee", 0.001)
                    if self.daily_loss <= -self.max_daily_loss or self.risk_controls.check_circuit_breaker(self.daily_loss, 0.2):
                        logger.warning(f"Max daily loss or circuit breaker triggered: {self.daily_loss}. Trading paused.")
                        send_notification(f"Max daily loss or circuit breaker triggered: {self.daily_loss}. Trading paused.")
                        continue
                    if "buy" in rec_text or "sell" in rec_text:
                        trades_this_hour += 1
                        unique_coins_traded.add(symbol)
                    usdt_balance = binance.get_balance("USDT")
                    price = market_data.price
                    min_qty, step_size = binance.get_lot_size(symbol)
                    min_notional = binance.get_min_notional(symbol)
                    # Adaptive position sizing
                    volatility = 0.02  # Placeholder, replace with real
                    max_drawdown = 0.05  # Placeholder, replace with real
                    win_rate = 0.6  # Placeholder, replace with real
                    base_frac = weights[i] * 0.7
                    qty = self.adaptive_position_sizing.calculate_position_size(usdt_balance, volatility, max_drawdown, win_rate, base_frac=base_frac) / price if price > 0 else 0
                    if step_size > 0:
                        qty = qty - (qty % step_size)
                    qty = round(qty, 8)
                    notional = qty * price
                    fmt_qty = binance.format_quantity(qty, step_size)
                    trailing_stop = price * (1 - self.trail_perc / 100)
                    take_profit = price * (1 + self.take_profit_perc / 100)
                    # Order execution algorithms (TWAP example)
                    if "buy" in rec_text:
                        min_usdt_needed = price * min_qty
                        if usdt_balance < min_usdt_needed:
                            continue
                        est_fee = qty * price * fee
                        logger.info(f"Estimated BUY fee for {symbol}: {est_fee} USDT")
                        if qty >= min_qty and notional >= min_notional and est_fee < (qty * price * 0.01):
                            try:
                                twap_orders = self.order_execution.twap_order(symbol, float(fmt_qty), price)
                                for twap_symbol, twap_qty, twap_price in twap_orders:
                                    route_result = self.smart_router.route_order(twap_symbol, "BUY", twap_qty, twap_price)
                                    logger.info(f"Order routed: {route_result}")
                                    order = binance.create_order(twap_symbol, "BUY", twap_qty)
                                    logger.info(f"BUY order placed: {order}")
                                    binance.log_trade("BUY", twap_symbol, twap_qty, twap_price, est_fee, "SUCCESS", str(order), strategy=strategy_signal)
                                    trade_pnl = -twap_qty * twap_price - est_fee
                                    self.daily_loss += trade_pnl
                                    self.risk_monitor.update(trade_pnl)
                                    trade_results.append(("SUCCESS", twap_qty))
                                    send_notification(f"BUY {twap_symbol} {twap_qty} at {twap_price}")
                            except Exception as oe:
                                logger.error(f"BUY order failed: {oe}")
                                binance.log_trade("BUY", symbol, float(fmt_qty), price, est_fee, "FAILED", str(oe), strategy=strategy_signal)
                                trade_results.append(("FAILED", qty))
                        else:
                            logger.info(f"BUY skipped for {symbol} due to high fee, zero qty, below min lot size, or notional.")
                            binance.log_trade("BUY", symbol, float(fmt_qty), price, est_fee, "SKIPPED", "High fee, zero qty, below min lot size, or notional", strategy=strategy_signal)
                            trade_results.append(("SKIPPED", qty))
                    elif "sell" in rec_text:
                        base_asset = symbol.replace("USDT", "")
                        asset_balance = binance.get_balance(base_asset)
                        if price == 0:
                            logger.warning(f"Skipping {symbol} sell: price is zero.")
                            continue
                        min_qty, step_size = binance.get_lot_size(symbol)
                        min_notional = binance.get_min_notional(symbol)
                        qty = asset_balance * weights[i]
                        if step_size > 0:
                            qty = qty - (qty % step_size)
                        qty = round(qty, 8)
                        notional = qty * price
                        fmt_qty = binance.format_quantity(qty, step_size)
                        est_fee = qty * price * fee
                        logger.info(f"Estimated SELL fee for {symbol}: {est_fee} USDT")
                        if qty >= min_qty and notional >= min_notional and est_fee < (qty * price * 0.01):
                            try:
                                twap_orders = self.order_execution.twap_order(symbol, float(fmt_qty), price)
                                for twap_symbol, twap_qty, twap_price in twap_orders:
                                    route_result = self.smart_router.route_order(twap_symbol, "SELL", twap_qty, twap_price)
                                    logger.info(f"Order routed: {route_result}")
                                    order = binance.create_order(twap_symbol, "SELL", twap_qty)
                                    logger.info(f"SELL order placed: {order}")
                                    binance.log_trade("SELL", twap_symbol, twap_qty, twap_price, est_fee, "SUCCESS", str(order), strategy=strategy_signal)
                                    trade_pnl = twap_qty * twap_price - est_fee
                                    self.daily_loss += trade_pnl
                                    self.risk_monitor.update(trade_pnl)
                                    trade_results.append(("SUCCESS", twap_qty))
                                    send_notification(f"SELL {twap_symbol} {twap_qty} at {twap_price}")
                            except Exception as oe:
                                logger.error(f"SELL order failed: {oe}")
                                binance.log_trade("SELL", symbol, float(fmt_qty), price, est_fee, "FAILED", str(oe), strategy=strategy_signal)
                                trade_results.append(("FAILED", qty))
                        else:
                            logger.info(f"SELL skipped for {symbol} due to high fee, zero qty, below min lot size, or notional.")
                            binance.log_trade("SELL", symbol, float(fmt_qty), price, est_fee, "SKIPPED", "High fee, zero qty, below min lot size, or notional", strategy=strategy_signal)
                            trade_results.append(("SKIPPED", qty))
                    # Performance monitoring & auto-shutdown
                    try:
                        strat_name = 'strategy_manager'  # Replace with actual
                        win_rate = 0.6  # Placeholder
                        pnl = 100  # Placeholder
                        if self.performance_monitor.should_disable_strategy(win_rate, pnl):
                            logger.warning(f"Strategy {strat_name} disabled due to poor performance.")
                            send_notification(f"Strategy {strat_name} disabled due to poor performance.")
                            continue
                    except Exception as e:
                        logger.error(f"Performance monitor error: {e}")
                except Exception as e:
                    logger.error(f"Error in Research/AnalysisAgent for {symbol}: {e}")

            # 30% allocation: microcap strategy (unchanged, can be enhanced similarly)
            try:
                all_tickers = binance.client.get_ticker()
                microcaps = sorted([t for t in all_tickers if t['symbol'].endswith('USDT') and t['symbol'] not in self.symbols], key=lambda x: float(x.get('quoteVolume', 0)))[:10]
                for micro in microcaps:
                    micro_symbol = micro['symbol']
                    research_agent = ResearchAgent(micro_symbol)
                    self.register_agent(f"microcap_{micro_symbol}", research_agent)
                    market_data = await research_agent.fetch_market_data()
                    logger.info(f"[MICROCAP] Fetched market data: {market_data}")
                    price = market_data.price
                    volume = market_data.volume
                    min_qty, step_size = binance.get_lot_size(micro_symbol)
                    min_notional = binance.get_min_notional(micro_symbol)
                    usdt_balance = binance.get_balance("USDT")
                    qty = (usdt_balance * 0.3) / price if price > 0 else 0
                    if step_size > 0:
                        qty = qty - (qty % step_size)
                    qty = round(qty, 8)
                    notional = qty * price
                    fmt_qty = binance.format_quantity(qty, step_size)
                    if float(micro.get('priceChangePercent', 0)) > 5 or float(micro.get('quoteVolume', 0)) > 2 * float(micro.get('volume', 0)):
                        est_fee = qty * price * 0.001
                        logger.info(f"[MICROCAP] Considering BUY {micro_symbol}: qty={qty}, price={price}, notional={notional}, fee={est_fee}")
                        if qty >= min_qty and notional >= min_notional and est_fee < (qty * price * 0.01):
                            try:
                                twap_orders = self.order_execution.twap_order(micro_symbol, float(fmt_qty), price)
                                for twap_symbol, twap_qty, twap_price in twap_orders:
                                    route_result = self.smart_router.route_order(twap_symbol, "BUY", twap_qty, twap_price)
                                    logger.info(f"[MICROCAP] Order routed: {route_result}")
                                    order = binance.create_order(twap_symbol, "BUY", twap_qty)
                                    logger.info(f"[MICROCAP] BUY order placed: {order}")
                                    binance.log_trade("BUY", twap_symbol, twap_qty, twap_price, est_fee, "SUCCESS", str(order), strategy="microcap")
                                    trade_results.append(("SUCCESS", twap_qty))
                                    send_notification(f"[MICROCAP] BUY {twap_symbol} {twap_qty} at {twap_price}")
                            except Exception as oe:
                                logger.error(f"[MICROCAP] BUY order failed: {oe}")
                                binance.log_trade("BUY", micro_symbol, float(fmt_qty), price, est_fee, "FAILED", str(oe), strategy="microcap")
                                trade_results.append(("FAILED", qty))
                        else:
                            logger.info(f"[MICROCAP] BUY skipped for {micro_symbol} due to high fee, zero qty, below min lot size, or notional.")
                            binance.log_trade("BUY", micro_symbol, float(fmt_qty), price, est_fee, "SKIPPED", "High fee, zero qty, below min lot size, or notional", strategy="microcap")
                            trade_results.append(("SKIPPED", qty))
            except Exception as e:
                logger.error(f"Error in Microcap strategy: {e}")
            await asyncio.sleep(30)
            # Health checks, circuit breakers, dashboard hooks, etc.
            # (Dashboard runs as separate Flask process)
