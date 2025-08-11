from trading_bot.utils.risk_monitor import RiskMonitor

import asyncio
import logging
from loguru import logger
logger.add("trading_bot.log", rotation="10 MB", retention="10 days")
from trading_bot.config.settings import settings
from trading_bot.utils.data_models import AgentMessage
from trading_bot.utils.smart_order_router import SmartOrderRouter
from trading_bot.utils.sentiment import fetch_sentiment
from trading_bot.utils.notifications import send_notification
from trading_bot.utils.ml_signals import generate_ml_signal
import importlib
import os


class ManagerAgent:

    async def run(self):
        import datetime
        from trading_bot.agents.research_agent import ResearchAgent
        from trading_bot.agents.analysis_agent import AnalysisAgent
        from trading_bot.utils.binance_client import BinanceClient
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
        current_holding_symbol = None
        current_holding_qty = 0.0

        import threading
        def retrain_loop():
            while True:
                try:
                    self.continuous_learning.retrain_meta_learner('trade_log.csv', 'trading_bot/utils/meta_learner_model.pkl')
                except Exception as e:
                    logger.error(f"Meta-learner retrain failed: {e}")
                import time; time.sleep(3600)
        threading.Thread(target=retrain_loop, daemon=True).start()

        # --- BEGIN PATCH: Cache valid spot symbols and filter microcap pairs ---
        # Cache valid spot symbols at startup
        valid_spot_symbols = set()
        stablecoin_names = ["USDT", "BUSD", "USDC", "TUSD", "DAI", "FDUSD"]
        try:
            exchange_info = binance.client.get_exchange_info()
            for s in exchange_info['symbols']:
                if s['status'] == 'TRADING':
                    valid_spot_symbols.add(s['symbol'])
        except Exception as e:
            logger.error(f"[INIT] Could not fetch exchange info for valid symbols: {e}")
        # Blacklist for symbols that repeatedly fail NOTIONAL or are invalid
        notional_blacklist = set()
        notional_fail_count = {}
        invalid_symbol_blacklist = set()
        invalid_symbol_fail_count = {}
        # Config: require all 3 signals to be buy, or just 2 out of 3
        require_all_signals = bool(int(os.getenv('REQUIRE_ALL_SIGNALS', '1')))
        while True:
            logger.info(f"ManagerAgent heartbeat: {datetime.datetime.utcnow().isoformat()}")
            now = datetime.datetime.utcnow().date()
            if now != last_pnl_check:
                self.daily_loss = 0.0
                last_pnl_check = now

            # ...existing code for research, summary, allocation, etc...

            # --- Microcap strategy with valid symbol filtering ---
            try:
                all_tickers = binance.client.get_ticker()
                # Only consider microcaps with nonzero price/volume, valid spot symbol, and not a stablecoin-to-stablecoin pair
                def is_stable_pair(symbol):
                    return any(symbol.startswith(stable) for stable in stablecoin_names) and any(symbol.endswith(stable) for stable in stablecoin_names if not symbol.startswith(stable))
                microcaps = [t for t in all_tickers if t['symbol'].endswith('USDT') and t['symbol'] not in self.symbols and t['symbol'] in valid_spot_symbols and not is_stable_pair(t['symbol'])]
                filtered_microcaps = []
                for t in microcaps:
                    try:
                        price = float(t.get('lastPrice', 0) or t.get('price', 0) or 0)
                        quote_vol = float(t.get('quoteVolume', 0) or 0)
                        base_vol = float(t.get('volume', 0) or 0)
                        if price > 0 and (quote_vol > 0 or base_vol > 0):
                            filtered_microcaps.append(t)
                    except Exception as e:
                        logger.exception(f"[MICROCAP] Error processing microcap ticker {t.get('symbol', 'unknown')}: {e}")
                filtered_microcaps = sorted(filtered_microcaps, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)[:10]
                # Blacklist for symbols that repeatedly fail NOTIONAL
                notional_blacklist = set()
                notional_fail_count = {}
                for micro in filtered_microcaps:
                    try:
                        micro_symbol = micro['symbol']
                        if micro_symbol in notional_blacklist or micro_symbol in invalid_symbol_blacklist:
                            logger.info(f"[MICROCAP] Skipping blacklisted symbol: {micro_symbol}")
                            continue
                        if micro_symbol not in valid_spot_symbols or is_stable_pair(micro_symbol):
                            logger.info(f"[MICROCAP] Skipping invalid or stablecoin pair: {micro_symbol}")
                            continue
                        research_agent = ResearchAgent(micro_symbol)
                        self.register_agent(f"microcap_{micro_symbol}", research_agent)
                        try:
                            market_data = await research_agent.fetch_market_data()
                        except Exception as e:
                            # Blacklist symbol if repeated invalid symbol errors
                            if 'Invalid symbol' in str(e):
                                invalid_symbol_fail_count[micro_symbol] = invalid_symbol_fail_count.get(micro_symbol, 0) + 1
                                if invalid_symbol_fail_count[micro_symbol] >= 2:
                                    invalid_symbol_blacklist.add(micro_symbol)
                                    logger.info(f"[MICROCAP] Blacklisted {micro_symbol} after repeated INVALID SYMBOL errors.")
                            logger.exception(f"[MICROCAP] Error fetching market data for {micro_symbol}: {e}")
                            continue
                        logger.info(f"[MICROCAP] Fetched market data: {market_data}")
                        # Fetch minNotional and lot size
                        try:
                            symbol_info = binance.client.get_symbol_info(micro_symbol)
                        except Exception as e:
                            logger.exception(f"[MICROCAP] Error fetching symbol info for {micro_symbol}: {e}")
                            continue
                        min_notional = None
                        min_qty = 0.0
                        step_size = 0.0
                        logger.info(f"[MICROCAP][DEBUG] symbol_info['filters'] for {micro_symbol}: {symbol_info['filters']}")
                        for f in symbol_info['filters']:
                            if f['filterType'] == 'NOTIONAL':
                                min_notional = float(f['minNotional'])
                            if f['filterType'] == 'LOT_SIZE':
                                min_qty = float(f['minQty'])
                                step_size = float(f['stepSize'])
                        # Skip microcaps with minNotional > $10 (configurable)
                        if min_notional is not None and min_notional > 10:
                            logger.info(f"[MICROCAP] Skipping {micro_symbol}: minNotional {min_notional} > $10")
                            continue
                        try:
                            usdt_balance = binance.get_balance("USDT")
                        except Exception as e:
                            logger.exception(f"[MICROCAP] Error getting USDT balance: {e}")
                            continue
                        # Fetch latest price before order
                        try:
                            price = binance.get_price(micro_symbol)
                        except Exception as e:
                            logger.exception(f"[MICROCAP] Error fetching latest price for {micro_symbol}: {e}")
                            continue
                        # Calculate qty to try to meet minNotional (with 15% buffer)
                        min_notional_buffer = min_notional * 1.15 if min_notional else 0
                        qty = (usdt_balance * 0.5) / price if price > 0 else 0
                        if step_size > 0:
                            qty = qty - (qty % step_size)
                        qty = max(qty, min_qty)
                        qty = float(binance.format_quantity(qty, step_size))
                        notional = qty * price
                        logger.info(f"[MICROCAP][DEBUG] Order payload: symbol={micro_symbol}, qty={qty}, price={price}, notional={notional}, min_notional={min_notional}, min_notional_buffer={min_notional_buffer}, min_qty={min_qty}, step_size={step_size}, usdt_balance={usdt_balance}")
                        # Strictly check minNotional with buffer after rounding
                        if min_notional is not None and notional < min_notional_buffer:
                            logger.warning(f"[MICROCAP] SKIP: {micro_symbol} notional {notional} < minNotional buffer {min_notional_buffer} after rounding. No order sent.")
                            binance.log_trade("BUY", micro_symbol, qty, price, 0, "SKIPPED", f"Notional {notional} < min_notional buffer {min_notional_buffer}", strategy="microcap")
                            trade_results.append(("SKIPPED", qty))
                            notional_fail_count[micro_symbol] = notional_fail_count.get(micro_symbol, 0) + 1
                            if notional_fail_count[micro_symbol] >= 2:
                                notional_blacklist.add(micro_symbol)
                                logger.info(f"[MICROCAP] Blacklisted {micro_symbol} after repeated NOTIONAL failures.")
                            continue
                        # --- BEGIN: Real buy/sell logic using agent/layer signals ---
                        # Example: Use analysis_agent, ML signals, and research_agent for decision
                        try:
                            # Get ML/ensemble signal (e.g., 1=buy, -1=sell, 0=hold)
                            ml_signal = generate_ml_signal(micro_symbol)
                            analysis_signal = await analysis_agent.analyze(market_data)
                            # Convert analysis_signal to int: 1=buy, 0=hold, -1=sell
                            analysis_recommendation = analysis_signal.get('recommendation', '').lower()
                            if 'buy' in analysis_recommendation:
                                analysis_signal_int = 1
                            elif 'sell' in analysis_recommendation:
                                analysis_signal_int = -1
                            else:
                                analysis_signal_int = 0
                            research_signal = await research_agent.research(market_data)
                            # Log all signal integer values
                            logger.info(f"[MICROCAP][SIGNALS_INT] symbol={micro_symbol}, ml_signal={ml_signal}, analysis_signal_int={analysis_signal_int}, research_signal={research_signal}")
                            logger.info(f"[MICROCAP][SIGNALS] symbol={micro_symbol}, ml_signal={ml_signal}, analysis_signal={analysis_signal}, research_signal={research_signal}")
                            # Buy condition: all 3 signals agree on buy, or 2 out of 3 if relaxed
                            if require_all_signals:
                                buy_condition_met = (ml_signal == 1 and analysis_signal_int == 1 and research_signal == 1)
                            else:
                                buy_condition_met = (sum([ml_signal == 1, analysis_signal_int == 1, research_signal == 1]) >= 2)
                        except Exception as sig_e:
                            logger.error(f"[MICROCAP][SIGNALS] Error fetching signals for {micro_symbol}: {sig_e}")
                            buy_condition_met = False
                        logger.info(f"[MICROCAP][BUY_CONDITION] symbol={micro_symbol}, price={price}, qty={qty}, notional={notional}, min_notional_buffer={min_notional_buffer}, market_data={market_data}, buy_condition_met={buy_condition_met}")
                        if buy_condition_met:
                            logger.info(f"[MICROCAP][BUY_CONDITION] Buy condition MET for {micro_symbol}. Placing real order.")
                            try:
                                order = binance.place_spot_order(symbol=micro_symbol, side="BUY", quantity=qty, price=price)
                                binance.log_trade("BUY", micro_symbol, qty, price, order.get('orderId', 0), "FILLED", "Order placed successfully", strategy="microcap")
                                trade_results.append(("FILLED", qty))
                            except Exception as order_e:
                                logger.error(f"[MICROCAP][ORDER] Order error for {micro_symbol}: {order_e}")
                                binance.log_trade("BUY", micro_symbol, qty, price, 0, "FAILED", f"Order error: {order_e}", strategy="microcap")
                                trade_results.append(("FAILED", qty))
                        else:
                            logger.info(f"[MICROCAP][BUY_CONDITION] Buy condition NOT met for {micro_symbol}. Skipping trade.")
                            binance.log_trade("BUY", micro_symbol, qty, price, 0, "SKIPPED", "Buy condition not met", strategy="microcap")
                            trade_results.append(("SKIPPED", qty))
                        # --- END: Real buy/sell logic ---
                    except Exception as micro_e:
                        logger.exception(f"Error in Microcap trade for {micro.get('symbol', 'unknown')}: {micro_e}")
            except Exception as e:
                logger.exception(f"Error in Microcap strategy: {e}")
            await asyncio.sleep(30)
        # --- END PATCH ---
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
        except Exception as e:
            logger.error(f"Error researching hot coin: {e}")
