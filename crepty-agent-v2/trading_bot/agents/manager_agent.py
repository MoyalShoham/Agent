from trading_bot.utils.risk_monitor import RiskMonitor
from trading_bot.utils.event_bus import subscribe

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
import csv
from datetime import datetime


class ManagerAgent:
    def __init__(self):
        """Initialize registry and state used inside run().
        Adds back previously referenced attributes (register_agent, symbols, daily_loss, continuous_learning).
        """
        self.agents = {}
        self.daily_loss = 0.0
        # Core symbols list (used to exclude when scanning microcaps); fallback to common majors
        try:
            from trading_bot.config.settings import settings as _settings
            self.symbols = [s.strip().upper() for s in getattr(_settings, 'FUTURES_SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')]
        except Exception:
            self.symbols = ['BTCUSDT','ETHUSDT']
        # Continuous learning module wrapper
        try:
            import trading_bot.utils.continuous_learning as _cl
            self.continuous_learning = _cl
        except Exception:
            self.continuous_learning = None
        # Placeholder risk monitor values
        self.max_daily_loss = getattr(_settings, 'MAX_DAILY_LOSS', 0) if ' _settings' in locals() else 0
        self.current_loss = 0.0

    def register_agent(self, name: str, agent):
        """Register an agent object by name."""
        self.agents[name] = agent
        try:
            logger.info(f"Registered agent: {name}")
        except Exception:
            pass

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
        futures_symbols = [s.strip().upper() for s in getattr(settings, 'FUTURES_SYMBOLS', 'BTCUSDT,ETHUSDT').split(',')]
        # Filter out futures symbols not supported by current futures gateway (prevents -1121)
        try:
            from trading_bot.utils import order_execution as _oe
            if _oe._gateway and hasattr(_oe._gateway, '_symbol_filters'):
                supported = set(_oe._gateway._symbol_filters.keys())
                original = list(futures_symbols)
                futures_symbols = [s for s in futures_symbols if s in supported]
                removed = set(original) - set(futures_symbols)
                if removed:
                    logger.warning(f"Removed unsupported futures symbols: {removed}")
        except Exception as _ferr:
            logger.debug(f"Futures symbol filter skip: {_ferr}")
        core_symbol = futures_symbols[0] if futures_symbols else 'BTCUSDT'
        core_research_agent = ResearchAgent(core_symbol)
        research_last_time = datetime.datetime.utcnow()
        research_interval = 300

        # Prepare secondary CSV log file
        csv_path = 'futures_trades_log.csv'
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp','symbol','signal','price','atr','target_qty','prev_pos','new_pos','side','order_id','realized_pnl','simulated'])

        def append_csv(row):
            try:
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(row)
            except Exception as e:
                logger.error(f"CSV log write error: {e}")

        # Event-driven logging for fills
        def on_order_filled(ev):
            try:
                append_csv([
                    datetime.datetime.utcnow().isoformat(),
                    ev.get('symbol'),
                    ev.get('signal',''),
                    ev.get('price'),
                    ev.get('atr'),
                    ev.get('target_qty'),
                    ev.get('prev_pos'),
                    ev.get('new_pos'),
                    ev.get('side'),
                    ev.get('order_id', ''),
                    ev.get('realized_pnl',''),
                    ev.get('simulated', False)
                ])
            except Exception as e:
                logger.error(f"on_order_filled log error: {e}")
        subscribe('ORDER_FILLED', on_order_filled)

        # Snapshot tracking
        last_snapshot_time = datetime.datetime.utcnow()
        snapshot_path = 'futures_snapshots.csv'
        if not os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, 'w', newline='') as f:
                    csv.writer(f).writerow(['timestamp','symbol','position_size','entry_price','realized_pnl','equity'])
            except Exception as e:
                logger.error(f"Snapshot file init error: {e}")

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

            # --- Microcap strategy conditional ---
            if getattr(settings, 'MICROCAP_ENABLED', True) and not settings.FUTURES_ENABLED:
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
                                # Convert analysis_signal to int using new schema key 'action' (fallback to legacy 'recommendation')
                                analysis_action_raw = (analysis_signal.get('action') or analysis_signal.get('recommendation') or '').lower()
                                if analysis_action_raw.startswith('buy'):
                                    analysis_signal_int = 1
                                elif analysis_action_raw.startswith('sell'):
                                    analysis_signal_int = -1
                                else:
                                    analysis_signal_int = 0
                                research_signal = await research_agent.research(market_data)
                                # Log all signal integer values
                                logger.info(f"[MICROCAP][SIGNALS_INT] symbol={micro_symbol}, ml_signal={ml_signal}, analysis_signal_int={analysis_signal_int}, research_signal={research_signal}")
                                logger.info(f"[MICROCAP][SIGNALS] symbol={micro_symbol}, ml_signal={ml_signal}, analysis_action={analysis_action_raw}, analysis_signal={analysis_signal}, research_signal={research_signal}")
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
            # --- End conditional ---
            # Periodic research snapshot update
            try:
                now_ts = datetime.datetime.utcnow()
                if (now_ts - research_last_time).total_seconds() >= research_interval:
                    try:
                        snapshot = await core_research_agent.research_snapshot()
                        strategy_manager.set_research_state(snapshot)
                        logger.info(f"[RESEARCH] Snapshot updated: macro_bias={snapshot.get('macro_bias')} sentiment={snapshot.get('sentiment_score')}")
                    except Exception as rse:
                        logger.error(f"[RESEARCH] snapshot error: {rse}")
                    research_last_time = now_ts
            except Exception:
                pass
            # Consensus-based futures execution (every loop on core symbols)
            try:
                for sym in futures_symbols:
                    try:
                        # Fetch full kline data for ATR (OHLC)
                        klines = binance.client.get_klines(symbol=sym, interval='1h', limit=250)
                        if not klines or len(klines) < 60:
                            continue
                        import pandas as pd
                        df = pd.DataFrame(klines, columns=['open_time','open','high','low','close','volume','close_time','qav','trades','tbbav','tbqav','ignore'])
                        df['open'] = df['open'].astype(float)
                        df['high'] = df['high'].astype(float)
                        df['low'] = df['low'].astype(float)
                        df['close'] = df['close'].astype(float)
                        # True Range & ATR (14) classic Wilder
                        tr_list = []
                        prev_close = None
                        for _, row in df.iterrows():
                            high = row['high']; low = row['low']; close = row['close']
                            if prev_close is None:
                                tr = high - low
                            else:
                                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                            tr_list.append(tr)
                            prev_close = close
                        import numpy as np
                        tr_series = pd.Series(tr_list)
                        if len(tr_series) < 15:
                            continue
                        atr = tr_series.rolling(14).mean().iloc[-1]
                        price = df['close'].iloc[-1]
                        # Pass close-only dataframe to strategy manager (some strategies expect high/low present so keep them)
                        strat_df = df[['open','high','low','close']].copy()
                        signal = strategy_manager.consensus_signal(strat_df)
                        from trading_bot.utils import order_execution as oe
                        pos_before = oe._position_manager.get_position(sym).size if oe._position_manager else 0.0
                        oe.submit_signal(sym, signal, price, atr=atr)
                    except Exception as es:
                        logger.error(f"[FUTURES_EXEC] {sym} error: {es}")
                # Snapshot code remains below
                try:
                    if (datetime.datetime.utcnow() - last_snapshot_time).total_seconds() > 300:
                        from trading_bot.utils import order_execution as oe2
                        equity = oe2._position_manager.equity if oe2._position_manager else ''
                        with open(snapshot_path, 'a', newline='') as f:
                            w = csv.writer(f)
                            for sym in futures_symbols:
                                try:
                                    pos = oe2._position_manager.get_position(sym) if oe2._position_manager else None
                                    if pos:
                                        w.writerow([
                                            datetime.datetime.utcnow().isoformat(),
                                            sym,
                                            pos.size,
                                            pos.entry_price,
                                            pos.realized_pnl,
                                            equity
                                        ])
                                except Exception as pxe:
                                    logger.error(f"Snapshot position error {sym}: {pxe}")
                        last_snapshot_time = datetime.datetime.utcnow()
                except Exception as snap_e:
                    logger.error(f"Snapshot logging failed: {snap_e}")
            except Exception as e:
                logger.error(f"[FUTURES_EXEC] loop error: {e}")
            await asyncio.sleep(30)
