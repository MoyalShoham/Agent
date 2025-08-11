from trading_bot.utils.meta_learner import MetaLearner, build_meta_features
from trading_bot.utils.indicator_cache import enrich_dataframe
"""
Strategy Manager - Loads and runs all strategies, returns consensus or weighted signal.
"""

import importlib
import os
import pandas as pd
from trading_bot.utils.regime_detection import detect_regime
from time import time
from loguru import logger

STRATEGY_DIR = os.path.dirname(__file__)

class StrategyManager:
    def __init__(self):
        self.strategy_modules = self._load_strategies()
        self.optimized_params = self._load_optimized_params()
        self.performance = {mod.__name__.split('.')[-1]: {'pnl': 0, 'trades': 0, 'win': 0, 'history': []} for mod in self.strategy_modules}
        self.meta_learner = MetaLearner()
        self._last_model_mtime = None
        self._model_reload_interval = 3600  # 1 hour throttle
        self._last_model_reload_time = 0
        self._reload_meta_learner_model(force=True)
        self.research_state = None  # holds latest research snapshot dict
        self.meta_feature_dim = None

    def _reload_meta_learner_model(self, force: bool = False):
        import joblib
        model_path = os.path.join('trading_bot', 'utils', 'meta_learner_model.pkl')
        now = time()
        if not force and now - self._last_model_reload_time < self._model_reload_interval:
            return
        if os.path.exists(model_path):
            try:
                mtime = os.path.getmtime(model_path)
                if not force and self._last_model_mtime is not None and mtime == self._last_model_mtime:
                    # No file change
                    self._last_model_reload_time = now
                    return
                data = joblib.load(model_path)
                self.meta_learner.model = data['model']
                self.meta_learner.strategy_list = data.get('strategies', list(self.performance.keys()))
                self.meta_feature_dim = data.get('feature_dim')
                self._last_model_mtime = mtime
                self._last_model_reload_time = now
                logger.info(f"MetaLearner model loaded (or updated) from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load MetaLearner model: {e}")
                self._last_model_reload_time = now

    def reload_meta_learner_periodically(self):
        # Call this in a background thread or async task if you want periodic reloads
        import threading, time as _t
        def reload_loop():
            while True:
                self._reload_meta_learner_model()
                _t.sleep(300)  # check every 5 min; will only reload if interval & mtime conditions met
        t = threading.Thread(target=reload_loop, daemon=True)
        t.start()

    def _load_optimized_params(self):
        params = {}
        for fname in os.listdir(STRATEGY_DIR):
            if fname.endswith('_strategy.py'):
                strat_name = fname[:-3]
                json_path = os.path.join(os.path.dirname(STRATEGY_DIR), f'optimized_{strat_name}.json')
                if os.path.exists(json_path):
                    try:
                        import json
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            params[strat_name] = data.get('params', {})
                    except Exception as e:
                        print(f"Failed to load optimized params for {strat_name}: {e}")
        return params

    def _load_strategies(self):
        modules = []
        for fname in os.listdir(STRATEGY_DIR):
            if fname.endswith('_strategy.py'):
                mod_name = f"trading_bot.strategies.{fname[:-3]}"
                try:
                    mod = importlib.import_module(mod_name)
                    modules.append(mod)
                except Exception as e:
                    print(f"Failed to load {mod_name}: {e}")
        return modules


    def get_signals(self, df):
        # Enrich once per cycle
        enrich_dataframe(df)
        signals = []
        for mod in self.strategy_modules:
            try:
                strat_name = mod.__name__.split('.')[-1]
                params = self.optimized_params.get(strat_name, {})
                if params:
                    sig = mod.generate_signal(df, **params)
                else:
                    sig = mod.generate_signal(df)
                signals.append((mod.__name__, sig))
            except Exception as e:
                print(f"Error in {mod.__name__}: {e}")
        return signals

    def update_performance(self, strat_name, trade_pnl, win):
        perf = self.performance.setdefault(strat_name, {'pnl': 0, 'trades': 0, 'win': 0, 'history': []})
        perf['pnl'] += trade_pnl
        perf['trades'] += 1
        perf['win'] += int(win)
        perf['history'].append(trade_pnl)
        if len(perf['history']) > 100:
            perf['history'].pop(0)

    def set_research_state(self, snapshot: dict):
        self.research_state = snapshot
        # Optionally adjust internal weights or flags later
        return self.research_state

    def consensus_signal(self, df):
        # Enrich indicators early
        enrich_dataframe(df)
        # Integrate research macro bias influence (lightweight): if strong bias and meta learner absent, nudge weights
        research_bias = None
        if self.research_state and isinstance(self.research_state, dict):
            research_bias = self.research_state.get('macro_bias')
        # Reload optimized params each call for live updating
        self.optimized_params = self._load_optimized_params()
        regime = detect_regime(df)
        signals = self.get_signals(df)
        # Simple bias override: if majority hold but research bias present, convert to bias direction
        if research_bias in ('bullish','bearish') and signals and all(s == 'hold' for _, s in signals):
            if research_bias == 'bullish':
                return 'buy'
            if research_bias == 'bearish':
                return 'sell'
        logger.debug(f"[CONSENSUS] Regime={regime} raw_signals={{" + ", ".join(f"{name.split('.')[-1]}:{sig}" for name, sig in signals) + "}}")
        if not signals:
            return 'hold'
        # Optional bypass via env flag
        import os
        bypass_meta = os.getenv('BYPASS_META', '0') == '1'
        # Count basic aggregate for auto-bypass condition
        buy_count = sum(1 for _, s in signals if s == 'buy')
        sell_count = sum(1 for _, s in signals if s == 'sell')
        # If meta model absent or explicitly bypassed, skip to fallback
        if self.meta_learner.model is None or bypass_meta:
            return self._fallback_weighted(signals, regime, buy_count, sell_count)
        # Meta-learning
        features = build_meta_features(df, self.performance, regime, research=self.research_state)
        try:
            if self.meta_learner.model is not None and features.size > 0:
                flat_input = features.flatten().reshape(1, -1)
                if self.meta_feature_dim and flat_input.shape[1] != self.meta_feature_dim:
                    logger.warning(f"[META] Feature length mismatch trained={self.meta_feature_dim} live={flat_input.shape[1]} -> resetting meta model and falling back")
                    # Reset model so fallback path is used; update expected dim for future retrains
                    self.meta_learner.model = None
                    self.meta_feature_dim = flat_input.shape[1]
                    raise RuntimeError("meta_reset")
                best_idx = int(self.meta_learner.predict(flat_input)[0])
                perf_keys = list(self.performance.keys())
                if best_idx < len(perf_keys):
                    best_strat = perf_keys[best_idx]
                    for name, sig in signals:
                        if name.split('.')[-1] == best_strat:
                            logger.debug(f"[CONSENSUS] MetaLearner selected {best_strat} signal={sig}")
                            # Auto-bypass if persistent hold while many buys present
                            if sig == 'hold' and buy_count >= max(3, len(signals)//4):
                                logger.debug("[CONSENSUS] Auto-bypass meta (persistent hold vs multi buy signals)")
                                return self._fallback_weighted(signals, regime, buy_count, sell_count)
                            return sig
        except Exception as e:
            if str(e) != 'meta_reset':
                logger.error(f"Meta learner prediction failed: {e}; fallback.")
        return self._fallback_weighted(signals, regime, buy_count, sell_count)

    def _fallback_weighted(self, signals, regime, buy_count, sell_count):
        # Fallback: dynamic weights
        perf_weights = {}
        for mod in self.strategy_modules:
            strat_name = mod.__name__.split('.')[-1]
            perf = self.performance.get(strat_name, {'pnl': 0, 'trades': 1, 'win': 0, 'history': []})
            win_rate = perf['win'] / perf['trades'] if perf['trades'] else 0
            perf_weights[strat_name] = max(0.1, win_rate)
        regime_weights = {
            'bull': {
                'bollinger_strategy': 0.2,
                'macd_strategy': 0.4,
                'rsi_strategy': 0.4,
                'momentum_strategy': 0.5,
                'ma_crossover_strategy': 0.5,
                'supertrend_strategy': 0.7,
                'donchian_strategy': 0.5,
                'ichimoku_strategy': 0.6,
                'stochastic_strategy': 0.3,
                'atr_trailing_stop_strategy': 0.5,
                'adx_strategy': 0.6,
                'vwap_reversion_strategy': 0.4,
                'keltner_breakout_strategy': 0.6,
                'obv_divergence_strategy': 0.3,
                'cmf_trend_strategy': 0.5,
                'rsi_mfi_confluence_strategy': 0.5,
                'ema_ribbon_trend_strategy': 0.7,
                'adaptive_kalman_trend_strategy': 0.5,
                'vol_regime_switch_strategy': 0.5
            },
            'bear': {
                'bollinger_strategy': 0.4,
                'macd_strategy': 0.4,
                'rsi_strategy': 0.2,
                'mean_reversion_strategy': 0.5,
                'volatility_expansion_strategy': 0.5,
                'supertrend_strategy': 0.6,
                'donchian_strategy': 0.5,
                'ichimoku_strategy': 0.5,
                'stochastic_strategy': 0.4,
                'atr_trailing_stop_strategy': 0.6,
                'adx_strategy': 0.6,
                'vwap_reversion_strategy': 0.5,
                'keltner_breakout_strategy': 0.5,
                'obv_divergence_strategy': 0.4,
                'cmf_trend_strategy': 0.6,
                'rsi_mfi_confluence_strategy': 0.4,
                'ema_ribbon_trend_strategy': 0.5,
                'adaptive_kalman_trend_strategy': 0.5,
                'vol_regime_switch_strategy': 0.6
            },
            'sideways': {
                'bollinger_strategy': 0.5,
                'macd_strategy': 0.2,
                'rsi_strategy': 0.3,
                'mean_reversion_strategy': 0.7,
                'breakout_strategy': 0.3,
                'supertrend_strategy': 0.3,
                'donchian_strategy': 0.6,
                'ichimoku_strategy': 0.4,
                'stochastic_strategy': 0.7,
                'atr_trailing_stop_strategy': 0.4,
                'adx_strategy': 0.5,
                'vwap_reversion_strategy': 0.6,
                'keltner_breakout_strategy': 0.4,
                'obv_divergence_strategy': 0.4,
                'cmf_trend_strategy': 0.5,
                'rsi_mfi_confluence_strategy': 0.6,
                'ema_ribbon_trend_strategy': 0.4,
                'adaptive_kalman_trend_strategy': 0.4,
                'vol_regime_switch_strategy': 0.5
            },
        }
        weights = regime_weights.get(regime, {})
        score = {'buy': 0, 'sell': 0, 'hold': 0}
        # Apply hold penalization & dominance logic
        import os
        hold_penalty = float(os.getenv('HOLD_WEIGHT_PENALTY', '0.3'))  # <1 reduces hold dominance
        dominance_ratio = float(os.getenv('DOMINANCE_RATIO', '1.4'))   # how much a side must exceed others
        for name, sig in signals:
            strat = name.split('.')[-1]
            w = weights.get(strat, 1) * perf_weights.get(strat, 1)
            if sig == 'hold':
                w *= hold_penalty
            score[sig] += w
            logger.debug(f"[CONSENSUS] strat={strat} sig={sig} weight={w:.3f} (penalized_hold={sig=='hold'})")
        # Dominance rule: require side to exceed others by dominance_ratio to override hold bias
        decision = None
        if score['buy'] >= score['sell'] * dominance_ratio and score['buy'] >= score['hold'] * dominance_ratio:
            decision = 'buy'
            logger.debug(f"[CONSENSUS] Dominance rule triggered: BUY score={score}")
        elif score['sell'] >= score['buy'] * dominance_ratio and score['sell'] >= score['hold'] * dominance_ratio:
            decision = 'sell'
            logger.debug(f"[CONSENSUS] Dominance rule triggered: SELL score={score}")
        if decision is None:
            # fallback to highest (already penalized holds)
            decision = max(score, key=score.get)
        logger.info(f"[CONSENSUS] (fallback) regime={regime} score={score} decision={decision} hold_penalty={hold_penalty} dominance_ratio={dominance_ratio}")
        return decision
