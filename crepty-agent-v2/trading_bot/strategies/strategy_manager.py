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
import hashlib
import json
from datetime import datetime

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
        self.last_selected_strategy = None
        self.last_meta_used = False
        self.meta_stats = {
            'calls': 0,
            'meta_success': 0,
            'fallback': 0,
            'feature_mismatch_adaptations': 0
        }
        self._pruned = set()
        self.meta_model_hash = None
        self.sample_log_path = 'meta_samples.csv'
        self._sample_header_written = False

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
                # Compute hash for reproducibility
                try:
                    with open(model_path, 'rb') as mf:
                        self.meta_model_hash = hashlib.sha256(mf.read()).hexdigest()[:16]
                except Exception:
                    self.meta_model_hash = None
                # Persist a small meta info file
                try:
                    with open(os.path.join('trading_bot','utils','meta_learner_model.meta.json'),'w') as mf:
                        json.dump({'loaded_at': datetime.utcnow().isoformat(), 'feature_dim': self.meta_feature_dim, 'hash': self.meta_model_hash}, mf)
                except Exception:
                    pass
                logger.info(f"MetaLearner model loaded (or updated) from {model_path} hash={self.meta_model_hash} feature_dim={self.meta_feature_dim}")
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

    def _is_pruned(self, strat_name: str) -> bool:
        return strat_name in self._pruned

    def _maybe_prune_strategy(self, strat_name: str):
        try:
            env_min_trades = int(os.getenv('MIN_PRUNE_TRADES', '30'))
            win_thr = float(os.getenv('PRUNE_WIN_RATE_THRESHOLD', '0.35'))
            decay_alpha = float(os.getenv('PRUNE_DECAY_ALPHA', '0.2'))
            max_active = int(os.getenv('MAX_ACTIVE_STRATEGIES', '0'))  # 0 = no cap
            perf = self.performance.get(strat_name)
            if not perf or self._is_pruned(strat_name):
                return
            if perf['trades'] < env_min_trades:
                return
            hist = perf.get('history', [])
            if not hist:
                return
            # decayed mean
            decayed = 0.0; wsum = 0.0
            for i, v in enumerate(reversed(hist)):
                w = (1 - decay_alpha) ** i
                decayed += w * v; wsum += w
            decayed = decayed / wsum if wsum else 0.0
            win_rate = perf['win'] / perf['trades'] if perf['trades'] else 0.0
            if decayed <= 0 and win_rate < win_thr:
                self._pruned.add(strat_name)
                logger.warning(f"[PRUNE] Strategy pruned: {strat_name} decayed={decayed:.4f} win_rate={win_rate:.2f}")
            # Optional cap: keep only top performing by decayed mean
            if max_active > 0:
                # Build decayed scores
                scores = []
                for s, p in self.performance.items():
                    if s in self._pruned:
                        continue
                    h = p.get('history', [])
                    if not h:
                        continue
                    d = 0.0; w=0.0
                    for i, v in enumerate(reversed(h)):
                        ww = (1 - decay_alpha) ** i
                        d += ww * v; w += ww
                    d = d / w if w else 0.0
                    scores.append((d, s))
                scores.sort(reverse=True)
                allowed = {s for _, s in scores[:max_active]}
                for _, s in scores[max_active:]:
                    if s not in self._pruned:
                        self._pruned.add(s)
                        logger.warning(f"[PRUNE_CAP] Strategy capped (deactivated): {s}")
        except Exception as e:
            logger.error(f"[PRUNE] Error pruning {strat_name}: {e}")

    def get_signals(self, df):
        # Enrich once per cycle
        enrich_dataframe(df)
        signals = []
        for mod in self.strategy_modules:
            try:
                strat_name = mod.__name__.split('.')[-1]
                if self._is_pruned(strat_name):
                    continue
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
        # Try pruning check
        self._maybe_prune_strategy(strat_name)

    def set_research_state(self, snapshot: dict):
        self.research_state = snapshot
        # Optionally adjust internal weights or flags later
        return self.research_state

    def _write_sample(self, regime, decision, selected_strategy, features_flat, meta_used):
        if not bool(int(os.getenv('META_SAMPLE_LOG', '1'))):
            return
        try:
            import csv
            if not self._sample_header_written or not os.path.exists(self.sample_log_path):
                with open(self.sample_log_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    header = ['timestamp','regime','decision','selected_strategy','meta_used','feature_dim']
                    # feature columns f_0..f_n
                    for i in range(len(features_flat)):
                        header.append(f'f_{i}')
                    w.writerow(header)
                self._sample_header_written = True
            with open(self.sample_log_path, 'a', newline='') as f:
                w = csv.writer(f)
                row = [datetime.utcnow().isoformat(), regime, decision, selected_strategy or '', int(meta_used), len(features_flat)] + list(features_flat)
                w.writerow(row)
        except Exception as e:
            logger.error(f"[META_SAMPLE_LOG] write error: {e}")

    def consensus_signal(self, df):
        # Enrich indicators early
        enrich_dataframe(df)
        # Meta stats init per call
        self.meta_stats['calls'] += 1
        self.last_selected_strategy = None
        self.last_meta_used = False
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
                flat_input_logged = flat_input  # keep reference
                # --- FEATURE DIMENSION ADAPTATION PATCH ---
                if self.meta_feature_dim is None:
                    self.meta_feature_dim = flat_input.shape[1]
                elif flat_input.shape[1] != self.meta_feature_dim:
                    logger.warning(f"[META] Feature length mismatch trained={self.meta_feature_dim} live={flat_input.shape[1]} -> adapting (slice/pad)")
                    self.meta_stats['feature_mismatch_adaptations'] += 1
                    import numpy as np
                    if flat_input.shape[1] > self.meta_feature_dim:
                        flat_input = flat_input[:, :self.meta_feature_dim]
                    else:
                        pad = np.zeros((1, self.meta_feature_dim - flat_input.shape[1]))
                        flat_input = np.concatenate([flat_input, pad], axis=1)
                # -------------------------------------------
                try:
                    best_idx = int(self.meta_learner.predict(flat_input)[0])
                except ValueError as ve:
                    # Secondary safeguard: adapt to underlying model's expected n_features_ if available
                    import numpy as np
                    target_dim = getattr(self.meta_learner.model, 'n_features_', None)
                    if target_dim and flat_input.shape[1] != target_dim:
                        logger.error(f"[META] Predict dimension mismatch {flat_input.shape[1]} vs model {target_dim}; retrying with adapt")
                        if flat_input.shape[1] > target_dim:
                            flat_input = flat_input[:, :target_dim]
                        else:
                            pad = np.zeros((1, target_dim - flat_input.shape[1]))
                            flat_input = np.concatenate([flat_input, pad], axis=1)
                        self.meta_feature_dim = target_dim
                        try:
                            best_idx = int(self.meta_learner.predict(flat_input)[0])
                        except Exception as ve2:
                            logger.error(f"[META] Retry predict failed: {ve2}")
                            raise ve2
                    else:
                        raise ve
                perf_keys = list(self.performance.keys())
                if best_idx < len(perf_keys):
                    best_strat = perf_keys[best_idx]
                    for name, sig in signals:
                        if name.split('.')[-1] == best_strat:
                            logger.debug(f"[CONSENSUS] MetaLearner selected {best_strat} signal={sig}")
                            self.last_selected_strategy = best_strat
                            self.last_meta_used = True
                            self.meta_stats['meta_success'] += 1
                            # Log sample before return
                            try:
                                if flat_input_logged is not None:
                                    self._write_sample(regime, sig, best_strat, flat_input_logged.flatten(), True)
                            except Exception:
                                pass
                            # Auto-bypass if persistent hold while many buys present
                            if sig == 'hold' and buy_count >= max(3, len(signals)//4):
                                logger.debug("[CONSENSUS] Auto-bypass meta (persistent hold vs multi buy signals)")
                                return self._fallback_weighted(signals, regime, buy_count, sell_count)
                            return sig
        except Exception as e:
            logger.error(f"Meta learner prediction failed: {e}; fallback.")
        decision = self._fallback_weighted(signals, regime, buy_count, sell_count)
        if not self.last_meta_used:
            self.meta_stats['fallback'] += 1
        # Append meta decision log CSV
        try:
            csv_path = 'meta_decisions_log.csv'
            exists = os.path.exists(csv_path)
            with open(csv_path, 'a') as f:
                if not exists:
                    f.write('timestamp,regime,decision,selected_strategy,meta_used,buy_count,sell_count,model_hash\n')
                f.write(f"{datetime.utcnow().isoformat()},{regime},{decision},{self.last_selected_strategy or ''},{int(self.last_meta_used)},{buy_count},{sell_count},{self.meta_model_hash or ''}\n")
        except Exception:
            pass
        # Also write sample (use flat_input_logged if available else rebuild minimal features for logging)
        try:
            if flat_input_logged is not None:
                self._write_sample(regime, decision, self.last_selected_strategy, flat_input_logged.flatten(), self.last_meta_used)
            else:
                # attempt to rebuild simple features for logging
                if features is not None and features.size > 0:
                    self._write_sample(regime, decision, self.last_selected_strategy, features.flatten(), self.last_meta_used)
        except Exception:
            pass
        # Periodic stats log
        if self.meta_stats['calls'] % int(os.getenv('META_STATS_INTERVAL','200')) == 0:
            logger.info(f"[META_STATS] {self.meta_stats}")
        return decision

    def _fallback_weighted(self, signals, regime, buy_count, sell_count):
        # Fallback: dynamic weights
        perf_weights = {}
        # --- NEW: decay & sharpe-like performance weighting ---
        decay_alpha = float(os.getenv('PERF_DECAY_ALPHA', '0.15'))  # higher = more weight to recent
        min_trades_for_perf = int(os.getenv('MIN_TRADES_FOR_PERF', '5'))
        for mod in self.strategy_modules:
            strat_name = mod.__name__.split('.')[-1]
            perf = self.performance.get(strat_name, {'pnl': 0, 'trades': 0, 'win': 0, 'history': []})
            hist = perf.get('history', [])
            # Exponential decay average of pnl
            decayed_mean = 0.0
            weight_sum = 0.0
            for i, v in enumerate(reversed(hist)):
                w = (1 - decay_alpha) ** i
                decayed_mean += w * v
                weight_sum += w
            decayed_mean = decayed_mean / weight_sum if weight_sum else 0.0
            # Win rate (recent 30 trades)
            recent = hist[-30:]
            wins_recent = sum(1 for x in recent if x > 0)
            win_rate_recent = wins_recent / len(recent) if recent else 0.0
            # Volatility penalty
            import numpy as _np
            pnl_std = _np.std(recent) if len(recent) > 5 else 0.0
            vol_penalty = 1.0 / (1.0 + pnl_std) if pnl_std > 0 else 1.0
            base = 0.1
            if perf.get('trades', 0) >= min_trades_for_perf:
                perf_score = max(0.0, decayed_mean) * 0.5 + win_rate_recent * 0.5
            else:
                perf_score = win_rate_recent * 0.3
            perf_weights[strat_name] = max(base, perf_score * vol_penalty)
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
                'vol_regime_switch_strategy': 0.5,
                'volatility_adaptive_hybrid_strategy': 0.6  # new hybrid strategy
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
                'vol_regime_switch_strategy': 0.6,
                'volatility_adaptive_hybrid_strategy': 0.5
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
                'vol_regime_switch_strategy': 0.5,
                'volatility_adaptive_hybrid_strategy': 0.7
            },
        }
        # --- NEW: group caps to reduce redundancy ---
        group_map = {
            'momentum_strategy': 'momentum',
            'ma_crossover_strategy': 'trend',
            'adaptive_kalman_trend_strategy': 'trend',
            'ema_ribbon_trend_strategy': 'trend',
            'supertrend_strategy': 'trend',
            'ichimoku_strategy': 'trend',
            'bollinger_strategy': 'mean_rev',
            'mean_reversion_strategy': 'mean_rev',
            'vwap_reversion_strategy': 'mean_rev',
            'rsi_strategy': 'osc',
            'stochastic_strategy': 'osc',
            'rsi_mfi_confluence_strategy': 'osc',
            'macd_strategy': 'momentum',
            'adx_strategy': 'trend_conf',
            'atr_trailing_stop_strategy': 'risk',
            'keltner_breakout_strategy': 'vol_break',
            'breakout_strategy': 'vol_break',
            'donchian_strategy': 'breakout',
            'volatility_expansion_strategy': 'vol',
            'volatility_compression_breakout_strategy': 'vol',
            'vol_regime_switch_strategy': 'vol',
            'cmf_trend_strategy': 'volume',
            'obv_divergence_strategy': 'volume',
            'oi_price_divergence_strategy': 'flow',
            'regime_adaptive_switch_strategy': 'meta',
            'volatility_adaptive_hybrid_strategy': 'hybrid',
            'example_strategy': 'misc'
        }
        group_cap = float(os.getenv('GROUP_MAX_FRACTION', '0.45'))  # max fraction of total vote weight per group
        weights = regime_weights.get(regime, {})
        raw_weights = {}
        for name, sig in signals:
            strat = name.split('.')[-1]
            raw_weights[strat] = weights.get(strat, 1) * perf_weights.get(strat, 1)
        # Compute initial totals per group (ignoring holds yet)
        from collections import defaultdict
        group_totals = defaultdict(float)
        for strat, w in raw_weights.items():
            grp = group_map.get(strat, 'other')
            group_totals[grp] += w
        total_weight_sum = sum(raw_weights.values()) or 1.0
        # Scale groups exceeding cap
        group_scale = {}
        for grp, tw in group_totals.items():
            frac = tw / total_weight_sum
            group_scale[grp] = min(1.0, (group_cap / frac) if frac > group_cap else 1.0)
        score = {'buy': 0, 'sell': 0, 'hold': 0}
        hold_penalty = float(os.getenv('HOLD_WEIGHT_PENALTY', '0.3'))
        dominance_ratio = float(os.getenv('DOMINANCE_RATIO', '1.4'))
        for name, sig in signals:
            strat = name.split('.')[-1]
            grp = group_map.get(strat, 'other')
            w = raw_weights.get(strat, 1) * group_scale.get(grp, 1.0)
            if sig == 'hold':
                w *= hold_penalty
            score[sig] += w
            logger.debug(f"[CONSENSUS] strat={strat} sig={sig} weight={w:.3f} group={grp} scale={group_scale.get(grp,1.0):.2f} (penalized_hold={sig=='hold'})")
        # Dominance rule
        decision = None
        if score['buy'] >= score['sell'] * dominance_ratio and score['buy'] >= score['hold'] * dominance_ratio:
            decision = 'buy'
            logger.debug(f"[CONSENSUS] Dominance rule triggered: BUY score={score}")
        elif score['sell'] >= score['buy'] * dominance_ratio and score['sell'] >= score['hold'] * dominance_ratio:
            decision = 'sell'
            logger.debug(f"[CONSENSUS] Dominance rule triggered: SELL score={score}")
        if decision is None:
            decision = max(score, key=score.get)
        logger.info(f"[CONSENSUS] (fallback) regime={regime} score={score} decision={decision} hold_penalty={hold_penalty} dominance_ratio={dominance_ratio}")
        return decision

    def get_meta_stats(self):
        return dict(self.meta_stats)
