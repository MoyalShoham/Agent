from trading_bot.utils.meta_learner import MetaLearner, build_meta_features
"""
Strategy Manager - Loads and runs all strategies, returns consensus or weighted signal.
"""

import importlib
import os
import pandas as pd
from trading_bot.utils.regime_detection import detect_regime

STRATEGY_DIR = os.path.dirname(__file__)

class StrategyManager:
    def __init__(self):
        self.strategy_modules = self._load_strategies()
        self.optimized_params = self._load_optimized_params()
        self.performance = {mod.__name__.split('.')[-1]: {'pnl': 0, 'trades': 0, 'win': 0, 'history': []} for mod in self.strategy_modules}
        self.meta_learner = MetaLearner()
        self._reload_meta_learner_model()

    def _reload_meta_learner_model(self):
        import os
        import joblib
        model_path = os.path.join('trading_bot', 'utils', 'meta_learner_model.pkl')
        if os.path.exists(model_path):
            try:
                data = joblib.load(model_path)
                self.meta_learner.model = data['model']
                self.meta_learner.strategy_list = data.get('strategies', list(self.performance.keys()))
                from loguru import logger
                logger.info(f"MetaLearner model loaded from {model_path}")
            except Exception as e:
                from loguru import logger
                logger.error(f"Failed to load MetaLearner model: {e}")

    def reload_meta_learner_periodically(self):
        # Call this in a background thread or async task if you want periodic reloads
        import threading, time
        def reload_loop():
            while True:
                self._reload_meta_learner_model()
                time.sleep(600)  # Reload every 10 minutes
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

    def consensus_signal(self, df):
        # Reload optimized params each call for live updating
        self.optimized_params = self._load_optimized_params()
        regime = detect_regime(df)
        signals = self.get_signals(df)
        if not signals:
            return 'hold'
        # Meta-learning: use meta-learner if trained, else fallback to weighted voting
        features = build_meta_features(df, self.performance, regime)
        if self.meta_learner.model is not None:
            # Predict best strategy index
            best_idx = int(self.meta_learner.predict(features)[0])
            best_strat = list(self.performance.keys())[best_idx]
            for name, sig in signals:
                if name.split('.')[-1] == best_strat:
                    return sig
        # Fallback: dynamic weights as before
        perf_weights = {}
        for mod in self.strategy_modules:
            strat_name = mod.__name__.split('.')[-1]
            perf = self.performance.get(strat_name, {'pnl': 0, 'trades': 1, 'win': 0, 'history': []})
            win_rate = perf['win'] / perf['trades'] if perf['trades'] else 0
            perf_weights[strat_name] = max(0.1, win_rate)
        regime_weights = {
            'bull': {'bollinger_strategy': 0.2, 'macd_strategy': 0.4, 'rsi_strategy': 0.4, 'momentum_strategy': 0.5, 'ma_crossover_strategy': 0.5},
            'bear': {'bollinger_strategy': 0.4, 'macd_strategy': 0.4, 'rsi_strategy': 0.2, 'mean_reversion_strategy': 0.5, 'volatility_expansion_strategy': 0.5},
            'sideways': {'bollinger_strategy': 0.5, 'macd_strategy': 0.2, 'rsi_strategy': 0.3, 'mean_reversion_strategy': 0.7, 'breakout_strategy': 0.3},
        }
        weights = regime_weights.get(regime, {})
        score = {'buy': 0, 'sell': 0, 'hold': 0}
        for name, sig in signals:
            strat = name.split('.')[-1]
            w = weights.get(strat, 1) * perf_weights.get(strat, 1)
            score[sig] += w
        best = max(score, key=score.get)
        return best
