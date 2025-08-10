"""
Strategy Manager - Loads and runs all strategies, returns consensus or weighted signal.
"""
import importlib
import os
import pandas as pd

STRATEGY_DIR = os.path.dirname(__file__)

class StrategyManager:
    def __init__(self):
        self.strategy_modules = self._load_strategies()

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
                sig = mod.generate_signal(df)
                signals.append(sig)
            except Exception as e:
                print(f"Error in {mod.__name__}: {e}")
        return signals

    def consensus_signal(self, df):
        signals = self.get_signals(df)
        if not signals:
            return 'hold'
        # Majority vote, prefer buy/sell over hold
        buy = signals.count('buy')
        sell = signals.count('sell')
        if buy > sell and buy > 0:
            return 'buy'
        elif sell > buy and sell > 0:
            return 'sell'
        return 'hold'
