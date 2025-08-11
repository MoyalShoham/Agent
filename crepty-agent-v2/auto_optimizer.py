"""
Auto-Optimizer: Periodically backtests and optimizes strategy parameters using recent data.
"""
import time
import importlib
import pandas as pd
import os
from backtest_optimize import optimize_params, load_data

STRATEGY_DIR = os.path.join(os.path.dirname(__file__), 'trading_bot', 'strategies')
DATA_FILE = 'BTCUSDT_1h.csv'  # Change as needed
OPTIMIZE_INTERVAL = 3600  # seconds (1 hour)

# Example: parameter grids for each strategy
PARAM_GRIDS = {
    'rsi_strategy': {'rsi_buy': range(20, 35, 2), 'rsi_sell': range(65, 80, 2)},
    # Add grids for other strategies as needed
}

def auto_optimize():
    while True:
        print("[Auto-Optimizer] Starting optimization cycle...")
        df = load_data(DATA_FILE)
        for fname in os.listdir(STRATEGY_DIR):
            if fname.endswith('_strategy.py'):
                strat_name = fname[:-3]
                mod = importlib.import_module(f'trading_bot.strategies.{strat_name}')
                grid = PARAM_GRIDS.get(strat_name)
                if grid:
                    best, result = optimize_params(mod, df, grid)
                    print(f"[Auto-Optimizer] {strat_name}: Best params: {best}, Result: {result}")
                    # Save or deploy best params (e.g., write to a config file)
                    with open(f'optimized_{strat_name}.json', 'w') as f:
                        import json
                        json.dump({'params': best, 'result': result}, f)
        print(f"[Auto-Optimizer] Sleeping {OPTIMIZE_INTERVAL} seconds...")
        time.sleep(OPTIMIZE_INTERVAL)

if __name__ == '__main__':
    auto_optimize()
