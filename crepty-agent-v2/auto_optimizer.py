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
    'rsi_strategy': {'rsi_buy': range(20, 35, 5), 'rsi_sell': range(60, 80, 5)},
    # Newly added strategies
    'adx_strategy': {
        'period': [10, 14, 20],
        'adx_threshold': [15, 20, 25, 30]
    },
    'vwap_reversion_strategy': {
        'lookback': [48, 96, 144],
        'deviation_pct': [0.001, 0.002, 0.003]
    },
    'keltner_breakout_strategy': {
        'ema_period': [20, 30],
        'atr_period': [14, 20],
        'atr_mult': [1.2, 1.5, 2.0],
        'squeeze_ratio_thresh': [0.9, 1.0, 1.1]
    },
    'obv_divergence_strategy': {
        'window': [30, 40, 60],
        'pivot_lookback': [5, 10]
    },
    'cmf_trend_strategy': {
        'period': [20, 30],
        'ema_period': [34, 55],
        'upper_thresh': [0.05, 0.1, 0.15],
        'lower_thresh': [-0.05, -0.1, -0.15]
    },
    'rsi_mfi_confluence_strategy': {
        'period': [10, 14],
        'oversold': [35, 40],
        'overbought': [60, 65]
    },
    'ema_ribbon_trend_strategy': {
        'pullback_depth': [0.236, 0.382, 0.5]
    },
    'adaptive_kalman_trend_strategy': {
        'lookback': [150, 200, 300],
        'process_var': [1e-4, 5e-4],
        'measure_var': [5e-3, 1e-2]
    },
    'vol_regime_switch_strategy': {
        'lookback': [40, 60],
        'vol_period': [20, 30],
        'high_vol_mult': [1.1, 1.2, 1.3],
        'k': [1.0, 1.5, 2.0]
    }
}

MAX_COMBINATIONS = 1500  # safety cap

def auto_optimize():
    while True:
        print("[Auto-Optimizer] Starting optimization cycle...")
        df = load_data(DATA_FILE)
        for fname in os.listdir(STRATEGY_DIR):
            if fname.endswith('_strategy.py') and fname != 'strategy_manager.py':
                strat_name = fname[:-3]
                try:
                    mod = importlib.import_module(f'trading_bot.strategies.{strat_name}')
                except Exception as e:
                    print(f"[Auto-Optimizer] Failed import {strat_name}: {e}")
                    continue
                grid = PARAM_GRIDS.get(strat_name)
                if grid:
                    # Rough combination count
                    combs = 1
                    for v in grid.values():
                        combs *= len(list(v)) if not isinstance(v, range) else len(list(v))
                    if combs > MAX_COMBINATIONS:
                        print(f"[Auto-Optimizer] Skip {strat_name} grid too large ({combs})")
                        continue
                    best, result = optimize_params(mod, df, grid)
                    print(f"[Auto-Optimizer] {strat_name}: Best params: {best}, Result: {result}")
                    # Save optimized params
                    try:
                        with open(f'optimized_{strat_name}.json', 'w') as f:
                            import json
                            json.dump({'params': best, 'result': result}, f)
                    except Exception as e:
                        print(f"[Auto-Optimizer] Failed to save params for {strat_name}: {e}")
        print(f"[Auto-Optimizer] Sleeping {OPTIMIZE_INTERVAL} seconds...")
        time.sleep(OPTIMIZE_INTERVAL)

if __name__ == '__main__':
    auto_optimize()
