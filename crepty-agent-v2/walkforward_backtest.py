"""Walk-Forward Backtest Framework

Concept:
1. Split historical OHLCV data into sequential folds: (train_window, test_window)
2. For each fold: optimize params (optional), train meta learner, evaluate on test window.
3. Aggregate performance metrics and per-strategy stats to identify underperformers.

Simplified Implementation: focuses on evaluating existing strategies without parameter optimization loop
(to integrate with optimize_params later).
"""
from __future__ import annotations
import pandas as pd
import importlib
import os
import numpy as np
import json
from trading_bot.strategies.strategy_manager import StrategyManager
from trading_bot.utils.indicator_cache import enrich_dataframe

STRATEGY_DIR = os.path.join('trading_bot','strategies')
PRUNE_FILE = 'strategy_pruning_recommendations.json'


def load_ohlcv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    return df.reset_index(drop=True)


def list_strategy_modules():
    mods = []
    for f in os.listdir(STRATEGY_DIR):
        if f.endswith('_strategy.py') and f != 'strategy_manager.py':
            name = f[:-3]
            mods.append(name)
    return mods


def generate_signals_for_window(df: pd.DataFrame, strategies: list[str]):
    enriched = enrich_dataframe(df.copy())
    signals = {s: [] for s in strategies}
    for i in range(len(enriched)):
        window = enriched.iloc[:i+1]
        for s in strategies:
            try:
                mod = importlib.import_module(f'trading_bot.strategies.{s}')
                sig = mod.generate_signal(window)
            except Exception:
                sig = 'hold'
            signals[s].append(sig)
    return pd.DataFrame({'index': range(len(enriched)), **signals})


def simple_pnl(price_series: pd.Series, signal_series: pd.Series):
    position = 0
    entry_price = 0
    pnl = 0.0
    trades = 0
    for price, sig in zip(price_series, signal_series):
        if sig == 'buy' and position == 0:
            position = 1
            entry_price = price
            trades += 1
        elif sig == 'sell' and position == 1:
            pnl += price - entry_price
            position = 0
    if position == 1:
        pnl += price_series.iloc[-1] - entry_price
    return pnl, trades


def walk_forward(df: pd.DataFrame, train_size: int = 500, test_size: int = 200):
    strategies = list_strategy_modules()
    results = []
    start = 0
    while start + train_size + test_size <= len(df):
        train_df = df.iloc[start:start+train_size].copy()
        test_df = df.iloc[start+train_size:start+train_size+test_size].copy()
        fold_id = len(results)
        sigs = generate_signals_for_window(pd.concat([train_df, test_df]), strategies)
        fold_metrics = {}
        for strat in strategies:
            strat_sigs = sigs[strat].iloc[train_size:]
            pnl, trades = simple_pnl(test_df['close'].reset_index(drop=True), strat_sigs.reset_index(drop=True))
            fold_metrics[strat] = {'pnl': pnl, 'trades': trades}
        results.append({'fold': fold_id, 'metrics': fold_metrics})
        start += test_size
    return results


def aggregate_results(results):
    agg = {}
    for r in results:
        for strat, m in r['metrics'].items():
            a = agg.setdefault(strat, {'pnl':0.0,'trades':0,'folds':0})
            a['pnl'] += m['pnl']
            a['trades'] += m['trades']
            a['folds'] += 1
    for strat, a in agg.items():
        a['avg_pnl_per_fold'] = a['pnl']/a['folds'] if a['folds'] else 0
    return agg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='BTCUSDT_1h.csv')
    parser.add_argument('--train', type=int, default=500)
    parser.add_argument('--test', type=int, default=200)
    parser.add_argument('--prune_threshold', type=float, default=0.0, help='Total pnl threshold for pruning')
    args = parser.parse_args()
    df = load_ohlcv(args.csv)
    results = walk_forward(df, train_size=args.train, test_size=args.test)
    agg = aggregate_results(results)
    print('Fold Results:')
    for r in results:
        print(r['fold'], r['metrics'])
    print('\nAggregate:')
    for strat, m in sorted(agg.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(strat, m)
    with open('walkforward_results.json','w') as f:
        json.dump({'folds': results, 'aggregate': agg}, f, indent=2)
    under = [s for s,m in agg.items() if m['pnl'] <= args.prune_threshold]
    recommendations = {
        'threshold': args.prune_threshold,
        'underperformers': under,
        'total_strategies': len(agg),
        'kept': [s for s in agg if s not in under]
    }
    with open(PRUNE_FILE, 'w') as f:
        json.dump(recommendations, f, indent=2)
    print('\nPruning recommendations written to', PRUNE_FILE)

if __name__ == '__main__':
    main()
