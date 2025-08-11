"""
Backtest and optimize any strategy in trading_bot/strategies.
Usage: python backtest_optimize.py --strategy rsi_strategy --param rsi_buy=25 --param rsi_sell=75
"""
import importlib
import pandas as pd
import numpy as np
import argparse
import os

def load_data(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    if 'timestamp' in df:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def run_backtest(strategy_mod, df, params=None):
    params = params or {}
    signals = []
    for i in range(len(df)):
        window = df.iloc[:i+1].copy()
        sig = strategy_mod.generate_signal(window, **params) if params else strategy_mod.generate_signal(window)
        signals.append(sig)
    df['signal'] = signals
    # Simple backtest: buy/sell at close, track PnL
    position = 0
    entry_price = 0
    pnl = 0
    trades = 0
    for i, row in df.iterrows():
        if row['signal'] == 'buy' and position == 0:
            position = 1
            entry_price = row['close']
            trades += 1
        elif row['signal'] == 'sell' and position == 1:
            pnl += row['close'] - entry_price
            position = 0
    # If still in position, close at last price
    if position == 1:
        pnl += df.iloc[-1]['close'] - entry_price
    return {'pnl': pnl, 'trades': trades}

def optimize_params(strategy_mod, df, param_grid):
    from itertools import product
    best = None
    best_result = None
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        try:
            result = run_backtest(strategy_mod, df, params)
            if not best_result or result['pnl'] > best_result['pnl']:
                best = params
                best_result = result
        except Exception as e:
            print(f"Error with params {params}: {e}")
    return best, best_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--csv', default='trade_log.csv')
    parser.add_argument('--param', action='append', default=[])
    parser.add_argument('--optimize', action='store_true')
    args = parser.parse_args()
    mod = importlib.import_module(f'trading_bot.strategies.{args.strategy}')
    df = load_data(args.csv)
    param_dict = {}
    for p in args.param:
        k, v = p.split('=')
        try:
            param_dict[k] = float(v)
        except ValueError:
            param_dict[k] = v
    if args.optimize:
        # Example grid for RSI
        grid = {'rsi_buy': range(20, 35, 2), 'rsi_sell': range(65, 80, 2)}
        best, result = optimize_params(mod, df, grid)
        print(f'Best params: {best}, Result: {result}')
    else:
        result = run_backtest(mod, df, param_dict)
        print(f'Result: {result}')

if __name__ == '__main__':
    main()
