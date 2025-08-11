"""Generate synthetic meta-learner training dataset.

Logic:
- Load historical price CSVs for symbols (default BTCUSDT_1h.csv if present).
- Construct minimal OHLC dataframe (close given, synth high/low using small noise around close if not provided).
- Iterate over rolling window; at each step, run every loaded strategy's generate_signal on the slice.
- Map signal -> directional return using next bar close change (buy = +ret, sell = -ret, hold = 0).
- Label = strategy with max directional return (ties -> first).
- Track per-strategy performance dict to feed build_meta_features (updates with realized return per step).
- Extract feature matrix via build_meta_features and flatten; store row per timestamp with label strategy name.
- Output CSV with columns: timestamp,label_strategy,feature_0...feature_N.
"""
from __future__ import annotations
import os
import importlib
import pandas as pd
import numpy as np
from datetime import datetime
from trading_bot.strategies.strategy_manager import StrategyManager
from trading_bot.utils.meta_learner import build_meta_features

OUTPUT_CSV = "synthetic_meta_training.csv"
MIN_HISTORY = 120  # bars before starting labeling
LOOKAHEAD = 1      # bars ahead for PnL evaluation

SIG_RET_MAP = {  # function takes signal string and return pct
    'buy': lambda r: r,
    'sell': lambda r: -r,
    'hold': lambda r: 0.0,
}

def load_price_series():
    # Prefer BTCUSDT_1h.csv present in project root
    candidates = [f for f in os.listdir('.') if f.endswith('_1h.csv')]
    if not candidates:
        raise FileNotFoundError("No *_1h.csv price files found for synthetic generation.")
    # Use first candidate
    path = candidates[0]
    df = pd.read_csv(path)
    # Expect at least 'close' column
    if 'close' not in df.columns:
        # heuristically pick last column
        df.columns = [c.lower() for c in df.columns]
    if 'close' not in df.columns:
        raise ValueError("CSV must contain 'close' column for synthetic dataset.")
    # Create synthetic high/low if missing
    if 'high' not in df.columns or 'low' not in df.columns:
        close = df['close']
        # small random noise for realism (deterministic seed)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.001, size=len(close))
        df['high'] = close * (1 + np.abs(noise))
        df['low'] = close * (1 - np.abs(noise))
    return df.reset_index(drop=True)

def main():
    price_df = load_price_series()
    sm = StrategyManager()
    strategy_modules = sm.strategy_modules
    strategy_names = [m.__name__.split('.')[-1] for m in strategy_modules]
    # Initialize performance tracking structure
    performance = {name: {'pnl':0.0,'trades':0,'win':0,'history':[]} for name in strategy_names}
    rows = []
    for idx in range(MIN_HISTORY, len(price_df) - LOOKAHEAD):
        window_df = price_df.iloc[:idx].copy()
        # For each strategy get signal
        signal_results = {}
        for mod, name in zip(strategy_modules, strategy_names):
            try:
                sig = mod.generate_signal(window_df)
            except Exception:
                sig = 'hold'
            signal_results[name] = sig
        # Compute next-bar return
        nxt_close = price_df['close'].iloc[idx + LOOKAHEAD]
        cur_close = price_df['close'].iloc[idx]
        ret = (nxt_close - cur_close) / cur_close if cur_close != 0 else 0.0
        # Strategy directional returns
        strat_ret = {}
        for name, sig in signal_results.items():
            mapper = SIG_RET_MAP.get(sig, lambda r: 0.0)
            r = mapper(ret)
            strat_ret[name] = r
        # Determine label (best strategy)
        label_name = max(strat_ret.items(), key=lambda x: x[1])[0]
        # Update performance with realized return
        for name, r in strat_ret.items():
            perf = performance[name]
            perf['history'].append(r)
            perf['trades'] += 1
            perf['win'] += int(r > 0)
            perf['pnl'] += r
            if len(perf['history']) > 200:
                perf['history'].pop(0)
        # Build features (regime placeholder 'sideways')
        feature_matrix = build_meta_features(window_df[['close','high','low']].copy(), performance, regime='sideways')
        flat = feature_matrix.flatten()
        row = {
            'timestamp': datetime.utcnow().isoformat(),
            'label_strategy': label_name
        }
        for i, val in enumerate(flat):
            row[f'f_{i}'] = val
        rows.append(row)
    if not rows:
        raise RuntimeError("No synthetic rows generated (insufficient price history).")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Synthetic meta dataset written: {OUTPUT_CSV} rows={len(out_df)} features_per_row={len(out_df.columns)-2}")

if __name__ == '__main__':
    main()
