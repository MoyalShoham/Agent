"""
Script to train the MetaLearner using synthetic or real data
Usage: python train_meta_learner.py
Now adapts feature_dim to current build_meta_features output to avoid runtime mismatch.
"""
import pandas as pd
from trading_bot.utils.meta_learner import MetaLearner, build_meta_features
import joblib
import os
from trading_bot.strategies.strategy_manager import StrategyManager
import numpy as np
SYN_PATH = 'synthetic_meta_training.csv'


def compute_live_feature_dim():
    """Compute current live flattened feature dimension based on active strategies."""
    try:
        sm = StrategyManager()
        # create minimal dummy df (need enough rows for volatility calc) using BTC if available
        dummy_path = 'BTCUSDT_1h.csv'
        if os.path.exists(dummy_path):
            df = pd.read_csv(dummy_path).tail(120)
            if 'close' not in df and 'Close' in df:
                df.rename(columns={'Close': 'close'}, inplace=True)
        else:
            # fabricate simple increasing close series
            df = pd.DataFrame({'close': np.linspace(10000, 10100, 120)})
        perf = {s: {'pnl':0,'trades':0,'win':0,'history':[]} for s in sm.performance.keys()}
        feat = build_meta_features(df, perf, 'sideways')
        if feat.size == 0:
            return 0
        return feat.flatten().shape[0]
    except Exception:
        return 0


def main():
    model_path = os.path.join('trading_bot', 'utils', 'meta_learner_model.pkl')
    if not os.path.exists(SYN_PATH):
        print(f"Synthetic dataset {SYN_PATH} not found; aborting meta learner training.")
        return
    print(f"Using synthetic dataset {SYN_PATH} for meta learner training.")
    syn = pd.read_csv(SYN_PATH)
    y_syn = syn['label_strategy'].tolist()
    feature_cols = [c for c in syn.columns if c.startswith('f_')]
    X_syn = syn[feature_cols].values
    strategy_list = sorted(list(set(y_syn)))
    name_to_idx = {s: i for i, s in enumerate(strategy_list)}
    y_idx = [name_to_idx.get(n, 0) for n in y_syn]

    live_feature_dim = compute_live_feature_dim()
    print(f"Live (current code) flattened feature_dim={live_feature_dim}")

    if live_feature_dim > 0 and X_syn.shape[1] != live_feature_dim:
        print(f"Adapting synthetic feature matrix from {X_syn.shape[1]} -> {live_feature_dim}")
        if X_syn.shape[1] > live_feature_dim:
            X_syn = X_syn[:, :live_feature_dim]
        else:
            pad_width = live_feature_dim - X_syn.shape[1]
            X_syn = np.hstack([X_syn, np.zeros((X_syn.shape[0], pad_width))])

    if len(set(y_idx)) >= 2 and X_syn.shape[0] > 10:
        meta_learner = MetaLearner()
        meta_learner.fit(X_syn, y_idx)
        joblib.dump({
            'model': meta_learner.model,
            'strategies': strategy_list,
            'feature_dim': X_syn.shape[1],
            'live_feature_dim': live_feature_dim
        }, model_path)
        print(f"MetaLearner trained -> {model_path}; stored_feature_dim={X_syn.shape[1]} live_feature_dim={live_feature_dim}")
    else:
        print("Synthetic dataset lacks class diversity or size; aborting meta learner training.")

if __name__ == '__main__':
    main()
