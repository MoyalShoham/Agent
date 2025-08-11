"""
Script to train the MetaLearner using trade_log.csv
Usage: python train_meta_learner.py
"""
import pandas as pd
from trading_bot.utils.meta_learner import MetaLearner, build_meta_features
import joblib
import os
from trading_bot.strategies.strategy_manager import StrategyManager
import numpy as np
SYN_PATH = 'synthetic_meta_training.csv'

def load_trade_log(csv_path):
    df = pd.read_csv(csv_path)
    return df

def prepare_training_data(df, strategy_list):
    X, y = [], []
    # Initialize empty perf structure per strategy (could be enhanced with rolling PnL)
    base_perf = {s: {'pnl': 0, 'trades': 0, 'win': 0, 'history': []} for s in strategy_list}
    for i, row in df.iterrows():
        symbol = row.get('symbol')
        price_df = None
        price_csv = f"{symbol}_1h.csv"
        if symbol and os.path.exists(price_csv):
            try:
                price_df = pd.read_csv(price_csv)
            except Exception:
                price_df = None
        features_matrix = build_meta_features(price_df, base_perf, row.get('regime', 'sideways'))
        if features_matrix.size == 0:
            continue
        # Flatten per-strategy matrix; use same order each time
        flat = features_matrix.flatten()
        # Record y as strategy index actually used
        strat_name = row.get('strategy')
        if strat_name not in strategy_list:
            continue
        y.append(strategy_list.index(strat_name))
        X.append(flat)
    if not X:
        return np.empty((0,)), []
    # Pad rows to equal length (should already be equal if same number strategies each row)
    max_len = max(len(r) for r in X)
    X = [list(r) + [0.0]*(max_len - len(r)) for r in X]
    return np.array(X, dtype=float), y

def main():
    csv_path = 'trade_log_clean_fixed_with_strategy.csv'
    model_path = os.path.join('trading_bot', 'utils', 'meta_learner_model.pkl')
    if not os.path.exists(csv_path):
        print(f"Training log {csv_path} not found")
        return
    df = load_trade_log(csv_path)
    sm = StrategyManager()
    strategy_list = list(sm.performance.keys())
    X, y = prepare_training_data(df, strategy_list)
    if X.size == 0 or len(y) == 0:
        print("No training data assembled; aborting meta learner training.")
        return
    if len(set(y)) < 2:
        print(f"Insufficient class diversity in y (only {len(set(y))} class). Need at least 2. Aborting training.")
        return
    meta_learner = MetaLearner()
    meta_learner.fit(X, y)
    joblib.dump({'model': meta_learner.model, 'strategies': strategy_list, 'feature_dim': X.shape[1]}, model_path)
    print(f"MetaLearner trained and saved to {model_path}; feature_dim={X.shape[1]}")
    if os.path.exists(SYN_PATH):
        print(f"Using synthetic dataset {SYN_PATH} for meta learner training.")
        syn = pd.read_csv(SYN_PATH)
        y = syn['label_strategy'].tolist()
        feature_cols = [c for c in syn.columns if c.startswith('f_')]
        X = syn[feature_cols].values
        # Map strategy names to indices using current strategy_list ordering
        name_to_idx = {s:i for i,s in enumerate(strategy_list)}
        y_idx = [name_to_idx.get(n, 0) for n in y]
        # Guard: Need 2 classes
        if len(set(y_idx)) < 2:
            print("Synthetic dataset still single-class; aborting.")
            return
        meta_learner = MetaLearner()
        meta_learner.fit(X, y_idx)
        joblib.dump({'model': meta_learner.model, 'strategies': strategy_list, 'feature_dim': X.shape[1]}, model_path)
        print(f"MetaLearner trained from synthetic dataset -> {model_path}; feature_dim={X.shape[1]}")
        return

if __name__ == '__main__':
    main()
