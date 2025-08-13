"""Train MetaLearner from live collected meta_samples.csv.

Usage (PowerShell):
  python train_meta_from_samples.py

Environment variables (optional):
  META_SAMPLE_FILE=meta_samples.csv   # Source samples file
  META_MIN_ROWS=200                   # Minimum rows required
  META_REQUIRE_META_USED=1            # 1 => only rows where meta_used==1, 0 => all rows
  META_USE_DECISION_AS_LABEL=0        # 1 => label = decision (buy/sell/hold), 0 => label = selected_strategy
  META_STRATEGY_MIN_FREQ=5            # Min occurrences to keep a strategy (else dropped)
  META_BALANCE_CLASSES=1              # Simple random downsample for large classes
  META_TEST_SIZE=0.15                 # Fraction for validation split
  META_OUTPUT_PATH=trading_bot/utils/meta_learner_model.pkl

The script:
  * Loads meta_samples
  * Filters / prepares features and labels
  * Trains GradientBoostingClassifier (same as MetaLearner)
  * Saves model dict with feature_dim and strategies (or decisions list)
  * Writes a small JSON metadata file for reproducibility
"""
from __future__ import annotations
import os
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import joblib

META_DEFAULT_FILE = 'meta_samples.csv'

@dataclass
class Config:
    sample_file: str = os.getenv('META_SAMPLE_FILE', META_DEFAULT_FILE)
    min_rows: int = int(os.getenv('META_MIN_ROWS', '200'))
    require_meta_used: bool = bool(int(os.getenv('META_REQUIRE_META_USED', '1')))
    use_decision_label: bool = bool(int(os.getenv('META_USE_DECISION_AS_LABEL', '0')))
    strategy_min_freq: int = int(os.getenv('META_STRATEGY_MIN_FREQ', '5'))
    balance_classes: bool = bool(int(os.getenv('META_BALANCE_CLASSES', '1')))
    test_size: float = float(os.getenv('META_TEST_SIZE', '0.15'))
    output_path: str = os.getenv('META_OUTPUT_PATH', 'trading_bot/utils/meta_learner_model.pkl')


def load_samples(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.sample_file):
        raise FileNotFoundError(f"Sample file not found: {cfg.sample_file}")
    df = pd.read_csv(cfg.sample_file)
    # Basic sanity
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    if not feature_cols:
        raise ValueError("No feature columns (f_*) found in samples file.")
    if cfg.require_meta_used and 'meta_used' in df.columns:
        df = df[df['meta_used'] == 1]
    df = df.dropna(subset=feature_cols)
    if len(df) < cfg.min_rows:
        raise ValueError(f"Not enough rows after filtering ({len(df)} < {cfg.min_rows}).")
    return df


def prepare_labels(df: pd.DataFrame, cfg: Config):
    if cfg.use_decision_label:
        label_col = 'decision'
        if 'decision' not in df.columns:
            raise ValueError("'decision' column not in samples for decision labeling.")
    else:
        label_col = 'selected_strategy'
        if 'selected_strategy' not in df.columns:
            raise ValueError("'selected_strategy' column missing in samples.")
    labels = df[label_col].fillna('')
    # Filter low frequency strategies (if using strategy labels)
    if not cfg.use_decision_label:
        freq = labels.value_counts()
        keep = set(freq[freq >= cfg.strategy_min_freq].index)
        mask = labels.isin(keep)
        df = df[mask]
        labels = labels[mask]
    # Rebuild after filtering
    labels = labels.reset_index(drop=True)
    df = df.reset_index(drop=True)
    # Map to indices
    unique_labels = sorted(labels.unique())
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    y = labels.map(label_to_idx).astype(int).values
    return df, y, unique_labels


def balance(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    # Simple down-sampling of majority classes
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    keep_indices = []
    for cls, cnt in zip(unique, counts):
        idx = np.where(y == cls)[0]
        if cnt > min_count * 1.2:  # allow slight tolerance
            idx = rng.choice(idx, size=min_count, replace=False)
        keep_indices.append(idx)
    keep_indices = np.concatenate(keep_indices)
    rng.shuffle(keep_indices)
    return X[keep_indices], y[keep_indices]


def main():
    cfg = Config()
    print(f"[CONFIG] {cfg}")
    try:
        df = load_samples(cfg)
    except Exception as e:
        print(f"[ERROR] Load samples: {e}")
        return

    feature_cols = [c for c in df.columns if c.startswith('f_')]
    try:
        df, y, label_list = prepare_labels(df, cfg)
    except Exception as e:
        print(f"[ERROR] Prepare labels: {e}")
        return

    feature_cols = [c for c in df.columns if c.startswith('f_')]
    X = df[feature_cols].values.astype(float)
    feature_dim = X.shape[1]
    print(f"[INFO] Samples={X.shape[0]} feature_dim={feature_dim} classes={len(label_list)}")

    if len(label_list) < 2:
        print("[ABORT] Need at least 2 classes.")
        return

    rng = np.random.default_rng(42)
    if cfg.balance_classes:
        X, y = balance(X, y, rng)
        print(f"[INFO] After balancing: samples={X.shape[0]}")

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=42)
    except ValueError:
        # Fallback if stratify fails
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_size, random_state=42)

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"[METRICS] BalancedAccuracy={bal_acc:.4f}")
    try:
        print(classification_report(y_test, y_pred))
    except Exception:
        pass

    # Save model bundle
    out_dir = os.path.dirname(cfg.output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    bundle = {
        'model': model,
        'strategies': label_list,  # could be decisions if use_decision_label
        'feature_dim': feature_dim,
        'label_type': 'decision' if cfg.use_decision_label else 'strategy'
    }
    joblib.dump(bundle, cfg.output_path)
    print(f"[SAVED] Model -> {cfg.output_path}")

    # Hash file for reproducibility
    try:
        with open(cfg.output_path, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        h = None

    meta = {
        'saved_at': datetime.utcnow().isoformat(),
        'hash': h,
        'samples': int(X.shape[0]),
        'feature_dim': feature_dim,
        'classes': label_list,
        'balanced_accuracy': bal_acc,
        'config': {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    }
    meta_path = cfg.output_path + '.meta.json'
    try:
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"[META] {meta_path}")
    except Exception as e:
        print(f"[WARN] Could not write meta: {e}")

if __name__ == '__main__':
    main()
