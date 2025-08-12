"""
Meta-Learning Layer for Strategy Selection
Uses recent features and strategy performance to select/weight strategies in real time.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle

class MetaLearner:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load(model_path)

    def fit(self, X, y):
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            return None
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            return None
        return self.model.predict_proba(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

# Example feature engineering for meta-learner
# X: [regime, volatility, recent_pnl, win_rate, sharpe, max_dd, correlations...]
# y: best strategy index (or signal)

def _compute_sharpe(returns):
    if len(returns) < 5:
        return 0.0
    r = np.array(returns)
    if r.std() == 0:
        return 0.0
    return (r.mean() / (r.std() + 1e-9)) * np.sqrt(252)


def _compute_max_drawdown(equity_curve):
    if not len(equity_curve):
        return 0.0
    arr = np.array(equity_curve)
    peak = arr[0]
    max_dd = 0
    for x in arr:
        if x > peak:
            peak = x
        dd = (peak - x)
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _safe_number(x, default=0.0):
    if x is None:
        return default
    try:
        if np.isnan(x):
            return default
    except Exception:
        pass
    if np.isinf(x):
        return default
    return float(x)


def _exp_decay_mean(values, alpha: float = 0.2):
    if not values:
        return 0.0
    acc = 0.0
    wsum = 0.0
    for i, v in enumerate(reversed(values)):
        w = (1 - alpha) ** i
        acc += w * v
        wsum += w
    return acc / wsum if wsum else 0.0


def build_meta_features(df, strategy_perf, regime, research: dict | None = None):
    """Build a (n_strategies x n_features) feature matrix with NaN-safe values.

    Added features:
    - short_vol / long_vol ratio
    - decayed pnl mean
    - recent win rate (last 10) vs long win rate (all) ratio
    - volatility of pnl history (stdev) and its inverse as stability
    - drawdown ratio (max_dd / cumulative positive pnl + 1e-9)
    Total feature count increase.
    """
    features = []
    # Volatility (20-period) – guard against NaN
    if df is not None and isinstance(df, (pd.DataFrame, dict)) and 'close' in df:
        try:
            closes = df['close'] if isinstance(df, pd.DataFrame) else pd.Series(df['close'])
            if len(closes) > 50:
                ret = closes.pct_change()
                vol_val = ret.rolling(20).std().iloc[-1]
                vol_short = ret.rolling(10).std().iloc[-1]
                vol_long = ret.rolling(50).std().iloc[-1]
                vol_ratio = (vol_short / (vol_long + 1e-9)) if vol_long else 0.0
            else:
                vol_val = 0.0; vol_ratio = 0.0
        except Exception:
            vol_val = 0.0; vol_ratio = 0.0
    else:
        vol_val = 0.0; vol_ratio = 0.0
    vol_val = _safe_number(vol_val)
    vol_ratio = _safe_number(vol_ratio)

    # Research features (optional)
    funding_z = _safe_number(research.get('funding_z', 0.0) if research else 0.0)
    oi_change = _safe_number(research.get('open_interest_change_pct', 0.0) if research else 0.0)
    imbalance = _safe_number(research.get('orderbook_imbalance', 0.0) if research else 0.0)

    strategy_names = list(strategy_perf.keys())
    returns_matrix = []

    for strat in strategy_names:
        perf = strategy_perf[strat]
        hist = perf.get('history', [])[-100:] if perf else []
        hist = [h for h in hist if h is not None]
        sharpe = _safe_number(_compute_sharpe(hist)) if hist else 0.0
        max_dd = _safe_number(_compute_max_drawdown(np.cumsum(hist))) if hist else 0.0
        trades = perf.get('trades', 0)
        wins = perf.get('win', 0)
        win_rate = (wins / trades) if trades else 0.0
        win_rate = _safe_number(win_rate)
        avg_pnl = _safe_number(np.mean(hist)) if hist else 0.0
        # New metrics
        decayed_pnl = _safe_number(_exp_decay_mean(hist, 0.25))
        recent_hist = hist[-10:]
        wins_recent = sum(1 for x in recent_hist if x > 0)
        win_rate_recent = wins_recent / len(recent_hist) if recent_hist else 0.0
        win_rate_ratio = _safe_number(win_rate_recent / (win_rate + 1e-9)) if win_rate else 0.0
        pnl_std = _safe_number(np.std(hist)) if len(hist) > 2 else 0.0
        stability = _safe_number(1.0 / (1.0 + pnl_std))
        drawdown_ratio = _safe_number(max_dd / (abs(sum(x for x in hist if x > 0)) + 1e-9)) if hist else 0.0

        features.append([
            {'bull': 0, 'bear': 1, 'sideways': 2}.get(regime, 2),
            vol_val,
            avg_pnl,
            win_rate,
            sharpe,
            max_dd,
            funding_z,
            oi_change,
            imbalance,
            vol_ratio,
            decayed_pnl,
            win_rate_recent,
            win_rate_ratio,
            pnl_std,
            stability,
            drawdown_ratio
        ])
        returns_matrix.append(hist if hist else [0.0])

    # Correlation features (mean correlation per strategy vs others)
    if returns_matrix:
        max_len = max(len(r) for r in returns_matrix)
        norm_returns = []
        for r in returns_matrix:
            if len(r) < max_len:
                r = list(r) + [0.0] * (max_len - len(r))
            norm_returns.append(r)
        if len(norm_returns) >= 2 and max_len > 1:
            try:
                corr = np.corrcoef(norm_returns)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                for i in range(len(features)):
                    others = [corr[i][j] for j in range(len(features)) if j != i]
                    mean_corr = np.mean(others) if others else 0.0
                    features[i].append(_safe_number(mean_corr))
            except Exception:
                for i in range(len(features)):
                    features[i].append(0.0)
        else:
            for i in range(len(features)):
                features[i].append(0.0)

    if not features:
        return np.empty((0, 0))

    feat_array = np.array(features, dtype=float)
    feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_array
