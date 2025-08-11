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
        self.model = GradientBoostingClassifier(n_estimators=50, max_depth=3)
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
# X: [regime, volatility, recent_pnl, win_rate, ...]
# y: best strategy index (or signal)
def build_meta_features(df, strategy_perf, regime):
    features = []
    if df is not None and 'close' in df and len(df['close']) > 20:
        vol = df['close'].pct_change().rolling(20).std().iloc[-1]
    else:
        vol = 0
    for strat, perf in strategy_perf.items():
        features.append([
            {'bull':0, 'bear':1, 'sideways':2}.get(regime,2),
            vol,
            np.mean(perf['history'][-20:]) if perf['history'] else 0,
            perf['win']/perf['trades'] if perf['trades'] else 0
        ])
    return np.array(features)
