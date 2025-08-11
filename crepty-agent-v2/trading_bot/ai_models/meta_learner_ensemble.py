"""
Meta-Learner Ensemble: Combines multiple ML models (LSTM, XGBoost, etc.) and strategy signals for adaptive trading.
"""


import asyncio
import logging
from collections import Counter

class MetaLearnerEnsemble:
    """
    Advanced Meta-Learner Ensemble for adaptive trading:
    - Supports weighted voting, async prediction, model performance tracking, and robust error handling.
    - Designed for production-grade, real-time trading systems.
    """
    def __init__(self, models=None, model_weights=None, logger=None):
        self.models = models or []
        self.model_weights = model_weights or [1.0] * len(self.models)
        self.performance = [0.0] * len(self.models)  # Track model accuracy or reward
        self.logger = logger or logging.getLogger("MetaLearnerEnsemble")

    def add_model(self, model, weight=1.0):
        self.models.append(model)
        self.model_weights.append(weight)
        self.performance.append(0.0)
        self.logger.info(f"Model {model.__class__.__name__} added with weight {weight}")

    def update_performance(self, model_idx, reward):
        # Update model performance (e.g., after each trade)
        self.performance[model_idx] += reward
        self.logger.debug(f"Updated performance for model {model_idx}: {self.performance[model_idx]}")

    def predict(self, features):
        # Aggregate predictions from all models with weights
        votes = []
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(features)
                votes.extend([pred] * int(self.model_weights[i]))
            except Exception as e:
                self.logger.error(f"Model {i} prediction error: {e}")
        if not votes:
            self.logger.warning("No valid predictions from ensemble models.")
            return None
        # Weighted majority vote
        vote_counts = Counter(votes)
        best_vote = vote_counts.most_common(1)[0][0]
        self.logger.info(f"Meta-ensemble prediction: {best_vote} (votes: {dict(vote_counts)})")
        return best_vote

    async def predict_async(self, features):
        # Async version for models with async predict
        votes = []
        tasks = []
        for i, model in enumerate(self.models):
            predict_fn = getattr(model, "predict_async", None)
            if callable(predict_fn):
                tasks.append(self._predict_model_async(model, features, i))
            else:
                try:
                    pred = model.predict(features)
                    votes.extend([pred] * int(self.model_weights[i]))
                except Exception as e:
                    self.logger.error(f"Model {i} prediction error: {e}")
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    self.logger.error(f"Async model {i} prediction error: {res}")
                elif res is not None:
                    votes.extend([res] * int(self.model_weights[i]))
        if not votes:
            self.logger.warning("No valid predictions from ensemble models (async).")
            return None
        vote_counts = Counter(votes)
        best_vote = vote_counts.most_common(1)[0][0]
        self.logger.info(f"Meta-ensemble async prediction: {best_vote} (votes: {dict(vote_counts)})")
        return best_vote

    async def _predict_model_async(self, model, features, idx):
        try:
            return await model.predict_async(features)
        except Exception as e:
            self.logger.error(f"Async model {idx} prediction error: {e}")
            return None
