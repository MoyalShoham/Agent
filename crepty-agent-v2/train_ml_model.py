#!/usr/bin/env python3
"""
ML Model Trainer - Train machine learning ensemble for trading signals.
Refactored to use EnhancedMLSignalGenerator (ensemble) with advanced features.
"""
import os, sys, pandas as pd, numpy as np
from datetime import datetime
from loguru import logger
from typing import Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Dependency check for 'ta'
try:
    import ta  # noqa: F401
    HAS_TA = True
except ImportError:  # pragma: no cover
    HAS_TA = False
    logger.warning("'ta' package not installed. Some features may be missing. Install with: pip install ta")

def _import_generator():
    """Import the appropriate signal generator class, returning an initialized instance."""
    try:
        from trading_bot.utils.ml_signals import EnhancedMLSignalGenerator
        logger.info("Using EnhancedMLSignalGenerator from ml_signals.py")
        return EnhancedMLSignalGenerator(model_path="enhanced_ml_model.pkl")
    except Exception as e1:  # pragma: no cover
        try:
            from trading_bot.utils.enhanced_ml_signals import MLSignalGenerator
            logger.info("Fallback to MLSignalGenerator from enhanced_ml_signals.py")
            return MLSignalGenerator(model_path="enhanced_ml_model.pkl")
        except Exception as e2:
            logger.error(f"Failed to import any ML signal generator: primary={e1} fallback={e2}")
            raise

def _load_historical_data() -> pd.DataFrame | None:
    data_files = ['futures_trades_log_fixed.csv', 'futures_trades_log.csv', 'BTCUSDT_1h.csv', 'trade_log_clean.csv', 'trade_log.csv']
    for file in data_files:
        if os.path.exists(file):
            try:
                logger.info(f"Loading data from {file}")
                df = pd.read_csv(file)
                if 'close' in df.columns:
                    logger.info(f"Using {file} for training")
                    return df
                else:
                    logger.info(f"Missing required 'close' column in {file}")
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
    return None

def _create_synthetic(rows: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.015, rows)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    dates = pd.date_range(start='2023-01-01', periods=len(prices), freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.008)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.008)) for p in prices],
        'volume': np.random.uniform(500, 5000, len(prices))
    })
    logger.info(f"Synthetic dataset generated ({len(df)} rows)")
    return df

def _prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['label'] = 0
    df.loc[df['future_return'] > 0.01, 'label'] = 1
    df.loc[df['future_return'] < -0.01, 'label'] = -1
    df = df.dropna().copy()
    return df

def _class_balance_ok(labels: pd.Series) -> bool:
    counts = labels.value_counts()
    if len(counts) < 2:
        logger.error("Not enough classes present for training")
        return False
    min_count = counts.min()
    if min_count < 20:
        logger.warning(f"Minority class too small for robust training (min={min_count})")
        return False
    return True

def train_ml_model_with_historical_data() -> bool:
    df = _load_historical_data()
    if df is None:
        logger.warning("No historical file found; using synthetic data")
        df = _create_synthetic()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df.set_index('timestamp', inplace=True)
    # Ensure numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['close']).copy()
    if len(df) < 150:
        logger.error("Insufficient data (<150 rows)")
        return False
    df = _prepare_labels(df)
    if len(df) < 120:
        logger.error("Insufficient data after labeling")
        return False
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    generator = _import_generator()
    # Use last N bars for feature richness
    price_df = df[['open','high','low','close','volume']].copy()
    features = getattr(generator, 'calculate_advanced_features', None)
    if features is None:
        logger.error("Generator missing calculate_advanced_features")
        return False
    feat_df = generator.calculate_advanced_features(price_df)
    if feat_df.empty:
        logger.error("Feature engineering produced empty dataframe")
        return False
    # Align
    common_index = feat_df.index.intersection(df.index)
    X = feat_df.loc[common_index]
    y = df.loc[common_index, 'label']
    # Drop rows with any NA
    mask = ~X.isna().any(axis=1)
    X, y = X[mask], y[mask]
    if not _class_balance_ok(y):
        logger.error("Class balance failed criteria")
        return False
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model_accuracies = {}
    feature_importance = {}
    # Train each model
    for name, model in generator.models.items():
        scaler = generator.scalers.get(name)
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            from sklearn.metrics import accuracy_score
            if name == 'xgboost':
                # Map labels: -1 -> 0, 0 -> 1, 1 -> 2
                y_train_xgb = y_train.map({-1: 0, 0: 1, 1: 2})
                y_test_xgb = y_test.map({-1: 0, 0: 1, 1: 2})
                model.fit(X_train_scaled, y_train_xgb)
                acc = accuracy_score(y_test_xgb, model.predict(X_test_scaled))
            else:
                model.fit(X_train_scaled, y_train)
                acc = accuracy_score(y_test, model.predict(X_test_scaled))
            model_accuracies[name] = acc
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = dict(sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])[:25])
            elif hasattr(model, 'coef_'):
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                imp = dict(sorted(zip(X.columns, np.abs(coef)), key=lambda x: -x[1])[:25])
                feature_importance[name] = imp
            logger.info(f"Trained {name} accuracy={acc:.3f}")
        except Exception as e:
            logger.error(f"Training failed for {name}: {e}")
    if not model_accuracies:
        logger.error("No models trained successfully")
        return False
    # Derive weights proportional to accuracy (min floor)
    total_acc = sum(model_accuracies.values())
    generator.model_weights = {m: (a / total_acc) for m, a in model_accuracies.items()}
    generator.feature_names = list(X.columns)
    generator.feature_importance = feature_importance
    generator.is_trained = True
    # Persist ensemble
    import joblib
    model_data = {
        'models': generator.models,
        'scalers': generator.scalers,
        'feature_names': generator.feature_names,
        'model_weights': generator.model_weights,
        'feature_importance': generator.feature_importance,
        'is_trained': True,
        'accuracies': model_accuracies,
        'training_date': datetime.now().isoformat(),
        'rows': len(X)
    }
    out_path = 'enhanced_ml_model.pkl'
    joblib.dump(model_data, out_path)
    logger.success(f"Ensemble model saved to {out_path}")
    logger.info(f"Model weights: {generator.model_weights}")
    # Quick validation signal
    sample_tail = price_df.tail(120)
    try:
        result = generator.generate_enhanced_signal('VALIDATION', sample_tail)
        logger.info(f"Validation signal: {result.get('enhanced_signal')} conf={result.get('confidence'):.2f}")
    except Exception as e:
        logger.warning(f"Validation signal failed: {e}")
    return True

def main():
    logger.info("🤖 Starting ML Ensemble Training")
    ok = train_ml_model_with_historical_data()
    if ok:
        logger.info("✅ Training complete")
        print("\n============================================\n Ensemble Training Complete \n============================================")
    else:
        logger.error("❌ Training failed")
        print("\n============================================\n Training Failed \n============================================")

if __name__ == '__main__':
    main()
