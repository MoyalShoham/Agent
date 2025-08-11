"""
Script to train the MetaLearner using trade_log.csv
Usage: python train_meta_learner.py
"""
import pandas as pd
from trading_bot.utils.meta_learner import MetaLearner, build_meta_features
import joblib
import os

def load_trade_log(csv_path):
    df = pd.read_csv(csv_path)
    # Expect columns: timestamp, symbol, strategy, signal, price, qty, pnl, regime, volatility, win, ...
    return df

def prepare_training_data(df, strategy_list):
    X, y = [], []
    for i, row in df.iterrows():
        # Build features for each row
        perf = {s: {'pnl': 0, 'trades': 0, 'win': 0, 'history': []} for s in strategy_list}
        # Optionally, fill perf with rolling stats up to this row
        symbol = row['symbol']
        price_df = None
        price_csv = f"{symbol}_1h.csv"
        if os.path.exists(price_csv):
            try:
                price_df = pd.read_csv(price_csv)
            except Exception:
                price_df = None
        features = build_meta_features(price_df, perf, row.get('regime', 'sideways'))
        X.append(features.flatten())
        # y: index of strategy used
        strat_idx = strategy_list.index(row['strategy']) if row['strategy'] in strategy_list else 0
        y.append(strat_idx)
    return X, y

def main():
    # Use the cleaned trade log for ML/analytics
    csv_path = 'trade_log_clean_fixed_with_strategy.csv'
    model_path = os.path.join('trading_bot', 'utils', 'meta_learner_model.pkl')
    df = load_trade_log(csv_path)
    strategy_list = sorted(df['strategy'].unique())
    X, y = prepare_training_data(df, strategy_list)
    meta_learner = MetaLearner()
    meta_learner.fit(X, y)
    joblib.dump({'model': meta_learner.model, 'strategies': strategy_list}, model_path)
    print(f"MetaLearner trained and saved to {model_path}")

if __name__ == '__main__':
    main()
