#!/usr/bin/env python3
"""Feature engineering for order book snapshots -> 5m aggregated microstructure features.

Input: data/orderbook/{symbol}_raw.csv produced by orderbook_collector.py
Output: data/orderbook/{symbol}_5m_features.csv

Features engineered per 5m bar:
  mid_price_{open,high,low,close}
  mid_return_5m
  spread_mean, spread_bps_mean
  depth_imbalance_mean, depth_imbalance_last
  orderflow_imbalance (based on last trade side counts)
  buy_trade_ratio
  volatility_mid (std of mid inside bar)
  microprice_premium_mean (microprice - mid)/mid using lvl1 queue sizes

These can be merged with candle features by timestamp.
"""
import os
import math
import argparse
import logging
import pandas as pd
from datetime import timezone
from typing import List

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger("orderbook_features")


def load_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise ValueError('timestamp column missing')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Derive microprice and trade side flags
    if {'best_bid','best_ask','bid_sz_1','ask_sz_1'}.issubset(df.columns):
        denom = (df['bid_sz_1'] + df['ask_sz_1']).replace(0, pd.NA)
        df['microprice'] = (df['best_ask'] * df['bid_sz_1'] + df['best_bid'] * df['ask_sz_1']) / denom
        df['microprice_premium'] = (df['microprice'] - df['mid_price']) / df['mid_price']
    else:
        df['microprice_premium'] = 0.0
    df['is_buy_trade'] = (df['last_trade_side'] == 'BUY').astype(int)
    df['is_sell_trade'] = (df['last_trade_side'] == 'SELL').astype(int)

    df.set_index('timestamp', inplace=True)
    # 5m resample (use "5min" to avoid future warning)
    agg = df.resample('5min').agg({
        'mid_price': ['first','max','min','last','std'],
        'spread': 'mean',
        'spread_bps': 'mean',
        'depth_imbalance': ['mean','last'],
        'is_buy_trade': 'sum',
        'is_sell_trade': 'sum',
        'microprice_premium': 'mean'
    })
    # Flatten multi-index columns
    flat_cols = []
    for tup in agg.columns:
        if isinstance(tup, tuple):
            base = '_'.join([c for c in tup if c])
        else:
            base = tup
        base = base.replace('mid_price_std','volatility_mid')
        flat_cols.append(base)
    agg.columns = flat_cols
    # Rename to canonical names
    rename_map = {
        'mid_price_first': 'mid_price_open',
        'mid_price_max': 'mid_price_high',
        'mid_price_min': 'mid_price_low',
        'mid_price_last': 'mid_price_close',
        'spread_mean': 'spread_mean',
        'spread_bps_mean': 'spread_bps_mean',
        'depth_imbalance_mean': 'depth_imbalance_mean',
        'depth_imbalance_last': 'depth_imbalance_last',
        'is_buy_trade_sum': 'buy_trades',
        'is_sell_trade_sum': 'sell_trades',
        'microprice_premium_mean': 'microprice_premium_mean'
    }
    agg.rename(columns=rename_map, inplace=True)
    # Defensive: ensure required columns exist
    required_after = ['mid_price_close','mid_price_open']
    for col in required_after:
        if col not in agg.columns:
            logger.warning(f"Missing expected column {col}; attempting fallback")
    if 'mid_price_close' not in agg.columns:
        # Fallback: if we still have mid_price_last from unexpected naming
        for cand in ['mid_price_last','mid_price']:  # search original possibilities
            if cand in agg.columns:
                agg['mid_price_close'] = agg[cand]
                break
    # Derived metrics (guard for presence)
    if 'mid_price_close' in agg.columns:
        agg['mid_return_5m'] = agg['mid_price_close'].pct_change()
    else:
        agg['mid_return_5m'] = 0.0
    if {'buy_trades','sell_trades'}.issubset(agg.columns):
        denom_flow = (agg['buy_trades'] + agg['sell_trades']).replace(0, pd.NA)
        agg['orderflow_imbalance'] = (agg['buy_trades'] - agg['sell_trades']) / denom_flow
        agg['buy_trade_ratio'] = agg['buy_trades'] / denom_flow
    else:
        agg['orderflow_imbalance'] = 0.0
        agg['buy_trade_ratio'] = 0.0
    logger.info(f"Aggregated bars: {len(agg)} columns: {len(agg.columns)}")
    return agg.dropna(how='all')


def save_features(df: pd.DataFrame, out_path: str):
    df.reset_index().to_csv(out_path, index=False)
    logger.info(f"Saved features -> {out_path} rows={len(df)}")


def main(symbol: str, data_dir: str):
    raw_path = os.path.join(data_dir, f"{symbol}_raw.csv")
    feat_path = os.path.join(data_dir, f"{symbol}_5m_features.csv")
    raw = load_raw(raw_path)
    feats = compute_features(raw)
    save_features(feats, feat_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--data-dir', default='data/orderbook')
    args = ap.parse_args()
    main(args.symbol.upper(), args.data_dir)
