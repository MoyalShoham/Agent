import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np

"""
Trade Log Cleaner & Realized PnL Backfiller

Functions:
- Infers side from position delta when missing
- Computes executed_qty, closed_qty, realized_pnl (net of fees), fees, cumulative_pnl
- Maintains per-symbol position state (size, entry_avg_price, group id)
- Generates trade_group_id (sequence id for each non-flat position lifecycle)
- Normalizes tiny floating artifacts & rounds numeric fields

Usage:
    python clean_trade_log.py --input futures_trades_log.csv --output futures_trades_log_cleaned.csv --fee-rate 0.0004

Environment variables (override CLI defaults):
    CLEAN_FEE_RATE          Fee rate per trade notional (e.g. 0.0004)
    CLEAN_PRICE_DECIMALS    Price rounding (default 8)
    CLEAN_QTY_DECIMALS      Quantity rounding (default 8)
    CLEAN_PNL_DECIMALS      PnL rounding (default 8)
    CLEAN_TOL               Zero tolerance (default 1e-9)

Outputs:
    New CSV with added columns:
        delta_pos, executed_qty, closed_qty, inferred_side, fees, gross_realized_pnl,
        realized_pnl (net), cumulative_realized_pnl, entry_avg_price_post, trade_group_id

Notes:
    - Assumes each row represents an intent already reflected in prev_pos/new_pos.
    - For flips (e.g., +x to -y), handles close then open within same row.
    - If realized_pnl already present and non-zero, script will by default recompute and overwrite.
"""

@dataclass
class PositionState:
    size: float = 0.0            # Signed position size
    entry_avg: float = 0.0       # Average entry price of current open position
    group_id: int = 0            # Current trade group id


def parse_args():
    p = argparse.ArgumentParser(description="Clean and enrich futures trade log.")
    p.add_argument('--input', '-i', required=True, help='Input trade log CSV')
    p.add_argument('--output', '-o', required=False, help='Output CSV (default: <input>_cleaned.csv)')
    p.add_argument('--fee-rate', type=float, default=float(os.getenv('CLEAN_FEE_RATE', '0.0004')), help='Fee rate (notional * fee_rate) per trade leg')
    p.add_argument('--overwrite', action='store_true', help='Overwrite input file after backing up to *.bak')
    p.add_argument('--strategy-col', default='strategy_id', help='Optional strategy id column name (if exists)')
    return p.parse_args()


def infer_side(delta: float, tol: float) -> str:
    if abs(delta) <= tol:
        return 'HOLD'
    return 'BUY' if delta > 0 else 'SELL'


def clean_numeric(val: float, tol: float) -> float:
    return 0.0 if abs(val) < tol else val


def process(df: pd.DataFrame, fee_rate: float, price_dec: int, qty_dec: int, pnl_dec: int, tol: float) -> pd.DataFrame:
    required_cols = ['timestamp', 'symbol', 'price', 'prev_pos', 'new_pos']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure consistent dtypes
    for col in ['price', 'prev_pos', 'new_pos']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Sort for stable processing
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # Initialize added columns
    df['delta_pos'] = df['new_pos'] - df['prev_pos']
    df['executed_qty'] = df['delta_pos'].abs()
    df['closed_qty'] = 0.0
    df['inferred_side'] = ''
    df['fees'] = 0.0
    df['gross_realized_pnl'] = 0.0
    df['realized_pnl'] = 0.0  # net
    df['cumulative_realized_pnl'] = 0.0
    df['entry_avg_price_post'] = 0.0
    df['trade_group_id'] = 0

    state: Dict[str, PositionState] = {}
    cumulative_realized = 0.0

    for idx, row in df.iterrows():
        symbol = row['symbol']
        price = float(row['price']) if not pd.isna(row['price']) else 0.0
        prev_pos = float(row['prev_pos'])
        new_pos = float(row['new_pos'])
        delta = new_pos - prev_pos

        if symbol not in state:
            state[symbol] = PositionState(size=0.0, entry_avg=0.0, group_id=0)
        st = state[symbol]

        # Sync state if mismatch (e.g., starting mid-stream)
        if abs(st.size - prev_pos) > tol:
            st.size = prev_pos
            # If we adopt an existing position without avg price, set entry_avg to trade price
            if abs(st.entry_avg) < tol and abs(st.size) > tol:
                st.entry_avg = price

        side_inferred = infer_side(delta, tol)

        closed_qty = 0.0
        gross_realized = 0.0

        # Determine if position is reduced or flipped
        if abs(prev_pos) > tol and np.sign(prev_pos) != np.sign(new_pos) and abs(new_pos) > tol:
            # Flip: close full prev, open remaining part
            closed_qty = abs(prev_pos)
            remaining_open_qty = abs(new_pos)
            # Realize PnL on closed leg
            gross_realized = (price - st.entry_avg) * np.sign(prev_pos) * closed_qty
            # New position entry avg is price for remaining part
            st.entry_avg = price
            st.size = new_pos
            # New trade group for flip (treat as fresh lifecycle)
            st.group_id += 1
        elif abs(prev_pos) > tol and abs(new_pos) < tol:
            # Closing to flat
            closed_qty = abs(prev_pos)
            gross_realized = (price - st.entry_avg) * np.sign(prev_pos) * closed_qty
            st.size = 0.0
            st.entry_avg = 0.0
        elif abs(prev_pos) > tol and abs(new_pos) > tol and abs(new_pos) < abs(prev_pos) - tol and np.sign(prev_pos) == np.sign(new_pos):
            # Partial reduction
            closed_qty = abs(prev_pos) - abs(new_pos)
            gross_realized = (price - st.entry_avg) * np.sign(prev_pos) * closed_qty
            st.size = new_pos
            # entry_avg unchanged
        elif abs(prev_pos) > tol and abs(new_pos) > tol and abs(new_pos) > abs(prev_pos) + tol and np.sign(prev_pos) == np.sign(new_pos):
            # Add to existing position: update weighted average
            added_qty = abs(new_pos) - abs(prev_pos)
            total_qty = abs(prev_pos) + added_qty
            if total_qty > tol:
                st.entry_avg = (st.entry_avg * abs(prev_pos) + price * added_qty) / total_qty
            st.size = new_pos
        elif abs(prev_pos) < tol and abs(new_pos) > tol:
            # Opening new position
            st.size = new_pos
            st.entry_avg = price
            st.group_id += 1
        else:
            # Flat / hold
            pass

        # Fees on total executed qty (delta) (use abs so both legs pay)
        executed_qty = abs(delta)
        notional = executed_qty * price
        fees = notional * fee_rate if executed_qty > tol else 0.0

        # Net realized PnL subtract fees proportional to closed fraction of executed (approx)
        net_realized = gross_realized - fees if closed_qty > tol else 0.0
        cumulative_realized += net_realized

        df.at[idx, 'inferred_side'] = side_inferred
        df.at[idx, 'closed_qty'] = closed_qty
        df.at[idx, 'gross_realized_pnl'] = gross_realized
        df.at[idx, 'realized_pnl'] = net_realized
        df.at[idx, 'fees'] = fees
        df.at[idx, 'cumulative_realized_pnl'] = cumulative_realized
        df.at[idx, 'entry_avg_price_post'] = st.entry_avg
        df.at[idx, 'trade_group_id'] = st.group_id

    # Normalization & rounding
    def rnd(series: pd.Series, dec: int):
        return series.apply(lambda x: 0.0 if abs(x) < tol else round(float(x), dec))

    df['price'] = rnd(df['price'], price_dec)
    for col in ['prev_pos', 'new_pos', 'delta_pos', 'executed_qty', 'closed_qty', 'entry_avg_price_post']:
        df[col] = rnd(df[col], qty_dec)
    for col in ['fees', 'gross_realized_pnl', 'realized_pnl', 'cumulative_realized_pnl']:
        df[col] = rnd(df[col], pnl_dec)

    # Fill any missing side column if exists in original schema
    if 'side' in df.columns:
        side_mask = (df['side'].isna()) | (df['side'] == '')
        df.loc[side_mask, 'side'] = df.loc[side_mask, 'inferred_side']
    else:
        df['side'] = df['inferred_side']

    return df


def main():
    args = parse_args()

    price_dec = int(os.getenv('CLEAN_PRICE_DECIMALS', '8'))
    qty_dec = int(os.getenv('CLEAN_QTY_DECIMALS', '8'))
    pnl_dec = int(os.getenv('CLEAN_PNL_DECIMALS', '8'))
    tol = float(os.getenv('CLEAN_TOL', '1e-9'))

    input_path = args.input
    output_path = args.output or f"{os.path.splitext(input_path)[0]}_cleaned.csv"

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path)

    cleaned = process(df, fee_rate=args.fee_rate, price_dec=price_dec, qty_dec=qty_dec, pnl_dec=pnl_dec, tol=tol)

    if args.overwrite:
        backup = input_path + '.bak'
        if not os.path.exists(backup):
            os.rename(input_path, backup)
        cleaned.to_csv(input_path, index=False)
        print(f"Cleaned & overwrote original. Backup at {backup}")
    else:
        cleaned.to_csv(output_path, index=False)
        print(f"Cleaned file written to {output_path}")

    # Simple summary
    total_realized = cleaned['realized_pnl'].sum()
    print(f"Total realized PnL (net): {total_realized}")
    churn = (cleaned['inferred_side'].isin(['BUY', 'SELL'])).sum()
    print(f"Trade count (executions): {churn}")


if __name__ == '__main__':
    main()
