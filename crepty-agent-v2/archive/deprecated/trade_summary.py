import argparse
import pandas as pd
import numpy as np

"""
Per-symbol performance & churn summary.

Usage:
  python trade_summary.py -i futures_trades_log_cleaned.csv

Outputs sorted table (net_per_trade ascending) and a top / bottom snapshot.
"""

def parse_args():
    p = argparse.ArgumentParser(description="Per-symbol trade performance summary")
    p.add_argument('-i','--input', required=True, help='Cleaned trade log CSV (with realized_pnl, fees, inferred_side)')
    p.add_argument('--limit', type=int, default=30, help='Max symbols to display (0 = all)')
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    needed = {'symbol','realized_pnl','fees','inferred_side','gross_realized_pnl','executed_qty'}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Count executions (BUY/SELL) per symbol
    exec_mask = df['inferred_side'].isin(['BUY','SELL'])
    grp = df[exec_mask].groupby('symbol').agg(
        trades=('inferred_side','count'),
        net_pnl=('realized_pnl','sum'),
        gross_pnl=('gross_realized_pnl','sum'),
        fees=('fees','sum'),
        avg_exec_qty=('executed_qty','mean'),
        median_exec_qty=('executed_qty','median'),
    )
    if grp.empty:
        print('No execution rows found.')
        return

    grp['net_per_trade'] = grp['net_pnl'] / grp['trades'].replace(0,np.nan)
    grp['fee_ratio_gross_abs'] = grp['fees'].abs() / (grp['gross_pnl'].abs() + 1e-9)
    grp['fee_ratio_net_abs'] = grp['fees'].abs() / (grp['net_pnl'].abs() + 1e-9)
    grp['gross_minus_fees'] = grp['gross_pnl'] - grp['fees']

    # Sort by net_per_trade ascending (worst first)
    summary = grp.sort_values('net_per_trade')

    if args.limit > 0:
        summary_to_show = summary.head(args.limit)
    else:
        summary_to_show = summary

    pd.set_option('display.width', 140)
    pd.set_option('display.max_rows', 500)
    print('\nPer-symbol summary (worst net_per_trade first):')
    print(summary_to_show.round(6))

    # Show best 5 as well
    print('\nTop 5 (best net_per_trade):')
    print(summary.sort_values('net_per_trade', ascending=False).head(5).round(6))

    # Key aggregate metrics
    total_trades = int(summary['trades'].sum())
    total_net = summary['net_pnl'].sum()
    total_fees = summary['fees'].sum()
    total_gross = summary['gross_pnl'].sum()
    print('\nAggregate:')
    print(f"Gross PnL: {total_gross:.6f}  Fees: {total_fees:.6f}  Net: {total_net:.6f}  Net/Gross: {(total_net/(total_gross+1e-9)):.2%}")
    print(f"Total trades: {total_trades}")

    # Recommendation hints
    worst = summary.head(5)
    candidates_disable = worst[(worst['net_per_trade'] < 0) & (worst['fee_ratio_gross_abs'] > 0.3)].index.tolist()
    if candidates_disable:
        print('\nCandidates to throttle/disable (neg net_per_trade & high fee drag):', ', '.join(candidates_disable))
    else:
        print('\nNo obvious disable candidates by current heuristic.')

if __name__ == '__main__':
    main()
