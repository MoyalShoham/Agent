import pandas as pd
import matplotlib.pyplot as plt


def plot_analytics_summary(csv_path='analytics_summary.csv'):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(df['timestamp'], df['total_usdt'], label='Total USDT', color='tab:blue')
    ax1.set_ylabel('Portfolio Value (USDT)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(df['timestamp'], df['trades_this_hour'], label='Trades/Hour', color='tab:orange', linestyle='--')
    ax2.set_ylabel('Trades per Hour', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    plt.title('Portfolio Value and Trades per Hour Over Time')
    fig.tight_layout()
    plt.show()

def plot_backtest_results(results_dir='backtests'):
    """
    Compare multiple backtest results (CSV files) in a directory.
    Each CSV should have columns: timestamp, portfolio_value, trades, win_rate, etc.
    """
    import os
    if not os.path.exists(results_dir):
        print(f"No backtest results directory: {results_dir}")
        return
    files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not files:
        print("No backtest result CSVs found.")
        return
    plt.figure(figsize=(14,7))
    for f in files:
        df = pd.read_csv(os.path.join(results_dir, f))
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            plt.plot(df['timestamp'], df['portfolio_value'], label=f)
    plt.title('Backtest Portfolio Value Comparison')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value (USDT)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("1. Portfolio Analytics Summary\n2. Compare Backtest Results")
    choice = input("Select plot type (1/2): ")
    if choice == '1':
        plot_analytics_summary()
    elif choice == '2':
        plot_backtest_results()
    else:
        print("Invalid choice.")
