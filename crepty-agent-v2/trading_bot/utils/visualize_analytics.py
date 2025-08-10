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

if __name__ == '__main__':
    plot_analytics_summary()
