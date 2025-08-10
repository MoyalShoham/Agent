"""
Download historical OHLCV data from Binance and save as CSV for backtesting.
Usage: python download_binance_ohlcv.py --symbol BTCUSDT --interval 1h --limit 1000 --outfile BTCUSDT_1h.csv
"""
import argparse
import pandas as pd
from trading_bot.utils.binance_client import BinanceClient

def fetch_ohlcv(symbol, interval, limit=1000):
    import time
    client = BinanceClient().client
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    # Sleep to avoid hitting rate limits (Binance allows ~1200 requests/minute, but be safe)
    time.sleep(1)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--interval', default='1h')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--outfile', required=True)
    args = parser.parse_args()
    df = fetch_ohlcv(args.symbol, args.interval, args.limit)
    df.to_csv(args.outfile, index=False)
    print(f"Saved {len(df)} rows to {args.outfile}")

if __name__ == '__main__':
    main()
