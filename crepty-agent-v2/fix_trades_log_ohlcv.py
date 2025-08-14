import pandas as pd

# Read the fixed CSV
input_path = 'futures_trades_log_fixed.csv'
df = pd.read_csv(input_path)

# Add missing OHLCV columns with default values if not present
for col in ['open', 'high', 'low']:
    if col not in df.columns:
        df[col] = df['close']
if 'volume' not in df.columns:
    df['volume'] = 1

# Optional: reorder columns to put OHLCV first
cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
df = df[cols]

df.to_csv(input_path, index=False)
print(f"Added missing OHLCV columns to {input_path}")
