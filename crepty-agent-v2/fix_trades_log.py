import pandas as pd

# Read the original CSV from the data directory
input_path = 'data/futures_trades_log.csv'
output_path = 'futures_trades_log_fixed.csv'

df = pd.read_csv(input_path)

# Rename 'price' to 'close' if it exists
if 'price' in df.columns:
    df = df.rename(columns={'price': 'close'})

# Save to a new file in the project root
df.to_csv(output_path, index=False)

print(f"Fixed file saved as {output_path}")
