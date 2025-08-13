import csv

# Strictly clean trade_log_clean.csv: keep only rows with exactly 8 fields (header + 7 commas)
with open('trade_log_clean.csv', 'r', encoding='utf-8') as infile, open('trade_log_clean_fixed.csv', 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        if len(row) == 8:
            writer.writerow(row)

print('trade_log_clean_fixed.csv created with only valid rows.')
