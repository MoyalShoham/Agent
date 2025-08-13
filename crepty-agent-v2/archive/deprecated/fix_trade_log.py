import csv

# Fix malformed trade_log.csv: keep only rows with exactly 8 fields (header + 7 commas)
with open('trade_log.csv', 'r', encoding='utf-8') as infile, open('trade_log_clean.csv', 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        if len(row) == 8:
            writer.writerow(row)

print('trade_log_clean.csv created with only valid rows.')
