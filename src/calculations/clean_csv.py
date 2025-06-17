import pandas as pd
import string
import csv

def is_printable(s, threshold=0.9):
    if not isinstance(s, str):
        return False
    printable = set(string.printable + '\n\r\t')
    return sum(c in printable for c in s) / max(1, len(s)) > threshold

for fname in ['test.csv', 'val.csv']:
    df = pd.read_csv(fname)
    df_clean = df[df['label'].apply(is_printable)]
    df_clean.to_csv(fname, index=False, quoting=csv.QUOTE_ALL, escapechar='\\') 