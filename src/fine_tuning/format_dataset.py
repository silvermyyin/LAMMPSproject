import pandas as pd
import os
import json

DATA_DIR = '../data/baseline'
OUT_DIR = './finetune/dataset_gpt4'
os.makedirs(OUT_DIR, exist_ok=True)

for split in ['test', 'val']:
    df = pd.read_csv(f'{DATA_DIR}/{split}.csv')
    # SFT格式：{"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": label}]}
    records = []
    for _, row in df.iterrows():
        records.append({
            "messages": [
                {"role": "user", "content": row['sample']},
                {"role": "assistant", "content": row['label']}
            ]
        })
    with open(f'{OUT_DIR}/{split}_gpt4.jsonl', 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n') 