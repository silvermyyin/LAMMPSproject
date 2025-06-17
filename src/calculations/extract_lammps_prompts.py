import os
import re
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

REFERENCE_DIR = 'data/reference_scripts_flat'
OUTPUT_CSV = 'dedup_prompts.csv'
TEST_CSV = 'test.csv'
VAL_CSV = 'val.csv'

# 需要提取的 LAMMPS 关键命令
KEYWORDS = [
    'fix', 'velocity', 'timestep', 'pair_style', 'pair_coeff', 'boundary', 'atom_style', 'run'
]

def extract_parameters(text):
    params = {}
    # 逐行查找关键命令
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        for key in KEYWORDS:
            if line.startswith(key):
                if key not in params:
                    params[key] = []
                params[key].append(line)
    return params

def generate_prompt(params):
    # 模拟类型
    sim_type = ''
    temp = ''
    press = ''
    for fix in params.get('fix', []):
        if 'nvt' in fix:
            sim_type = 'NVT'
        elif 'npt' in fix:
            sim_type = 'NPT'
        elif 'nve' in fix:
            sim_type = 'NVE'
        # 温度、压力
        if 'temp' in fix:
            temp = re.findall(r'temp\s+([\d\.Ee+-]+)', fix)
            if temp:
                temp = temp[0]
        if 'press' in fix:
            press = re.findall(r'press\s+([\d\.Ee+-]+)', fix)
            if press:
                press = press[0]
    # velocity
    velocity = ''
    for v in params.get('velocity', []):
        m = re.search(r'create\s+([\d\.Ee+-]+)', v)
        if m:
            velocity = m.group(1)
    # pair_style
    pair_style = params.get('pair_style', [''])[0].replace('pair_style', '').strip() if params.get('pair_style') else ''
    # pair_coeff
    pair_coeff = params.get('pair_coeff', [''])[0].replace('pair_coeff', '').strip() if params.get('pair_coeff') else ''
    # boundary
    boundary = params.get('boundary', [''])[0].replace('boundary', '').strip() if params.get('boundary') else ''
    # atom_style
    atom_style = params.get('atom_style', [''])[0].replace('atom_style', '').strip() if params.get('atom_style') else ''
    # timestep
    timestep = params.get('timestep', [''])[0].replace('timestep', '').strip() if params.get('timestep') else ''
    # run
    run = params.get('run', [''])[0].replace('run', '').strip() if params.get('run') else ''

    prompt = f"Generate a LAMMPS input script for a {sim_type or 'general'} simulation "
    if atom_style:
        prompt += f"of an {atom_style} system "
    if pair_style:
        prompt += f"using {pair_style} pair style "
    if pair_coeff:
        prompt += f"with pair coefficients {pair_coeff} "
    if boundary:
        prompt += f"and {boundary} boundary conditions "
    if timestep:
        prompt += f"with a timestep of {timestep} "
    if run:
        prompt += f"running for {run} steps "
    if temp:
        prompt += f"at temperature {temp} "
    if press:
        prompt += f"and pressure {press} "
    prompt = prompt.strip() + '.'
    return prompt

def main():
    prompt2label = {}
    for fname in os.listdir(REFERENCE_DIR):
        if not fname.startswith('in.'):
            continue
        with open(os.path.join(REFERENCE_DIR, fname), 'r', errors='ignore') as f:
            text = f.read()
        params = extract_parameters(text)
        prompt = generate_prompt(params)
        # 只保留第一个遇到的 label
        if prompt not in prompt2label:
            prompt2label[prompt] = text
    df = pd.DataFrame([
        {'sample': k, 'label': v} for k, v in prompt2label.items()
    ])
    # 划分 test/val
    test_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # 保存为 test.csv 和 val.csv，格式为 sample,label，全部加引号
    test_df.to_csv(TEST_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    val_df.to_csv(VAL_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')

if __name__ == '__main__':
    main() 