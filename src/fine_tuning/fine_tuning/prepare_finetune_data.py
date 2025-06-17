import os
import json

TRAIN_DIR = "data/train"
OUTPUT_JSONL = os.path.join(TRAIN_DIR, "fine_tune.jsonl")

def convert_to_finetune_format():
    """ 将 LAMMPS 代码转换为 GPT 微调格式 """
    data_samples = []
    
    for subdir in ["rule_based", "human_written"]:
        folder_path = os.path.join(TRAIN_DIR, subdir)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                code = f.read().strip()
            
            prompt = f"Generate a LAMMPS input script similar to:\n{code}\n"
            data_samples.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": code}]})

    # 保存为 JSONL 格式
    with open(OUTPUT_JSONL, "w") as f:
        for item in data_samples:
            json.dump(item, f)
            f.write("\n")

    print(f"微调数据已保存到 {OUTPUT_JSONL}")

if __name__ == "__main__":
    convert_to_finetune_format()
