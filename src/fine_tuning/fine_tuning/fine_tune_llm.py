import openai

FINE_TUNE_FILE = "data/train/fine_tune.jsonl"

def fine_tune_model():
    """ 运行 OpenAI 微调 """
    response = openai.File.create(
        file=open(FINE_TUNE_FILE, "rb"),
        purpose="fine-tune"
    )
    
    fine_tune_id = response["id"]
    print(f"上传数据成功，Fine-tune ID: {fine_tune_id}")

    response = openai.FineTune.create(
        training_file=fine_tune_id,
        model="gpt-4o"
    )
    print("微调模型创建成功")
    print(response)

if __name__ == "__main__":
    fine_tune_model()
