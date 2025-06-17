import openai
from scripts.utils import call_llm

def evaluate_fine_tuned():
    """ 评估微调后的模型 """
    prompt = "Generate a LAMMPS input script for a Cu metal system."
    
    # 选择 Fine-tuned 模型
    model_name = "ft:gpt-4o-custom"
    generated_code = call_llm(prompt, model=model_name)
    
    print("Fine-tuned LLM 生成的 LAMMPS 代码:\n", generated_code)

if __name__ == "__main__":
    evaluate_fine_tuned()
