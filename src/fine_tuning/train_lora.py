# LoRA Training Script (HuggingFace PEFT, pseudo-code)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import json
from datasets import load_dataset

MODEL_NAME = 'gpt2'  # or your base LLM
DATA_PATH = './finetune/dataset_gpt4/train_gpt4.jsonl'

train_dataset = load_dataset('json', data_files=DATA_PATH, split='train')

def preprocess(example):
    messages = example['messages']
    prompt = messages[0]['content']
    label = messages[1]['content']
    return {'text': prompt + '\n' + label}

train_dataset = train_dataset.map(preprocess)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

tokenized = train_dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir='./finetune/lora_model',
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=1000,
    logging_steps=100,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

trainer.train() 