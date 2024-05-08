import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# 기존에 학습된 Llama2 모델 불러오기
model_name = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 새로운 데이터셋 로드
train_dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")

# Fine Tuning 설정
training_args = TrainingArguments(
    output_dir="./llama2_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
)

# Fine Tuning 진행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# Fine Tuned 모델 저장
model.save_pretrained("./llama2_finetuned")
tokenizer.save_pretrained("./llama2_finetuned")