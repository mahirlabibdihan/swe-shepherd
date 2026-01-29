import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, TaskType

# 1. Setup Model and Tokenizer
model_id = "meta-llama/Llama-3.1-8B"  # Replace with your base model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 1 output neuron for regression
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=1, 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

# 2. Load and Prepare Dataset
# Format: {"prompt": "User query", "reward": 0.85}
dataset = load_dataset("json", data_files="dataset/prm_train.jsonl", split="train")

def formatting_func(example):
    # This combines your prompt into a string the model evaluates
    return f"Instruction: {example['prompt']}\nScore:"

# 3. Configure LoRA for Classification
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./pointwise_reward_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=1,
    bf16=True,
    remove_unused_columns=False # Important to keep the 'reward' column for labels
)

# 5. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=512,
    formatting_func=formatting_func,
    data_collator=None, # TRL will handle the label mapping
)

# Manually ensure the labels are passed correctly
# TRL SFTTrainer expects a column named 'labels' for regression
dataset = dataset.map(lambda x: {"labels": float(x["reward"])})

trainer.train()