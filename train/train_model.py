import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# --- 1. Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TRAIN_DATA_PATH = "./train_dataset2_converted.json"
OUTPUT_DIR = "./lora-adapter"

# --- 2. Data Preparation ---
print("Loading pre-processed training data...")
if not os.path.exists(TRAIN_DATA_PATH):
    print(f"\u274c Training data not found at {TRAIN_DATA_PATH}. Please run prepare_dataset.py first.")
    exit()

dataset = Dataset.from_json(TRAIN_DATA_PATH)

# --- 3. Model & Tokenizer Loading ---
print(f"Loading model: {MODEL_NAME}")

# --- 3. Model & Tokenizer Loading (Optimized for H100/H200) ---
print(f"Loading model: {MODEL_NAME} with bfloat16 and Flash Attention 2...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False # Required for gradient checkpointing
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 4. LoRA Configuration ---
print("Configuring LoRA...")
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 5. Training Setup ---
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Reduced from 64 to 1 for memory efficiency
    gradient_accumulation_steps=16,  # Increased to maintain effective batch size of 16
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True, # Use bfloat16 for Ampere GPUs
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

# --- 6. Start Training ---
print("Starting LoRA fine-tuning...")
trainer.train()

# --- 7. Save Model ---
print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
print("Training complete!")
