import os
import json
import torch
import argparse
import re
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# --- 1. Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "./eval_dataset.json"

def format_prompt(sample):
    """Formats a sample from the EmoKnow dataset into a prompt for the model."""
    return f"<s>[INST] Given the tweet: '{sample['tweet']}', identify the emotion. [/INST]"

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a LoRA-finetuned model.")
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Path to the LoRA adapter directory. If not provided, evaluates the base model.")
    return parser.parse_args()

def main():
    """Main evaluation function."""
    args = parse_args()
    print(f"--- Evaluating Model: {MODEL_NAME} ---")
    if args.lora_adapter_path:
        print(f"With LoRA adapter from: {args.lora_adapter_path}")
    else:
        print("Evaluating base model (no adapter).")

    # --- 2. Load Dataset ---
    if not os.path.exists(DATA_PATH):
        print(f"\u274c Evaluation data not found at {DATA_PATH}. Please run prepare_dataset.py first to generate it.")
        return

    with open(DATA_PATH, 'r') as f:
        eval_dataset = [json.loads(line) for line in f]
    
    print(f"Loaded {len(eval_dataset)} samples for evaluation.")



    # --- 3. Load Model & Tokenizer ---
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.lora_adapter_path:
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 4. Run Evaluation ---
    correct_predictions = 0
    total_predictions = 0

    print("\n--- First 5 Sample Predictions ---")
    for i, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        prompt = format_prompt(sample)
        true_emotion = sample['emotion'].strip().lower()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_emotion = response_text.split('[/INST]')[-1].strip().lower()

        if i < 5:
            print(f"Sample {i+1}:")
            print(f"  Tweet:           '{sample['tweet']}'")
            print(f"  True Emotion:    '{true_emotion}'")
            print(f"  Raw Prediction:    '{predicted_emotion}'")
            print(f"  Full Response:   '{response_text}'")
            print("-"*10)

        if predicted_emotion == true_emotion:
            correct_predictions += 1
        total_predictions += 1

    # --- 5. Report Results ---
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n--- Evaluation Complete ---")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions:   {total_predictions}")
    print(f"Accuracy:            {accuracy:.2f}%")
    print(f"-------------------------")

if __name__ == "__main__":
    main()
