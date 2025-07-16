import os
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# --- 1. Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH = "./eval_dataset.json"
RESULTS_FILE = "./eval_results_base.json"

def format_prompt(sample):
    """Formats a sample into a simplified prompt for the model."""
    return f"What is the primary emotion in the following tweet? Respond with a single word.\nTweet: {sample['tweet']}\nEmotion:"

def main():
    """Main evaluation function for the base model."""
    print(f"--- Evaluating Base Model: {MODEL_NAME} ---")

    # --- 2. Load Dataset ---
    if not os.path.exists(DATA_PATH):
        print(f"\u274c Evaluation data not found at {DATA_PATH}. Please run prepare_dataset.py first.")
        return

    with open(DATA_PATH, 'r') as f:
        eval_dataset = [json.loads(line) for line in f]
    
    print(f"Loaded {len(eval_dataset)} samples for evaluation.")

    # --- 3. Load Model & Tokenizer (Optimized for H100/H200) ---
    print("Loading model with bfloat16 and Flash Attention 2...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 4. Run Evaluation ---
    results = []
    incorrect_predictions = []

    print("\n--- Running Evaluation ---")
    for i, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        prompt = format_prompt(sample)
        true_emotion = sample['emotion'].strip().lower()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        full_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Simplified extraction: take the first word of the response
        predicted_emotion = full_response.split()[0].lower().strip(".,!?'\"")

        is_correct = predicted_emotion == true_emotion
        
        result_entry = {
            "sample_index": i,
            "tweet": sample['tweet'],
            "true_emotion": true_emotion,
            "predicted_emotion": predicted_emotion,
            "full_response": full_response,
            "is_correct": is_correct
        }
        results.append(result_entry)

        if not is_correct:
            incorrect_predictions.append(result_entry)

    # --- 5. Report and Save Results ---
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    accuracy = (correct_count / total_count) * 100

    print(f"\n--- Evaluation Complete ---")
    print(f"Correct Predictions: {correct_count}")
    print(f"Total Predictions:   {total_count}")
    print(f"Accuracy:            {accuracy:.2f}%")
    print(f"-------------------------")

    # Print incorrect predictions for review
    print(f"\n--- {len(incorrect_predictions)} Incorrect Predictions ---")
    for item in incorrect_predictions[:20]: # Print first 20 incorrect for brevity
        print(f"  - Tweet:    {item['tweet']}")
        print(f"    True:     '{item['true_emotion']}', Predicted: '{item['predicted_emotion']}'")

    # Save full results to a file
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nFull evaluation results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
