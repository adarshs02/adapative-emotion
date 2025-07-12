import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --- 1. Configuration ---
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_ADAPTER_PATH = "./lora-adapter"
EVAL_DATA_PATH = "eval_dataset.json"

# --- 2. Load Model and Tokenizer ---
print("Loading base model and tokenizer...")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
# Load the LoRA adapter and merge it with the base model
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model = model.merge_and_unload() # Merge LoRA weights into the base model

# --- 3. Load Evaluation Data ---
print(f"Loading evaluation data from {EVAL_DATA_PATH}...")
eval_df = pd.read_json(EVAL_DATA_PATH, lines=True)

# --- 4. Evaluation Loop ---
correct_predictions = 0
total_predictions = len(eval_df)

print("Starting evaluation...")
for index, row in tqdm(eval_df.iterrows(), total=total_predictions, desc="Evaluating"):
    tweet = row['tweet']
    true_emotion = str(row['emotion']).strip()

    # Construct the prompt without the answer
    prompt = f"<s>[INST] Given the tweet: '{tweet}', identify the emotion. [/INST] Emotion:"

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate the output
    outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.0)
    
    # Decode the generated tokens to get the predicted emotion
    predicted_emotion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Check if the prediction is correct
    if predicted_emotion.lower() == true_emotion.lower():
        correct_predictions += 1

# --- 5. Report Results ---
accuracy = (correct_predictions / total_predictions) * 100
print("\n--- Evaluation Complete ---")
print(f"Total Samples: {total_predictions}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
