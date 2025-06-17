# Basic environment setup
import os
import torch 
import json

# Now we can safely import the rest
import re
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Helper functions for system info
def print_gpu_info():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Running on GPU: {device_name}")
        # Print CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        # Print available memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        free_memory_gb = free_memory / (1024 ** 3)
        total_memory_gb = total_memory / (1024 ** 3)
        print(f"GPU Memory: {free_memory_gb:.2f}GB free / {total_memory_gb:.2f}GB total")
    else:
        print("❌ No GPU available, running on CPU")

# Determine project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

DATA_PATH = project_root / 'EmotionBench' / 'situations.json'

# Load EmotionBench situations
with open(DATA_PATH, 'r') as f:
    data = json.load(f)

# Define model parameters for Qwen2.5
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6
DO_SAMPLE = False

print_gpu_info()
print(f"Loading model {MODEL_NAME}...")

# Import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to configure transformers to avoid flash attention
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Define the model loading function with error handling
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print("Loading model with optimized attention implementation...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if DEVICE == "cuda" else None,  # Automatically handle device mapping
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,  # Use FP16 on GPU
        trust_remote_code=True,
        attn_implementation="eager"  # Use eager implementation instead of flash attention
    )
    model.eval()
    print("Model loaded successfully!")
    
except ImportError as ie:
    if "flash_attn_2_cuda" in str(ie) and "GLIBCXX_3.4.32" in str(ie):
        print(f"Expected FlashAttention error: {ie}")
        print("Attempting to load with different attention mechanism...")
        try:
            # Try again with more restrictive settings
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto" if DEVICE == "cuda" else None,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                trust_remote_code=True,
                use_flash_attention_2=False,  # Explicitly disable flash attention
                attn_implementation="eager"  # Force eager (non-flash) implementation
            )
            model.eval()
            print("Model loaded with eager attention!")
        except Exception:
            # If that also fails, go to fallback model
            raise
    else:
        raise
except Exception as e:
    print(f"Error loading Qwen2.5 model: {e}")
    print("Falling back to smaller distilgpt2 model...")
    MODEL_NAME = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    print("Fallback model loaded successfully!")

# Define a generation function to match what was used by ModelInitializer
def gen_response(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, do_sample=DO_SAMPLE):
    try:
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Generate text with error handling
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id 
            )
        
        # Decode the generated text
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_length = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
        completion = full_output[prompt_length:]
    
    except Exception as e:
        print(f"Error in generation: {e}")
        # Provide default values in case of failure
        full_output = prompt + "\n[Error in generation]"  
        completion = "[Error in generation]"
    
    return {"full_raw_output": full_output, "completion": completion}

results = []

PANAS_EMOTIONS = [
    "Interested", "Distressed", "Excited", "Upset", "Strong", "Guilty", "Scared", "Hostile", "Enthusiastic", "Proud",
    "Irritable", "Alert", "Ashamed", "Inspired", "Nervous", "Determined", "Attentive", "Jittery", "Active", "Afraid"
]

def extract_scores(response):
    # Assumes 're' is imported globally and 'tqdm.write' is available from 'from tqdm import tqdm'
    numbers = re.findall(r'\b[1-5]\b', response)
    if len(numbers) >= 20:
        parsed = list(map(int, numbers[:20]))
    else:
        tqdm.write(f"Warning: Could not parse 20 scores. Found {len(numbers)}. Filling missing with NaN.")
        parsed = list(map(int, numbers)) + [float('nan')] * (20 - len(numbers))
    tqdm.write(f"Parsed scores: {parsed}")
    return parsed


N_REPEATS = 1

output_rows = []
raw_results = []

# 1. Default (no scenario) baseline
for _ in tqdm(range(N_REPEATS), desc="Default baseline repeats"):
    # Randomize order each time
    order = list(range(20))
    random.shuffle(order)
    ordered_emotions = [PANAS_EMOTIONS[i] for i in order]
    prompt = (
        "Please indicate your degree of agreement regarding each statement.\n"
        "(1 = Not at all, 5 = Very much)\n"
    )
    for idx, emo in enumerate(ordered_emotions, 1):
        prompt += f"Q{idx}. I feel {emo}.\n"
    prompt += "You can only reply to numbers from 1 to 5. Respond with 20 numbers in order, separated by spaces or commas."
    response = gen_response(prompt)
    raw = response['full_raw_output']
    tqdm.write("-----\nDefault baseline run")
    tqdm.write(f"Prompt: {prompt}")
    tqdm.write(f"Model: {raw}")

    scores = extract_scores(raw)
    # Restore to original PANAS order
    restored_scores = [0]*20
    for idx, orig_idx in enumerate(order):
        restored_scores[orig_idx] = scores[idx]
    raw_results.append({'Type': 'Default', 'Scenario': '', 'scores': restored_scores})

default_means = [
    float('nan') if not raw_results else sum([r['scores'][i] for r in raw_results if r['Type']=='Default'])/N_REPEATS
    for i in range(20)
]
default_row = {'Type': 'Default', 'Scenario': ''}
default_row.update({emo: round(default_means[i], 2) for i, emo in enumerate(PANAS_EMOTIONS)})
output_rows.append(default_row)

# 2. Evoked (with scenario)
for emotion in tqdm(data['emotions'], desc="Processing emotions"):
    for factor in emotion['factors']:
        factor_name = factor['name']
        for scenario in tqdm(factor['scenarios'], desc=f"Scenarios for {emotion['name']}/{factor_name}", leave=False):
            scenario_raws = []
            for _ in range(N_REPEATS):
                order = list(range(20))
                random.shuffle(order)
                ordered_emotions = [PANAS_EMOTIONS[i] for i in order]
                prompt = (
                    f"Imagine you are in the situation: \"{scenario}\".\n"
                    "Please indicate your degree of agreement regarding each statement.\n"
                    "(1 = Not at all, 5 = Very much)\n"
                )
                for idx, emo in enumerate(ordered_emotions, 1):
                    prompt += f"Q{idx}. I feel {emo}.\n"
                prompt += (
                    "Respond ONLY with 20 numbers from 1 to 5, separated by spaces. Do NOT repeat the questions or add any other text.\n"
                    "For example: 3 4 5 2 1 5 4 3 2 1 4 5 3 2 1 5 4 3 2 1\n"
                )
                response = gen_response(prompt)
                raw = response['full_raw_output']
                tqdm.write("-----\nEvoked run")
                tqdm.write(f"Scenario: {scenario}")
                tqdm.write(f"Prompt: {prompt}")
                tqdm.write(f"Model: {raw}")
                scores = extract_scores(raw)
                # Restore to original PANAS order
                restored_scores = [0]*20
                for idx, orig_idx in enumerate(order):
                    restored_scores[orig_idx] = scores[idx]
                scenario_raws.append(restored_scores)
                raw_results.append({'Type': 'Evoked', 'Scenario': scenario, 'scores': restored_scores})
            # Average across N_REPEATS
            means = [
                float('nan') if not scenario_raws else sum([r[i] for r in scenario_raws])/N_REPEATS
                for i in range(20)
            ]
            row = {'Type': 'Evoked', 'Scenario': scenario}
            row.update({emo: round(means[i], 2) for i, emo in enumerate(PANAS_EMOTIONS)})
            output_rows.append(row)

# Save as PANAS-style CSV
output_df = pd.DataFrame(output_rows)
output_df.to_csv(project_root / "results" / "emotionbench" / "mistral-PANAS-testing.csv", index=False)

# (Optional) Also save raw results as before
with open(project_root / "results" / "emotionbench" / "emotionbench_llama_results1.json", 'w') as f:
    json.dump(raw_results, f, indent=2, ensure_ascii=False)
