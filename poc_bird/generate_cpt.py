import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import config

# --- Model Configuration ---
MODEL_NAME = config.LLM_MODEL_NAME
LORA_ADAPTER_PATH = "/mnt/shared/adarsh/train/lora-adapter"  # Path to your fine-tuned LoRA adapter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Prompt Templates ---
SYSTEM_PROMPT = """\
You are an expert at converting everyday situations into a Bayesian-emotion table.
"""

# Template for factor generation (base model)
FACTOR_GENERATION_PROMPT = """\
SCENARIO
"{scenario}"

GOAL
Identify {num_factors} binary factors that would influence emotions in this scenario.

OUTPUT --- return ONE JSON object, nothing else.

JSON SPEC
{{
  "factors": {{
    "<factor-1>": ["<value1>", "<value2>"],
    "<factor-2>": ["<value1>", "<value2>"],
    "<factor-3>": ["<value1>", "<value2>"]
  }}
}}

RULES
1. Pick factors that obviously influence emotions in this scenario.
2. Use clear, mutually-exclusive binary values (e.g. "first_time"/"repeated").
3. Make factors relevant and meaningful to the scenario.
4. Do **not** include comments or extra keys.
"""

# Template for emotion generation (LoRA adapter)
EMOTION_GENERATION_PROMPT = "PROMPT: Given the tweet, generate a JSON object with the probability for each emotion.\nTweet: {context_description}\nRESPONSE:"

USER_PROMPT_TEMPLATE = """\
SCENARIO
“{scenario}”

GOAL
Create a Conditional-Probability Table (CPT) that estimates the **probability distribution of emotions** for this scenario
using exactly **{num_factors} binary factors** (2 values each ⇒ {num_rows} rows).

OUTPUT --- return ONE JSON object, nothing else.

JSON SPEC
{{
  "factors": {{
    "<factor-1>": ["<v1>", "<v2>"],
    "<factor-2>": ["<v1>", "<v2>"],
    "<factor-3>": ["<v1>", "<v2>"]
  }},
  "cpt":[
    {{"<factor-1>":"v1","<factor-2>":"v1","<factor-3>":"v1", "emotions": {{"<emotion-1>": 0.50, "<emotion-2>": 0.45, "<emotion-3>": 0.05}} }},
    … {num_rows_minus_one} more rows, every combination exactly once …
  ]
}}

RULES
1. Pick factors that obviously influence emotions in this scenario (e.g. *offence_history*).
2. Use clear, mutually-exclusive values (e.g. "first_time"/"repeated").
3. **Choose a diverse, relevant set of emotions** for the `"emotions"` dictionary. Do not just use the examples.
4. Fill **all {num_rows}** factor-value combinations in `"cpt"`.
5. The `"emotions"` value must be a dictionary of emotion probabilities.
6. The probabilities for all emotions in a single row's `"emotions"` dictionary **must sum to 1.0**.
7. Make numbers monotonic and sensible: probabilities should change logically with the factors (e.g. anger increases if it's a repeated offence).
8. Do **not** include comments or extra keys.
9. Each combination must list at least 3 emotions with non-zero probabilities.

END
"""

def load_models():
    """Loads both base model and LoRA adapter for hybrid CPT generation.
    """
    print(f"Loading base model: {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load base model with optimized settings
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # For H200 optimization
    )
    
    # Load LoRA adapter for emotion probability distributions
    print(f"Loading LoRA adapter from: {LORA_ADAPTER_PATH}...")
    lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    base_model.eval()
    lora_model.eval()
    
    print("Both base model and LoRA adapter loaded successfully!")
    return tokenizer, base_model, lora_model

def extract_json_from_response(response_text):
    """Extracts a JSON object from the model's text response.
    """
    # Use a regex to find the JSON block, even with markdown backticks
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback if no markdown block is found, find the first '{' and last '}'
        try:
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_str = response_text[start_index:end_index+1]
            else:
                raise ValueError("No JSON object found in the response.")
        except ValueError:
             raise ValueError("No JSON object found in the response.")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("--- ERROR: Failed to parse JSON. ---")
        print(f"Original Error: {e}")
        print("--- Faulty JSON String ---")
        print(json_str)
        print("--------------------------")
        raise

def generate_factors_for_scenario(tokenizer, base_model, scenario, num_factors):
    """Generate factors for a scenario using the base model."""
    factor_prompt = FACTOR_GENERATION_PROMPT.format(
        scenario=scenario, num_factors=num_factors
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": factor_prompt},
    ]
    
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    
    print("--- Generating factors with base model ---")
    try:
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.6,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generation_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        factors_data = extract_json_from_response(generation_only)
        return factors_data
    except Exception as e:
        print(f"Factor generation error: {e}")
        return None

def generate_emotions_for_context(tokenizer, lora_model, context_description):
    """Generate emotion probabilities for a context using the LoRA adapter."""
    # Use EXACT format that LoRA adapter was trained on
    emotion_prompt = f"PROMPT: Given the tweet, generate a JSON object with the probability for each emotion.\nTweet: {context_description}\nRESPONSE:"
    
    inputs = tokenizer(emotion_prompt, return_tensors="pt").to(DEVICE)
    
    try:
        with torch.no_grad():
            outputs = lora_model.generate(
                **inputs,
                max_new_tokens=512,  # Increased for full emotion dictionary
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generation_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Debug output
        print(f"🔍 LoRA Raw Output: {repr(generation_only)}")
        
        emotion_data = extract_json_from_response(generation_only)
        if emotion_data:
            # Filter out emotions with zero probability to reduce CPT complexity
            filtered_emotions = {emotion: prob for emotion, prob in emotion_data.items() if prob > 0.0}
            
            if filtered_emotions:
                print(f"✅ Parsed JSON: {len(filtered_emotions)} non-zero emotions")
                print(f"   Emotions: {list(filtered_emotions.keys())}")
                return filtered_emotions
            else:
                print(f"❌ No non-zero emotions found in: {emotion_data}")
                return None
        else:
            print(f"❌ Failed to parse JSON from: {generation_only}")
            return None
    except Exception as e:
        print(f"Emotion generation error: {e}")
        return None

def generate_hybrid_cpt_for_scenario(tokenizer, base_model, lora_model, scenario, num_factors=3):
    """
    Generates a Conditional Probability Table (CPT) using hybrid approach:
    - Base model generates factors
    - LoRA adapter generates emotion probabilities for each factor combination
    """
    print(f"\n=== HYBRID CPT GENERATION FOR: {scenario} ===")
    
    # Step 1: Generate factors using base model
    print("Step 1: Generating factors...")
    factors_data = generate_factors_for_scenario(tokenizer, base_model, scenario, num_factors)
    
    if not factors_data or "factors" not in factors_data:
        print("❌ Failed to generate factors")
        return None
    
    factors = factors_data["factors"]
    print(f"✅ Generated factors: {list(factors.keys())}")
    
    # Step 2: Generate all factor combinations
    import itertools
    factor_names = list(factors.keys())
    factor_values = [factors[name] for name in factor_names]
    
    combinations = list(itertools.product(*factor_values))
    print(f"Step 2: Generated {len(combinations)} factor combinations")
    
    # Step 3: Generate emotion probabilities for each combination
    print("Step 3: Generating emotion probabilities...")
    cpt_rows = []
    
    for i, combination in enumerate(combinations):
        # Create a factor combination dictionary
        factor_dict = {}
        context_parts = []
        
        for j, factor_name in enumerate(factor_names):
            factor_value = combination[j]
            factor_dict[factor_name] = factor_value
            context_parts.append(f"{factor_name} is {factor_value}")
        
        # Create a tweet-like description that matches LoRA training format
        tweet_like_description = f"Feeling {scenario.lower()} when {', '.join(context_parts)}"
        
        # Use LoRA adapter to generate emotion probabilities
        emotion_probs = generate_emotions_for_context(tokenizer, lora_model, tweet_like_description)
        
        if emotion_probs:
            # Create CPT row
            cpt_row = factor_dict.copy()
            cpt_row["emotions"] = emotion_probs
            cpt_rows.append(cpt_row)
            print(f"  ✅ Combination {i+1}/{len(combinations)} complete")
        else:
            print(f"  ❌ Failed to generate emotions for combination {i+1}")
    
    if not cpt_rows:
        print("❌ Failed to generate any CPT rows")
        return None
    
    # Step 4: Assemble final CPT
    final_cpt = {
        "factors": factors,
        "cpt": cpt_rows
    }
    
    print(f"✅ Hybrid CPT generation complete: {len(cpt_rows)} rows generated")
    return final_cpt

def get_row_key(row, factor_names):
    """Creates a canonical, hashable key from a CPT row's factor values."""
    return tuple(sorted((f, row.get(f)) for f in factor_names))


def main():
    """
    Main function to batch-generate CPTs for all scenarios.
    It generates multiple samples for each scenario and averages the emotion probabilities.
    """
    cpt_dir = config.CPT_DIR
    scenarios_file = config.SCENARIOS_FILE
    num_samples = 2  # Number of samples to generate per scenario

    # Ensure the output directory exists
    os.makedirs(cpt_dir, exist_ok=True)

    try:
        with open(scenarios_file, 'r') as f:
            scenarios_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{scenarios_file}' was not found.")
        print("Please ensure the scenarios file is in the same directory as this script.")
        return

    tokenizer, base_model, lora_model = load_models()

    for scenario_item in scenarios_data['scenarios']:
        scenario_id = scenario_item['id']
        scenario_desc = scenario_item['description']
        output_filename = os.path.join(cpt_dir, f"{scenario_id}.json")

        if os.path.exists(output_filename):
            print(f"CPT for '{scenario_id}' already exists. Skipping.")
            continue

        print(f"\n--- Generating {num_samples} CPT samples for '{scenario_id}' ---")
        print(f"Scenario: {scenario_desc}")

        all_cpts = []
        for i in range(num_samples):
            print(f"--- Generating sample {i + 1}/{num_samples} ---")
            cpt_data = generate_hybrid_cpt_for_scenario(tokenizer, base_model, lora_model, scenario_desc)
            if cpt_data and "factors" in cpt_data and "cpt" in cpt_data:
                all_cpts.append(cpt_data)
            else:
                print(f"WARNING: Received invalid or empty CPT data for sample {i + 1}. Skipping sample.")

        if not all_cpts:
            print(f"Failed to generate any valid CPTs for '{scenario_id}'.")
            continue

        # --- Aggregation Step ---
        print("--- Aggregating CPT samples ---")
        
        reference_cpt = all_cpts[0]
        factor_names = list(reference_cpt["factors"].keys())

        aggregated_data = {}

        for cpt in all_cpts:
            if set(factor_names) != set(cpt["factors"].keys()):
                print(f"WARNING: Factor names in sample mismatch reference: {list(cpt['factors'].keys())}. Skipping sample.")
                continue
            
            for row in cpt["cpt"]:
                row_key = get_row_key(row, factor_names)
                if row_key not in aggregated_data:
                    aggregated_data[row_key] = {"emotion_sums": {}, "count": 0}
                
                aggregated_data[row_key]["count"] += 1
                for emotion, prob in row.get("emotions", {}).items():
                    sums = aggregated_data[row_key]["emotion_sums"]
                    sums[emotion] = sums.get(emotion, 0.0) + prob

        final_cpt = {"factors": reference_cpt["factors"], "cpt": []}
        
        for ref_row in reference_cpt["cpt"]:
            row_key = get_row_key(ref_row, factor_names)
            
            if row_key in aggregated_data:
                data = aggregated_data[row_key]
                count = data["count"]
                
                if count == 0: continue

                averaged_emotions = {
                    emotion: total_prob / count
                    for emotion, total_prob in data["emotion_sums"].items()
                }

                total_prob = sum(averaged_emotions.values())
                if total_prob > 0:
                    averaged_emotions = {e: p / total_prob for e, p in averaged_emotions.items()}
                
                new_row = dict(ref_row)
                new_row["emotions"] = averaged_emotions
                final_cpt["cpt"].append(new_row)
            else:
                print(f"WARNING: Row {row_key} from reference CPT not found in aggregated data. Using original.")
                final_cpt["cpt"].append(ref_row)

        if final_cpt["cpt"]:
            with open(output_filename, "w") as f:
                json.dump(final_cpt, f, indent=2)
            print(f"Successfully generated and saved aggregated CPT to {output_filename}")
        else:
            print(f"Failed to generate aggregated CPT for '{scenario_id}'.")

if __name__ == "__main__":
    main()
