import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

import config

# --- Model Configuration ---
MODEL_NAME = config.LLM_MODEL_NAME
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Prompt Templates ---
SYSTEM_PROMPT = """\
You are an expert at converting everyday situations into a Bayesian-emotion table.
"""

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

def load_model():
    """Loads the tokenizer and model from Hugging Face.
    """
    print(f"Loading model: {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model

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

def generate_cpt(tokenizer, model, scenario: str, num_factors: int = 3):
    """
    Generates a Conditional Probability Table (CPT) for a given scenario using a local LLM.
    """
    num_rows = 2**num_factors
    num_rows_minus_one = num_rows - 1
    user_prompt = USER_PROMPT_TEMPLATE.format(
        scenario=scenario, num_factors=num_factors, num_rows=num_rows, num_rows_minus_one=num_rows_minus_one
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Apply the chat template for the specific model
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    print("--- Sending request to LLM ---")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # The output includes the prompt, so we need to decode only the generated tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generation_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print("--- LLM Response Received ---")
        cpt_data = extract_json_from_response(generation_only)
        return cpt_data
    except Exception as e:
        print(f"An error occurred: {e}")
        if "generation_only" in locals():
            print("--- Raw LLM Response ---")
            print(generation_only)
        return None

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

    tokenizer, model = load_model()

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
            cpt_data = generate_cpt(tokenizer, model, scenario_desc)
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
