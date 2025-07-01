import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Model Configuration ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
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
  "factors":[
    {{"name":"<factor-1>","values":["<v1>","<v2>"]}},
    {{"name":"<factor-2>","values":["<v1>","<v2>"]}},
    {{"name":"<factor-3>","values":["<v1>","<v2>"]}}
  ],
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

    return json.loads(json_str)

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

def main():
    """
    Main function to batch-generate CPTs for all scenarios.
    """
    cpt_dir = "cpts"
    scenarios_file = "scenarios.json"

    with open(scenarios_file, 'r') as f:
        scenarios_data = json.load(f)

    tokenizer, model = load_model()

    for scenario_item in scenarios_data['scenarios']:
        scenario_id = scenario_item['id']
        scenario_desc = scenario_item['description']
        output_filename = os.path.join(cpt_dir, f"{scenario_id}.json")

        if os.path.exists(output_filename):
            print(f"CPT for '{scenario_id}' already exists. Skipping.")
            continue

        print(f"\n--- Generating CPT for '{scenario_id}' ---")
        print(f"Scenario: {scenario_desc}")
        cpt_data = generate_cpt(tokenizer, model, scenario_desc)

        if cpt_data:
            with open(output_filename, "w") as f:
                json.dump(cpt_data, f, indent=2)
            print(f"Successfully generated and saved CPT to {output_filename}")
        else:
            print(f"Failed to generate CPT for '{scenario_id}'.")

if __name__ == "__main__":
    main()
