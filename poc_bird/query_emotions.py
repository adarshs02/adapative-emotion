import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
SCENARIOS_FILE = "scenarios.json"
CPT_DIR = "cpts"
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SIMILARITY_MODEL_NAME = 'Qwen/Qwen3-Embedding-8B'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACTOR_PROMPT_TEMPLATE = """\
You are a JSON-only API. Your only function is to determine the values for a given set of factors based on a user's situation. Do not provide any explanations or text outside of the JSON object.

SITUATION:
"{user_situation}"

MATCHED SCENARIO:
"{scenario_description}"

FACTORS:
{factors_json}

Your response MUST be a single, valid JSON object and nothing else.

JSON SPEC:
{{
  "factor_values": {{
    "<factor_1_name>": "<chosen_value_1>",
    "<factor_2_name>": "<chosen_value_2>",
    ...
  }}
}}
"""



def load_llm_model():
    """Loads the tokenizer and model from Hugging Face for factor selection."""
    print(f"Loading LLM for querying: {LLM_MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("LLM loaded successfully!")
    return tokenizer, model

def extract_json_from_response(response_text):
    """Extracts a JSON object from the model's text response, handling markdown code blocks."""
    try:
        # Handle markdown code blocks (e.g., ```json ... ```)
        if "```" in response_text:
            # Extract content between the first and last backticks
            start_block = response_text.find("```")
            end_block = response_text.rfind("```")
            if start_block != -1 and end_block != -1 and start_block != end_block:
                # Potentially has a language specifier like 'json'
                code_block = response_text[start_block + 3 : end_block]
                # Find the start of the JSON object within the block
                json_start = code_block.find('{')
                if json_start != -1:
                    code_block = code_block[json_start:]
                json_str = code_block.strip()
            else: # Fallback for single ```
                 raise ValueError("Malformed markdown code block.")
        else:
            # Original logic for plain JSON
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_str = response_text[start_index:end_index+1]
            else:
                 raise ValueError("No JSON object found in the response.")

        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing JSON from LLM response: {e}")
        print(f"Raw response: {response_text}")
        return None

def get_factor_values_from_llm(user_situation, scenario_description, factors, tokenizer, model):
    """Asks the LLM to determine the values for a list of factors."""
    print("\n--- Asking LLM to determine factor values ---")
    
    factors_json = json.dumps(factors, indent=2)
    prompt = FACTOR_PROMPT_TEMPLATE.format(
        user_situation=user_situation,
        scenario_description=scenario_description,
        factors_json=factors_json
    )

    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generation_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    json_response = extract_json_from_response(generation_only)
    if json_response and "factor_values" in json_response:
        return json_response["factor_values"]
    
    return None

def find_top_similar_scenarios(user_situation, scenarios, similarity_model, top_k=5):
    """
    Finds the top_k most similar scenarios and their scores.
    Returns a list of dictionaries, e.g., [{'score': 0.8, 'scenario': {...}}, ...].
    """
    print(f"\n--- Finding top {top_k} similar scenarios using '{SIMILARITY_MODEL_NAME}' ---")
    scenario_descriptions = [s['description'] for s in scenarios]
    
    user_embedding = similarity_model.encode(user_situation, convert_to_tensor=True)
    scenario_embeddings = similarity_model.encode(scenario_descriptions, convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_embedding, scenario_embeddings)[0]
    
    top_results = torch.topk(cosine_scores, k=min(top_k, len(scenarios)))

    matches = []
    for score, idx in zip(top_results.values, top_results.indices):
        matches.append({'score': score.item(), 'scenario': scenarios[idx]})
    
    print("Top matches from similarity search:")
    for match in matches:
        print(f"  - Score: {match['score']:.4f} - {match['scenario']['description']}")
        
    return matches

def get_probabilities_for_factors(cpt_data, selected_factors):
    """Finds the matching emotion probabilities for a given set of factor values."""
    for entry in cpt_data['cpt']:
        match = all(entry.get(factor_name) == factor_value for factor_name, factor_value in selected_factors.items())
        if match:
            return entry.get('emotions')
    return None

def check_for_similar_logged_situation(user_situation, similarity_model, filename="no_scenario.json", threshold=0.9):
    """Checks if a very similar situation is already logged in the JSON file."""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return False

    try:
        with open(filename, 'r') as f:
            logged_data = json.load(f)
        if not isinstance(logged_data, list) or not logged_data:
            return False
    except (json.JSONDecodeError, IOError):
        return False # Cannot read or parse, so assume no similar situation exists

    logged_situations = [item.get("situation") for item in logged_data if item.get("situation")]
    if not logged_situations:
        return False

    user_embedding = similarity_model.encode(user_situation, convert_to_tensor=True)
    logged_embeddings = similarity_model.encode(logged_situations, convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_embedding, logged_embeddings)

    max_score = torch.max(cosine_scores).item()

    if max_score > threshold:
        print(f"Found an already logged situation with high similarity (score: {max_score:.2f}). Skipping log.")
        return True

    return False

def log_unmatched_situation(user_situation):
    """Logs a user situation that did not match any scenario to a JSON file."""
    filename = "no_scenario.json"
    print(f"Logging situation to '{filename}' for future review.")
    
    data = []
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"Warning: '{filename}' does not contain a JSON list. Overwriting with new list.")
                data = []
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read or parse '{filename}'. Overwriting.")
            data = []
            
    data.append({"situation": user_situation})
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"Error writing to '{filename}': {e}")

def main():
    """Main function to run the fully automated emotion query system."""
    if not os.path.exists(SCENARIOS_FILE):
        print(f"Error: Scenarios file not found at '{SCENARIOS_FILE}'")
        return

    with open(SCENARIOS_FILE, 'r') as f:
        scenarios_data = json.load(f)

    print(f"Loading similarity model: {SIMILARITY_MODEL_NAME} on {DEVICE}...")
    similarity_model = SentenceTransformer(SIMILARITY_MODEL_NAME, device=DEVICE)
    print("Similarity model loaded successfully!")

    user_situation = input("\nPlease describe your situation: ")

    all_scenarios = scenarios_data["scenarios"]
    
    top_matches = find_top_similar_scenarios(user_situation, all_scenarios, similarity_model, top_k=5)

    best_scenario = None

    # Decide whether to use similarity search fallback or the top match
    if not top_matches or top_matches[0]['score'] < 0.32:
        score = top_matches[0]['score'] if top_matches else 0
        print(f"\nCould not find a sufficiently similar scenario (top score: {score:.2f} < 0.32).")
        
        if not check_for_similar_logged_situation(user_situation, similarity_model):
            log_unmatched_situation(user_situation)

        print("Defaulting to 'unsure_scenario'.")
        best_scenario = next((s for s in all_scenarios if s['id'] == 'unsure_scenario'), None)
    else:
        # High score path: use the top match from similarity search
        best_scenario = top_matches[0]['scenario']
        print(f"\n--- Scenario selected by embedding model: '{best_scenario['description']}' (Score: {top_matches[0]['score']:.4f}) ---")

    # --- Process the chosen scenario ---
    if not best_scenario:
        print("Could not determine a scenario. Exiting.")
        return
        
    # Load the LLM for factor extraction
    tokenizer, model = load_llm_model()

    cpt_filename = os.path.join(CPT_DIR, f"{best_scenario['id']}.json")

    if not os.path.exists(cpt_filename):
        print(f"\nError: CPT file not found for scenario '{best_scenario['id']}'.")
        print(f"Please run 'python generate_cpt.py' to generate all required CPTs.")
        return

    with open(cpt_filename, 'r') as f:
        cpt_data = json.load(f)

    print(f"\n--- Analyzing scenario: '{best_scenario['description']}' ---")

    selected_factors = get_factor_values_from_llm(
        user_situation,
        best_scenario['description'],
        cpt_data['factors'],
        tokenizer,
        model
    )

    if not selected_factors:
        print("Could not determine factor values from LLM. Exiting.")
        return

    print("\n--- LLM Selected Factors ---")
    for factor, value in selected_factors.items():
        print(f"- {factor}: {value}")

    probabilities = get_probabilities_for_factors(cpt_data, selected_factors)

    if probabilities:
        print("\n--- Emotion Probabilities ---")
        for emotion, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
            print(f"- {emotion}: {prob:.2f}")
    else:
        print("\nCould not find a matching entry for the selected factors.")

    print(f'\nOriginal Situation: "{user_situation}"')

if __name__ == "__main__":
    main()
