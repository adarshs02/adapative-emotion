"""
Tag-based emotion query system that uses tags for scenario matching instead of raw text.
This is a modified version of query_emotions.py that uses the tag-based router.
"""

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tag_router import route, route_top_k
import config

# --- Configuration (now using centralized config) ---
SCENARIOS_FILE = config.SCENARIOS_FILE
CPT_DIR = config.CPT_DIR
LLM_MODEL_NAME = config.LLM_MODEL_NAME
DEVICE = config.get_device()

# Updated scenarios file with tags
SCENARIOS_WITH_TAGS_FILE = SCENARIOS_FILE.replace('.json', '_with_tags.json')

FACTOR_PROMPT_TEMPLATE = """\
You are a JSON-only API. Your only function is to determine the values for a given set of factors based on a user's situation. Do not provide any explanations or text outside of the JSON object.

SITUATION:
"{user_situation}"

MATCHED SCENARIO:
"{scenario_description}"

SCENARIO TAGS:
{scenario_tags}

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

def get_factor_values_from_llm(user_situation, scenario_description, scenario_tags, factors, tokenizer, model):
    """Asks the LLM to determine the values for a list of factors."""
    print("\n--- Asking LLM to determine factor values ---")
    
    factors_json = json.dumps(factors, indent=2)
    scenario_tags_str = ", ".join(scenario_tags) if scenario_tags else "No tags available"
    
    prompt = FACTOR_PROMPT_TEMPLATE.format(
        user_situation=user_situation,
        scenario_description=scenario_description,
        scenario_tags=scenario_tags_str,
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
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"LLM Response: {response}")

    parsed_response = extract_json_from_response(response)
    if parsed_response and "factor_values" in parsed_response:
        return parsed_response["factor_values"]
    else:
        print("Failed to parse factor values from LLM response.")
        return None

def find_top_similar_scenarios(user_situation, scenarios=None, top_k=5):
    """
    Finds the top_k most similar scenarios and their scores using tag_router.
    Returns a list of dictionaries, e.g., [{'score': 0.8, 'scenario': {...}}, ...]
    """
    print(f"\n--- Finding top {top_k} similar scenarios using tag-based matching ---")
    print(f"User situation: {user_situation}")
    
    try:
        # Use tag-based router
        top_matches = route_top_k(user_situation, k=top_k)
        
        if not top_matches:
            print("No matches found.")
            return []
        
        print(f"Found {len(top_matches)} matches:")
        for i, match in enumerate(top_matches):
            print(f"{i+1}. {match['scenario_id']}: {match['description'][:60]}... "
                  f"(Score: {match['confidence']:.4f})")
            print(f"   Tags: {match.get('tags', [])}")
        
        return top_matches
        
    except Exception as e:
        print(f"Error in tag-based scenario matching: {e}")
        return []

def get_probabilities_for_factors(cpt_data, selected_factors):
    """Finds the matching emotion probabilities for a given set of factor values."""
    factor_key = tuple(selected_factors[factor] for factor in cpt_data['factors'])
    return cpt_data['cpt'].get(str(factor_key))

def check_for_similar_logged_situation(user_situation, filename="no_scenario.json", threshold=0.9):
    """Checks if a very similar situation is already logged in the JSON file using tag_router."""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return False

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            return False
    except (json.JSONDecodeError, IOError):
        return False

    # Use tag-based matching to check similarity
    for entry in data:
        if "situation" in entry:
            # Use tag router to check similarity
            _, similarity = route(entry["situation"])
            if similarity >= threshold:
                print(f"Similar situation already logged with similarity {similarity:.2f}")
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
    """Main function to run the tag-based emotion query system."""
    # Check for scenarios with tags file first
    scenarios_file = SCENARIOS_WITH_TAGS_FILE if os.path.exists(SCENARIOS_WITH_TAGS_FILE) else SCENARIOS_FILE
    
    if not os.path.exists(scenarios_file):
        print(f"Error: Scenarios file not found at '{scenarios_file}'")
        if scenarios_file == SCENARIOS_WITH_TAGS_FILE:
            print("Please run 'python build_tag_index.py' to generate scenarios with tags.")
        return

    with open(scenarios_file, 'r') as f:
        scenarios_data = json.load(f)

    print("Using tag-based Llama-3.1-8B embeddings for similarity search...")

    user_situation = input("\nPlease describe your situation: ")

    all_scenarios = scenarios_data["scenarios"]
    
    top_matches = find_top_similar_scenarios(user_situation, top_k=5)

    best_scenario = None

    # Decide whether to use similarity search fallback or the top match
    if not top_matches or top_matches[0]['score'] < 0.32:
        score = top_matches[0]['score'] if top_matches else 0
        print(f"\nCould not find a sufficiently similar scenario (top score: {score:.2f} < 0.32).")
        
        if not check_for_similar_logged_situation(user_situation):
            log_unmatched_situation(user_situation)

        print("Defaulting to 'unsure_scenario'.")
        best_scenario = next((s for s in all_scenarios if s['id'] == 'unsure_scenario'), None)
    else:
        # High score path: use the top match from similarity search
        best_scenario = top_matches[0]['scenario']
        print(f"\n--- Scenario selected by tag-based model: '{best_scenario['description']}' (Score: {top_matches[0]['score']:.4f}) ---")
        print(f"Matched tags: {best_scenario.get('tags', [])}")

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
        best_scenario.get('tags', []),
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
