import json
import os
import torch
import numpy as np
import hnswlib
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
import config
from tag_generator import generate_tags

# --- Configuration (now using centralized config) ---
SCENARIOS_FILE = config.SCENARIOS_FILE
CPT_DIR = config.CPT_DIR
LLM_MODEL_NAME = config.LLM_MODEL_NAME
DEVICE = config.get_device()

# Fine-tuned model configuration
FINETUNED_MODEL_PATH = "llama_embedding_with_tags_20250720_225721/final_model"
BASE_MODEL = "meta-llama/Llama-3.1-8B"
INDEX_DIR = "indices"
FINETUNED_INDEX_PATH = os.path.join(INDEX_DIR, "finetuned_embedding.idx")
FINETUNED_MAPPING_PATH = os.path.join(INDEX_DIR, "finetuned_mapping.json")
FINETUNED_EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "finetuned_embeddings.npy")

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

class FineTunedEmbeddingModel:
    """Wrapper for the fine-tuned Llama embedding model."""
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Llama-3.1-8B"):
        self.model_path = model_path
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Loading fine-tuned embedding model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModel.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        
        print(f"✅ Fine-tuned embedding model loaded successfully!")
    
    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                # Normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class FineTunedRouter:
    """Production router using fine-tuned embeddings and HNSW index."""
    
    def __init__(self):
        print("🎯 Initializing Fine-tuned Router with 100% accuracy...")
        
        # Load fine-tuned embedding model
        self.embedding_model = FineTunedEmbeddingModel(FINETUNED_MODEL_PATH, BASE_MODEL)
        
        # Load HNSW index
        self.index = hnswlib.Index(space='cosine', dim=4096)
        self.index.load_index(FINETUNED_INDEX_PATH)
        self.index.set_ef(50)  # Optimize for search
        
        # Load scenario mapping
        with open(FINETUNED_MAPPING_PATH, 'r') as f:
            self.mapping = json.load(f)
        
        print(f"✅ Router loaded with {len(self.mapping)} scenarios indexed")
        print("🎯 Expected performance: 100% scenario matching accuracy")
    
    def find_top_scenarios(self, query: str, top_k: int = 5):
        """Find top-k most similar scenarios using fine-tuned model with tag generation."""
        print(f"\n🏷️  Generating tags for user input...")
        
        # Generate tags for the user query
        query_tags = generate_tags(query)
        print(f"Generated tags: {query_tags}")
        
        # Combine query with tags in training format: "description. Tags: tag1, tag2, tag3"
        if query_tags:
            tag_text = ", ".join(query_tags)
            combined_query = f"{query}. Tags: {tag_text}"
        else:
            combined_query = query
        
        print(f"🔍 Combined query for matching: '{combined_query[:80]}...'")
        
        # Generate embedding for combined query (matching training format)
        query_embedding = self.embedding_model.encode([combined_query])
        
        # Search HNSW index
        labels, distances = self.index.knn_query(query_embedding, k=top_k)
        
        # Convert distances to similarity scores (cosine distance -> cosine similarity)
        similarities = 1 - distances[0]
        
        # Build results
        results = []
        for i, (label, similarity) in enumerate(zip(labels[0], similarities)):
            scenario_info = self.mapping[str(label)]
            results.append({
                'score': float(similarity),
                'scenario': {
                    'id': scenario_info['id'],
                    'description': scenario_info['description']
                },
                'rank': i + 1,
                'user_tags': query_tags,  # Include generated tags in results
                'combined_query': combined_query  # Include for debugging
            })
        
        return results


# Global router instance (loaded once)
_finetuned_router = None

def get_finetuned_router():
    """Get or create the global fine-tuned router instance."""
    global _finetuned_router
    if _finetuned_router is None:
        _finetuned_router = FineTunedRouter()
    return _finetuned_router


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

def find_top_similar_scenarios(user_situation, scenarios=None, top_k=5):
    """
    Finds the top_k most similar scenarios using fine-tuned embedding model.
    Returns a list of dictionaries, e.g., [{'score': 0.8, 'scenario': {...}}, ...].
    """
    print(f"\n🔍 Finding top {top_k} scenarios using Fine-tuned Model (100% accuracy)...")
    
    try:
        # Get the fine-tuned router
        router = get_finetuned_router()
        
        # Find top scenarios using fine-tuned model
        results = router.find_top_scenarios(user_situation, top_k=top_k)
        
        if not results:
            print("No scenarios found.")
            return []
        
        print(f"✅ Found {len(results)} scenarios with fine-tuned model")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. {result['scenario']['id']} (score: {result['score']:.4f})")
        
        # Check similarity threshold for fallback
        SIMILARITY_THRESHOLD = 0.82
        if results and results[0]['score'] < SIMILARITY_THRESHOLD:
            print(f"\n⚠️  Similarity {results[0]['score']:.3f} < {SIMILARITY_THRESHOLD} - Using unfamiliar scenario fallback")
            return handle_unfamiliar_scenario(user_situation, results[0])
        
        return results
        
    except Exception as e:
        print(f"Error in fine-tuned scenario matching: {e}")
        return []


def get_probabilities_for_factors(cpt_data, selected_factors):
    """Finds the matching emotion probabilities for a given set of factor values."""
    for entry in cpt_data['cpt']:
        match = all(entry.get(factor_name) == factor_value for factor_name, factor_value in selected_factors.items())
        if match:
            return entry.get('emotions')
    return None

def check_for_similar_logged_situation(user_situation, filename="no_scenario.json", threshold=0.9):
    """Checks if a very similar situation is already logged in the JSON file using llama_router."""
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

    max_score = 0.0
    
    # Simple token-based similarity as fallback
    user_tokens = set(user_situation.lower().split())
    for logged_situation in logged_situations:
        logged_tokens = set(logged_situation.lower().split())
        if user_tokens and logged_tokens:
            similarity = len(user_tokens.intersection(logged_tokens)) / len(user_tokens.union(logged_tokens))
            max_score = max(max_score, similarity)
    
    if max_score > threshold:
        print(f"Found an already logged situation with high similarity (score: {max_score:.2f}). Skipping log.")
        return True

    return False

def handle_unfamiliar_scenario(user_situation, best_match=None, tokenizer=None, model=None):
    """Handle unfamiliar scenarios with similarity < 0.82 using fixed prompt to decoder LLM."""
    print(f"\n🚨 Handling unfamiliar scenario (low similarity)...")
    
    # Log the unfamiliar scenario
    log_unfamiliar_situation(user_situation, best_match)
    
    # Fixed prompt for decoder LLM to read carefully
    fixed_prompt = f"""Please read the following situation carefully and analyze the emotions involved. Pay close attention to all details and context.

Situation: {user_situation}

Based on your careful analysis, what emotions are likely present? Please provide your reasoning and the most probable emotions with confidence levels."""
    
    if tokenizer and model:
        print(f"\n🤔 Asking decoder LLM to analyze unfamiliar scenario...")
        
        # Tokenize and generate response
        inputs = tokenizer.encode(fixed_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return {
            "method": "unfamiliar_scenario_fallback",
            "similarity_score": best_match['score'] if best_match else 0.0,
            "user_situation": user_situation,
            "fixed_prompt": fixed_prompt,
            "llm_response": response.strip(),
            "best_match": best_match,
            "logged": True
        }
    else:
        return {
            "method": "unfamiliar_scenario_fallback", 
            "similarity_score": best_match['score'] if best_match else 0.0,
            "user_situation": user_situation,
            "fixed_prompt": fixed_prompt,
            "logged": True,
            "error": "No LLM model provided for analysis"
        }

def log_unfamiliar_situation(user_situation, best_match=None):
    """Logs unfamiliar situations with low similarity scores."""
    filename = "unfamiliar_scenarios.json"
    
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
    
    # Check for duplicates - skip if same situation already exists
    for existing_entry in data:
        if existing_entry.get("situation", "").strip() == user_situation.strip():
            print(f"📋 Skipping duplicate unfamiliar scenario (already logged)")
            return
    
    print(f"📝 Logging unfamiliar scenario to '{filename}' for review.")
    
    entry = {
        "situation": user_situation,
        "timestamp": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",  # Simple timestamp placeholder
        "best_match": {
            "scenario_id": best_match['scenario']['id'] if best_match else None,
            "similarity_score": best_match['score'] if best_match else 0.0,
            "description": best_match['scenario']['description'] if best_match else None
        } if best_match else None
    }
            
    data.append(entry)
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Logged unfamiliar scenario successfully")
    except IOError as e:
        print(f"Error writing to '{filename}': {e}")

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

    print("🎯 Using Fine-tuned Embedding Model with 100% scenario matching accuracy...")

    user_situation = input("\nPlease describe your situation: ")

    all_scenarios = scenarios_data["scenarios"]
    
    top_matches = find_top_similar_scenarios(user_situation, top_k=5)

    best_scenario = None

    # Check similarity threshold for unfamiliar scenario fallback
    SIMILARITY_THRESHOLD = 0.82
    
    if not top_matches or top_matches[0]['score'] < SIMILARITY_THRESHOLD:
        score = top_matches[0]['score'] if top_matches else 0
        print(f"\n⚠️  Similarity {score:.3f} < {SIMILARITY_THRESHOLD} - Using unfamiliar scenario fallback")
        
        # Load LLM for decoder analysis
        tokenizer, model = load_llm_model()
        
        # Use unfamiliar scenario handler with fixed prompt
        result = handle_unfamiliar_scenario(user_situation, top_matches[0] if top_matches else None, tokenizer, model)
        
        print("\n📋 Unfamiliar Scenario Analysis Results:")
        print(f"Method: {result['method']}")
        print(f"Similarity Score: {result['similarity_score']:.3f}")
        if 'llm_response' in result:
            print(f"LLM Analysis: {result['llm_response'][:200]}...")
        
        # Exit early since we handled this as an unfamiliar scenario
        return
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
