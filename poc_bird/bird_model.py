import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_router import route, get_router
import config

class BirdModel:
    """A model for predicting emotions based on situations using CPTs."""

    def __init__(self, config=None):
        """Initializes the BirdModel."""
        default_config = self._get_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        self._load_models()
        self._load_scenarios()

    def _get_default_config(self):
        """Returns the default configuration."""
        return {
            "llm_model_name": config.LLM_MODEL_NAME,
            "device": config.get_device(),
            "scenarios_file": config.SCENARIOS_FILE,
            "cpt_dir": config.CPT_DIR,
            "factor_prompt_template": self._get_default_prompt_template(),
            "llm_max_new_tokens": config.LLM_MAX_NEW_TOKENS,
            "llm_temperature": config.LLM_TEMPERATURE
        }

    def _get_default_prompt_template(self):
        """Returns the default prompt template."""
        return ("""
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
        """)

    def _load_models(self):
        """Loads the LLM model."""
        print(f"Loading LLM for querying: {self.config['llm_model_name']} on {self.config['device']}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['llm_model_name'])
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.config['llm_model_name'],
            torch_dtype=torch.bfloat16 if self.config['device'] == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.llm_model.eval()
        print("LLM loaded successfully!")
        print("Note: Similarity search handled by llama_router module")

    def _load_scenarios(self):
        """Loads the scenarios from the scenarios file."""
        with open(self.config['scenarios_file'], 'r') as f:
            self.scenarios_data = json.load(f)
        self.all_scenarios = self.scenarios_data["scenarios"]

    def _find_best_scenario(self, user_situation, threshold=None):
        """Finds the best scenario for a given situation using llama_router."""
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD

        scenario_id, score = route(user_situation, threshold)

        if scenario_id is None:
            # If no scenario meets the threshold, use a default 'unsure' scenario
            print("No suitable scenario found. Defaulting to 'unsure_scenario'.")
            scenario_id = 'unsure_scenario'

        # Get the full scenario info from the router
        router = get_router()
        scenario = router.get_scenario_info(scenario_id)

        if not scenario:
            print(f"Warning: Could not retrieve info for scenario ID '{scenario_id}'.")
            return None, score

        return scenario, score

    def _get_factor_values(self, user_situation, scenario):
        """Gets the factor values from the LLM."""
        cpt_data = self._load_cpt(scenario['id'])
        if not cpt_data:
            return None

        factors_json = json.dumps(cpt_data['factors'], indent=2)
        prompt = self.config['factor_prompt_template'].format(
            user_situation=user_situation,
            scenario_description=scenario['description'],
            factors_json=factors_json
        )

        messages = [{"role": "user", "content": prompt}]
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.config['device'])

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        json_response = self._extract_json_from_response(response_text)
        return json_response.get('factor_values') if json_response else None

    def _load_cpt(self, scenario_id):
        """Loads the CPT for a given scenario."""
        cpt_filename = os.path.join(self.config['cpt_dir'], f"{scenario_id}.json")
        if not os.path.exists(cpt_filename):
            return None
        with open(cpt_filename, 'r') as f:
            return json.load(f)

    def _get_probabilities_for_factors(self, cpt_data, selected_factors):
        """Gets the probabilities for a given set of factors from the CPT data."""
        if not cpt_data or 'cpt' not in cpt_data or not selected_factors:
            return None

        # The CPT is a list of dictionaries. We need to find the one that matches.
        for entry in cpt_data['cpt']:
            match = True
            # Create a copy of the entry to check against, excluding the 'emotions' key
            entry_factors = entry.copy()
            entry_factors.pop('emotions', None)

            if len(entry_factors) != len(selected_factors):
                continue

            for factor_name, factor_value in selected_factors.items():
                if entry.get(factor_name) != factor_value:
                    match = False
                    break
            
            if match:
                return entry.get('emotions') # Return the emotion probabilities

        return None # No matching entry found

    def predict_choice(self, prompt):
        """Generates a response for a multiple-choice question."""
        log = {
            'prompt': prompt,
            'raw_response': '',
            'cleaned_response': ''
        }

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.config['device'])

        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=self.config.get('llm_max_new_tokens', 50),
            temperature=self.config.get('llm_temperature', 0.6),
            do_sample=False
        )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        log['raw_response'] = response_text

        if "Answer:" in response_text:
            cleaned_response = response_text.split("Answer:")[-1].strip()
        else:
            cleaned_response = response_text[len(prompt):].strip()
        
        log['cleaned_response'] = cleaned_response
        return log

    def predict_proba(self, situation):
        """Predicts emotion probabilities for a given situation."""
        log = {
            'situation': situation,
            'best_scenario_id': None,
            'best_scenario_description': None,
            'similarity_score': 0.0,
            'factor_values': None,
            'probabilities': None,
            'error': None
        }

        scenario, score = self._find_best_scenario(situation)
        log['similarity_score'] = score

        if not scenario:
            log['error'] = "No suitable scenario found (similarity score below threshold)."
            return log

        log['best_scenario_id'] = scenario.get('id')
        log['best_scenario_description'] = scenario.get('description')

        factor_values = self._get_factor_values(situation, scenario)
        log['factor_values'] = factor_values

        if not factor_values:
            log['error'] = "Failed to extract factor values from LLM."
            return log

        cpt_data = self._load_cpt(scenario['id'])
        if not cpt_data:
            log['error'] = f"CPT file not found for scenario {scenario['id']}."
            return log

        probabilities = self._get_probabilities_for_factors(cpt_data, factor_values)
        log['probabilities'] = probabilities

        if not probabilities:
            log['error'] = "Could not find matching probabilities for the extracted factor values."

        return log

    def _extract_json_from_response(self, response_text):
        """Extracts a JSON object from the model's text response."""
        try:
            if "```" in response_text:
                start_block = response_text.find("```")
                end_block = response_text.rfind("```")
                if start_block != -1 and end_block != -1 and start_block != end_block:
                    code_block = response_text[start_block + 3 : end_block]
                    json_start = code_block.find('{')
                    if json_start != -1:
                        code_block = code_block[json_start:]
                    json_str = code_block.strip()
                else:
                     raise ValueError("Malformed markdown code block.")
            else:
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
