"""
Emobird: Dynamic Emotion Analysis System

This system generates scenarios and CPTs dynamically at inference time
rather than using pre-stored scenarios and CPT files.
"""

import json
import torch
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from scenario_generator import ScenarioGenerator
from cpt_generator import CPTGenerator
from config import EmobirdConfig


class Emobird:
    """
    Main Emobird inference engine that generates scenarios and CPTs dynamically.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Emobird system."""
        self.config = EmobirdConfig(config_path)
        
        print("🐦 Initializing Emobird system...")
        
        # Load LLM for inference first
        self._load_llm()
        
        # Initialize components
        self.scenario_generator = ScenarioGenerator(self.config)
        self.cpt_generator = CPTGenerator(self.config)
        
        # Set LLM for generators
        self.scenario_generator.set_llm(self.tokenizer, self.model)
        self.cpt_generator.set_llm(self.tokenizer, self.model)
        
        print("✅ Emobird system initialized successfully!")
    
    def _load_llm(self):
        """Load the language model for inference."""
        print(f"🚀 Loading LLM: {self.config.llm_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        print("✅ LLM loaded successfully!")
    
    def analyze_emotion(self, user_situation: str) -> Dict[str, Any]:
        """
        Main inference method: analyze emotion for a given user situation.
        
        Args:
            user_situation: User's description of their situation
            
        Returns:
            Dictionary containing:
            - scenario: Generated scenario description
            - factors: Identified factors and their values
            - emotions: Emotion probability distribution
            - metadata: Additional information about the inference
        """
        print(f"\n🔍 Analyzing situation: '{user_situation[:100]}...'")
        
        # Step 1: Generate scenario dynamically from user input
        print("📝 Generating scenario...")
        scenario = self.scenario_generator.generate_scenario(user_situation)
        
        # Step 2: Generate CPT dynamically for this scenario
        print("🎲 Generating CPT...")
        cpt_data = self.cpt_generator.generate_cpt(scenario, user_situation)
        
        # Step 3: Extract factor values from user situation
        print("⚙️ Extracting factor values...")
        factor_values = self._extract_factor_values(
            user_situation, scenario, cpt_data['factors']
        )
        
        # Step 4: Calculate emotion probabilities
        print("🎯 Calculating emotion probabilities...")
        emotions = self._calculate_emotions(cpt_data, factor_values)
        
        # Step 5: Apply Bayesian calibration if enabled
        if self.config.use_bayesian_calibration:
            print("🔧 Applying Bayesian calibration...")
            emotions = self._apply_bayesian_calibration(emotions, factor_values)
        
        result = {
            'scenario': scenario,
            'factors': factor_values,
            'emotions': emotions,
            'metadata': {
                'cpt_factors': cpt_data['factors'],
                'inference_method': 'dynamic_generation',
                'model_used': self.config.llm_model_name
            }
        }
        
        print("✅ Analysis complete!")
        return result
    
    def _extract_factor_values(self, user_situation: str, scenario: Dict[str, Any], 
                             factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract factor values from user situation using LLM."""
        
        factors_json = json.dumps(factors, indent=2)
        
        prompt = f"""You are a JSON-only API. Your only function is to determine the values for a given set of factors based on a user's situation. Do not provide any explanations or text outside of the JSON object.

SITUATION:
"{user_situation}"

SCENARIO:
"{scenario.get('description', '')}"

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
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Extract JSON from response
        try:
            # Handle potential markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()
            
            parsed = json.loads(response)
            return parsed.get('factor_values', {})
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse factor values: {e}")
            print(f"Raw response: {response}")
            return {}
    
    def _calculate_emotions(self, cpt_data: Dict[str, Any], 
                          factor_values: Dict[str, str]) -> Dict[str, float]:
        """Calculate emotion probabilities from CPT and factor values."""
        
        # Create factor tuple for lookup
        factor_names = [f['name'] for f in cpt_data['factors']]
        factor_tuple = tuple(factor_values.get(name, '') for name in factor_names)
        
        # Look up in CPT
        cpt_table = cpt_data.get('cpt', {})
        factor_key = str(factor_tuple)
        
        if factor_key in cpt_table:
            return cpt_table[factor_key]
        else:
            # Fallback: return uniform distribution
            emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            uniform_prob = 1.0 / len(emotions)
            return {emotion: uniform_prob for emotion in emotions}
    
    def _apply_bayesian_calibration(self, emotions: Dict[str, float], 
                                  factor_values: Dict[str, str]) -> Dict[str, float]:
        """Apply Bayesian calibration to emotion probabilities."""
        # Placeholder for Bayesian calibration implementation
        # This would implement the BIRD upgrade mentioned in memory
        return emotions
    
    def batch_analyze(self, situations: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple situations in batch."""
        results = []
        for i, situation in enumerate(situations):
            print(f"\n📦 Processing batch item {i+1}/{len(situations)}")
            result = self.analyze_emotion(situation)
            results.append(result)
        return results


def main():
    """Example usage of the Emobird system."""
    # Initialize Emobird
    emobird = Emobird()
    
    # Example situation
    user_situation = input("\nPlease describe your situation: ")
    
    # Analyze emotion
    result = emobird.analyze_emotion(user_situation)
    
    # Display results
    print(f"\n🎭 Generated Scenario: {result['scenario'].get('description', 'N/A')}")
    print(f"\n⚙️ Factor Values:")
    for factor, value in result['factors'].items():
        print(f"  - {factor}: {value}")
    
    print(f"\n😊 Emotion Probabilities:")
    sorted_emotions = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_emotions:
        print(f"  - {emotion}: {prob:.3f}")


if __name__ == "__main__":
    main()
