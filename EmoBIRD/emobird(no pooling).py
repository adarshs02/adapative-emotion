"""
Emobird: Dynamic Emotion Analysis System

This system generates scenarios and CPTs dynamically at inference time
rather than using pre-stored scenarios and CPT files.
"""

import json
import torch
from typing import Dict, List, Any, Tuple

from scenario_generator import ScenarioGenerator
from cpt_generator import CPTGenerator
from factor_generator import FactorGenerator
from config import EmobirdConfig
from vllm_wrapper import VLLMWrapper


class Emobird:
    """
    Main Emobird inference engine that generates scenarios and CPTs dynamically.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the Emobird system."""
        self.config = EmobirdConfig(config_path)
        
        print("🐦 Initializing Emobird system...")
        
        # Load vLLM for inference first
        self._load_llm()
        
        # Initialize components
        self.scenario_generator = ScenarioGenerator(self.config)
        self.factor_generator = FactorGenerator(self.config)
        self.cpt_generator = CPTGenerator(self.config)
        
        # Set vLLM wrapper for generators
        self.scenario_generator.set_vllm(self.vllm_wrapper)
        self.factor_generator.set_vllm(self.vllm_wrapper)
        self.cpt_generator.set_vllm(self.vllm_wrapper)
        
        print("✅ Emobird system initialized successfully!")
    
    def _load_llm(self):
        """Load the vLLM wrapper for inference."""
        print(f"🚀 Loading vLLM: {self.config.llm_model_name}")
        
        # Initialize vLLM wrapper
        self.vllm_wrapper = VLLMWrapper(self.config)
        
        print("✅ vLLM loaded successfully!")
    
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
        
        # Step 1: Generate abstract/summary from user input (done in scenario generator)
        print("📋 Generating abstract...")
        abstract = self.scenario_generator._generate_abstract(user_situation)
        
        # Step 2: Generate factors from user input and abstract
        print("⚙️ Generating psychological factors...")
        factors = self.factor_generator.generate_factors(user_situation, abstract)
        
        # Step 3: Extract specific factor values for this situation
        print("🎯 Extracting factor values...")
        factor_values = self.factor_generator.extract_factor_values(
            user_situation, abstract, factors
        )
        
        # Step 4: Generate scenario from abstract
        print("📝 Generating scenario...")
        scenario = self.scenario_generator.generate_scenario(user_situation)
        
        # Check if scenario generation was successful
        if not scenario or not isinstance(scenario, dict):
            print("⚠️ Scenario generation failed, using fallback")
            scenario_description = f"Emotional situation: {user_situation}"
        else:
            scenario_description = scenario.get('description', user_situation)
        
        # Step 5: Generate CPT dynamically using the generated factors
        print("🎲 Generating CPT...")
        cpt_data = self.cpt_generator.generate_cpt_with_factors(
            scenario_description, factors
        )
        
        # Step 6: Calculate emotion probabilities
        print("🎯 Calculating emotion probabilities...")
        emotions = self._calculate_emotions(cpt_data, factor_values)
        
        # Step 7: Apply Bayesian calibration if enabled
        if self.config.use_bayesian_calibration:
            print("🔧 Applying Bayesian calibration...")
            emotions = self._apply_bayesian_calibration(emotions, factor_values)
        
        result = {
            'scenario': scenario,
            'abstract': abstract,
            'factors_definition': factors,
            'factor_values': factor_values,
            'emotions': emotions,
            'metadata': {
                'inference_method': 'dynamic_generation_with_factors',
                'model_used': self.config.llm_model_name,
                'workflow_steps': [
                    'abstract_generation',
                    'factor_generation', 
                    'factor_value_extraction',
                    'scenario_generation',
                    'cpt_generation',
                    'emotion_calculation'
                ],
                'num_factors': len(factors),
                'total_combinations': self.factor_generator._calculate_combinations(factors)
            }
        }
        
        print("✅ Analysis complete!")
        return result
    
    def _extract_factor_values(self, user_situation: str, scenario: Dict[str, Any], 
                             factors: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract factor values from user situation using LLM."""
        
        factors_json = json.dumps(factors, indent=2)
        
        # Safe extraction of scenario description - handle both dict and string cases
        if isinstance(scenario, dict):
            scenario_desc = scenario.get('description', user_situation)
        elif isinstance(scenario, str):
            scenario_desc = scenario
        else:
            scenario_desc = user_situation  # fallback
        
        prompt = f"""You are a JSON-only API. Your only function is to determine the values for a given set of factors based on a user's situation. Do not provide any explanations or text outside of the JSON object.

SITUATION:
"{user_situation}"

SCENARIO:
"{scenario_desc}"

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
        
        # Use vLLM wrapper to generate JSON response
        factor_data = self.vllm_wrapper.generate_json(prompt)
        return factor_data.get('factor_values', {})
    
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
