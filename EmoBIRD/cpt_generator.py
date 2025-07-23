"""
CPT Generator for Emobird

Dynamically generates Conditional Probability Tables (CPTs) for scenarios at inference time.
"""

import json
import torch
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


class CPTGenerator:
    """
    Generates CPTs dynamically for scenarios.
    """
    
    def __init__(self, config):
        """Initialize the CPT generator."""
        self.config = config
        self.tokenizer = None
        self.model = None
        
        # Default emotions and factors
        self.default_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        
    def set_llm(self, tokenizer, model):
        """Set the LLM tokenizer and model from parent class."""
        self.tokenizer = tokenizer
        self.model = model
    
    def generate_cpt(self, scenario: Dict[str, Any], user_situation: str) -> Dict[str, Any]:
        """
        Generate a CPT for the given scenario.
        
        Args:
            scenario: Generated scenario dictionary
            user_situation: Original user situation for context
            
        Returns:
            Dictionary containing:
            - factors: List of relevant factors for this scenario
            - cpt: Conditional probability table mapping factor combinations to emotion probabilities
            - metadata: Additional information about the CPT generation
        """
        
        print(f"🎲 Generating CPT for scenario: {scenario.get('description', 'Unknown')[:50]}...")
        
        # Step 1: Generate relevant factors for this scenario
        factors = self._generate_factors(scenario, user_situation)
        
        # Step 2: Generate CPT entries for factor combinations
        cpt = self._generate_cpt_table(scenario, factors, user_situation)
        
        cpt_data = {
            'scenario_id': scenario.get('id', 'unknown'),
            'factors': factors,
            'cpt': cpt,
            'emotions': self.default_emotions,
            'metadata': {
                'generation_method': 'dynamic_llm',
                'scenario_tags': scenario.get('tags', []),
                'generated_for': user_situation[:100] + '...' if len(user_situation) > 100 else user_situation
            }
        }
        
        return cpt_data
    
    def _generate_factors(self, scenario: Dict[str, Any], user_situation: str) -> List[Dict[str, Any]]:
        """Generate relevant factors for the scenario."""
        
        prompt = self._build_factors_prompt(scenario, user_situation)
        
        if self.tokenizer and self.model:
            response = self._generate_with_llm(prompt)
            try:
                factors_data = json.loads(response)
                factors = factors_data.get('factors', [])
                
                # Validate and normalize factors
                return self._validate_factors(factors)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse factors JSON: {e}")
                return self._get_default_factors()
        else:
            return self._get_default_factors()
    
    def _build_factors_prompt(self, scenario: Dict[str, Any], user_situation: str) -> str:
        """Build prompt for factor generation."""
        
        prompt = f"""You are an emotion analysis expert. Given a scenario and user situation, identify the key psychological and contextual factors that would influence emotional outcomes.

SCENARIO: {scenario.get('description', '')}
CONTEXT: {scenario.get('context', '')}
USER SITUATION: {user_situation}

Generate 3-5 key factors that would determine emotional responses in this scenario. Each factor should have 2-4 possible values.

Respond with ONLY a JSON object in this format:
{{
  "factors": [
    {{
      "name": "factor_name",
      "description": "Brief description of what this factor represents",
      "values": ["value1", "value2", "value3"]
    }},
    ...
  ]
}}

Focus on factors that are:
- Psychologically relevant to emotional outcomes
- Observable or inferable from the situation
- Have clear, distinct value categories
- Cover different aspects (cognitive, social, situational, personal)"""

        return prompt
    
    def _generate_cpt_table(self, scenario: Dict[str, Any], factors: List[Dict[str, Any]], 
                           user_situation: str) -> Dict[str, Dict[str, float]]:
        """Generate the CPT table mapping factor combinations to emotion probabilities."""
        
        print("🔢 Generating CPT entries...")
        
        # Get all possible factor combinations
        factor_combinations = self._get_factor_combinations(factors)
        
        cpt = {}
        
        # Generate a subset of combinations to avoid exponential explosion
        max_combinations = min(len(factor_combinations), self.config.max_cpt_entries)
        selected_combinations = factor_combinations[:max_combinations]
        
        for i, combination in enumerate(selected_combinations):
            if i % 5 == 0:  # Progress indicator
                print(f"  Generated {i}/{len(selected_combinations)} CPT entries")
                
            emotion_probs = self._generate_emotion_probabilities(
                scenario, combination, factors, user_situation
            )
            
            # Create key from combination tuple
            factor_key = str(combination)
            cpt[factor_key] = emotion_probs
        
        print(f"✅ Generated {len(cpt)} CPT entries")
        return cpt
    
    def _get_factor_combinations(self, factors: List[Dict[str, Any]]) -> List[Tuple]:
        """Get all possible combinations of factor values."""
        
        if not factors:
            return [()]
        
        import itertools
        
        factor_values = [factor['values'] for factor in factors]
        combinations = list(itertools.product(*factor_values))
        
        return combinations
    
    def _generate_emotion_probabilities(self, scenario: Dict[str, Any], combination: Tuple,
                                      factors: List[Dict[str, Any]], user_situation: str) -> Dict[str, float]:
        """Generate emotion probabilities for a specific factor combination."""
        
        # Build context for this specific combination
        factor_context = []
        for i, factor in enumerate(factors):
            factor_name = factor['name']
            factor_value = combination[i] if i < len(combination) else 'unknown'
            factor_context.append(f"{factor_name}: {factor_value}")
        
        context_str = ", ".join(factor_context)
        
        prompt = f"""You are an emotion probability expert. Given a scenario and specific factor conditions, estimate the probability distribution over basic emotions.

SCENARIO: {scenario.get('description', '')}
FACTOR CONDITIONS: {context_str}
ORIGINAL SITUATION: {user_situation[:200]}

Estimate probabilities for these emotions: {', '.join(self.default_emotions)}

Respond with ONLY a JSON object:
{{
  "probabilities": {{
    "joy": 0.XX,
    "sadness": 0.XX,
    "anger": 0.XX,
    "fear": 0.XX,
    "surprise": 0.XX,
    "disgust": 0.XX
  }}
}}

Requirements:
- All probabilities must sum to 1.0
- Use psychological principles to estimate realistic distributions
- Consider how the factor conditions would influence emotional outcomes"""

        if self.tokenizer and self.model:
            response = self._generate_with_llm(prompt)
            try:
                probs_data = json.loads(response)
                probabilities = probs_data.get('probabilities', {})
                
                # Normalize probabilities to sum to 1.0
                return self._normalize_probabilities(probabilities)
                
            except json.JSONDecodeError:
                return self._get_uniform_probabilities()
        else:
            return self._get_uniform_probabilities()
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate response using the LLM."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Clean up response
        response = response.strip()
        
        # Extract JSON if wrapped in markdown
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        return response
    
    def _validate_factors(self, factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and normalize factor definitions."""
        
        validated_factors = []
        
        for factor in factors:
            if isinstance(factor, dict) and 'name' in factor and 'values' in factor:
                # Ensure proper structure
                validated_factor = {
                    'name': str(factor['name']),
                    'description': factor.get('description', ''),
                    'values': [str(v) for v in factor['values'][:4]]  # Limit to 4 values
                }
                
                # Ensure at least 2 values
                if len(validated_factor['values']) >= 2:
                    validated_factors.append(validated_factor)
        
        # Ensure we have at least some factors
        if not validated_factors:
            return self._get_default_factors()
        
        return validated_factors[:5]  # Limit to 5 factors max
    
    def _get_default_factors(self) -> List[Dict[str, Any]]:
        """Return default factors when generation fails."""
        
        return [
            {
                'name': 'intensity',
                'description': 'Emotional intensity of the situation',
                'values': ['low', 'medium', 'high']
            },
            {
                'name': 'control',
                'description': 'Level of perceived control over the situation',
                'values': ['no_control', 'some_control', 'full_control']
            },
            {
                'name': 'social_context',
                'description': 'Social setting of the situation',
                'values': ['private', 'public', 'intimate']
            }
        ]
    
    def _normalize_probabilities(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """Normalize probabilities to sum to 1.0."""
        
        # Ensure all emotions are present
        normalized = {}
        for emotion in self.default_emotions:
            normalized[emotion] = float(probabilities.get(emotion, 0.0))
        
        # Calculate sum
        total = sum(normalized.values())
        
        if total > 0:
            # Normalize
            for emotion in normalized:
                normalized[emotion] = normalized[emotion] / total
        else:
            # Fallback to uniform
            return self._get_uniform_probabilities()
        
        return normalized
    
    def _get_uniform_probabilities(self) -> Dict[str, float]:
        """Return uniform probability distribution."""
        
        uniform_prob = 1.0 / len(self.default_emotions)
        return {emotion: uniform_prob for emotion in self.default_emotions}
