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
        self.vllm_wrapper = None
        
        # Default emotions and factors
        self.default_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        
    def set_vllm(self, vllm_wrapper):
        """Set the vLLM wrapper from parent class."""
        self.vllm_wrapper = vllm_wrapper
    
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
    
    def generate_cpt_with_factors(self, scenario: str, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate CPT using pre-generated factors with proven poc_bird prompt structure.
        
        Args:
            scenario: Generated scenario description
            factors: Pre-generated list of psychological factors
            
        Returns:
            Dictionary containing the CPT data
        """
        
        print(f"🎲 Generating CPT with pre-generated factors for scenario: {scenario[:50]}...")
        print(f"✅ Using {len(factors)} pre-generated factors")
        
        # Calculate total combinations
        total_combinations = 1
        for factor in factors:
            total_combinations *= len(factor['possible_values'])
        
        # Build improved prompt using proven poc_bird structure
        prompt = self._build_improved_cpt_prompt(scenario, factors, total_combinations)
        
        # Generate CPT using natural language approach (same as successful factor generation)
        if self.vllm_wrapper:
            print("🔢 Generating complete CPT with emotion probabilities...")
            # Use regular generate instead of generate_abstract (CPTs need more tokens than 64-token abstract limit)
            response_text = self.vllm_wrapper.generate(
                prompt,
                component="cpt_generator",
                interaction_type="complete_cpt_generation"
            )
            
            print(f"🔍 Raw CPT response: '{response_text}'")
            print(f"🔍 Response length: {len(response_text)} chars")
            
            # Parse CPT from natural language response
            cpt_result = self._parse_cpt_from_text(response_text, factors)
            
            if cpt_result and cpt_result.get('cpt') and len(cpt_result['cpt']) > 0:
                return {
                    'factors': cpt_result['factors'],
                    'cpt': cpt_result['cpt'],
                    'metadata': {
                        'num_factors': len(factors),
                        'num_combinations': len(cpt_result.get('cpt', [])),
                        'generation_method': 'natural_language_vllm'
                    }
                }
            else:
                print("⚠️ CPT generation failed, using fallback")
                return self._generate_fallback_cpt(factors)
        else:
            # Fallback CPT generation
            return self._generate_fallback_cpt(factors)
    
    def _parse_cpt_from_text(self, response_text: str, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse CPT from natural language LLM response."""
        
        if not response_text or len(response_text.strip()) < 20:
            return {'factors': {}, 'cpt': []}
        
        # For now, try to extract JSON if present, otherwise fallback  
        try:
            # Clean the response
            cleaned_response = self._extract_clean_json_from_text(response_text)
            
            if cleaned_response:
                import json
                parsed = json.loads(cleaned_response)
                
                if isinstance(parsed, dict) and 'factors' in parsed and 'cpt' in parsed:
                    return parsed
        except Exception as e:
            print(f"⚠️ Error parsing CPT JSON: {e}")
        
        # If JSON parsing fails, create basic structure for fallback
        return {'factors': {}, 'cpt': []}
    
    def _extract_clean_json_from_text(self, response_text: str) -> str:
        """Extract clean JSON from potentially messy LLM response."""
        
        # Look for JSON blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                return response_text[start:end].strip()
        
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                return response_text[start:end].strip()
        
        # Look for JSON object patterns
        if "{" in response_text and "}" in response_text:
            start = response_text.find("{")
            # Find the matching closing brace
            brace_count = 0
            end = -1
            for i in range(start, len(response_text)):
                if response_text[i] == "{":
                    brace_count += 1
                elif response_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end != -1:
                return response_text[start:end].strip()
        
        return ""
    
    def _build_improved_cpt_prompt(self, scenario: str, factors: List[Dict[str, Any]], total_combinations: int) -> str:
        """
        Build improved CPT prompt using proven poc_bird structure.
        
        Args:
            scenario: Scenario description
            factors: List of psychological factors
            total_combinations: Total number of factor combinations
            
        Returns:
            Formatted prompt string
        """
        
        # Build factors section
        factors_spec = []
        for i, factor in enumerate(factors, 1):
            values = factor['possible_values']
            # Limit to 2 values for binary factors (poc_bird approach)
            if len(values) > 2:
                values = values[:2]
            factors_spec.append(f'    "<factor-{i}>": ["{values[0]}", "{values[1]}"]')
        
        factors_json = ",\n".join(factors_spec)
        
        # Build example CPT entry
        example_factors = []
        for i, factor in enumerate(factors, 1):
            values = factor['possible_values']
            example_factors.append(f'"<factor-{i}>":"{values[0] if len(values) > 0 else "v1"}"')
        
        example_entry = ", ".join(example_factors)
        
        prompt = f"""JSON ONLY. Start immediately with {{
Scenario: "{scenario}"
Factors: {len(factors)} factors, {total_combinations} combinations

Return this exact JSON structure:
{{
  "factors": {{
{factors_json}
  }},
  "cpt":[
    {{{example_entry}, "emotions": {{"happy": 0.50, "sad": 0.30, "anxious": 0.20}} }},
    ... {total_combinations - 1} more rows with all factor combinations
  ]
}}

Rules:
- All {total_combinations} factor combinations required
- Emotion probabilities must sum to 1.0 in each row
- Use relevant emotions for the scenario

START WITH {{ - NO OTHER TEXT:"""
        
        return prompt
    
    def _generate_fallback_cpt(self, factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a simple fallback CPT when vLLM generation fails.
        
        Args:
            factors: List of psychological factors
            
        Returns:
            Basic CPT with uniform emotion probabilities
        """
        
        print("🔄 Generating fallback CPT with uniform probabilities...")
        
        # Generate all factor combinations
        factor_combinations = []
        
        # For simplicity, use first 2 values of each factor (binary approach)
        factor_names = []
        factor_values = []
        
        for factor in factors:
            name = factor['name']
            values = factor['possible_values'][:2]  # Limit to 2 values
            factor_names.append(name)
            factor_values.append(values)
        
        # Generate all combinations
        from itertools import product
        
        combinations = list(product(*factor_values))
        
        # Create CPT entries with uniform emotion distribution
        cpt_entries = []
        emotions = ['happy', 'sad', 'angry', 'anxious', 'excited']
        uniform_prob = 1.0 / len(emotions)
        
        for combo in combinations:
            entry = {}
            for i, factor_name in enumerate(factor_names):
                entry[factor_name] = combo[i]
            
            # Add uniform emotion probabilities
            entry['emotions'] = {emotion: uniform_prob for emotion in emotions}
            cpt_entries.append(entry)
        
        return {
            'factors': {factor['name']: factor['possible_values'][:2] for factor in factors},
            'cpt': cpt_entries,
            'metadata': {
                'num_factors': len(factors),
                'num_combinations': len(cpt_entries),
                'generation_method': 'fallback_uniform'
            }
        }
    
    def _generate_factors(self, scenario: Dict[str, Any], user_situation: str) -> List[Dict[str, Any]]:
        """Generate relevant factors for the scenario."""
        
        prompt = self._build_factors_prompt(scenario, user_situation)
        
        if self.vllm_wrapper:
            factors_data = self.vllm_wrapper.generate_json(
                prompt, 
                component="cpt_generator", 
                interaction_type="factor_generation"
            )
            factors = factors_data.get('factors', [])
            
            # Validate and normalize factors
            return self._validate_factors(factors)
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
    "joy": 0.0,
    "sadness": 0.0,
    "anger": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "disgust": 0.0
  }}
}}

Requirements:
- All probabilities must sum to 1.0
- Use psychological principles to estimate realistic distributions
- Consider how the factor conditions would influence emotional outcomes"""

        if self.vllm_wrapper:
            probs_data = self.vllm_wrapper.generate_json(
                prompt, 
                component="cpt_generator", 
                interaction_type="emotion_probability_generation"
            )
            probabilities = probs_data.get('probabilities', {})
            
            # Normalize probabilities to sum to 1.0
            return self._normalize_probabilities(probabilities)
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
