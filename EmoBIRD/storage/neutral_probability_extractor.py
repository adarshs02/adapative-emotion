"""
Neutral Probability Extractor: Generate neutral conditional probabilities using qualitative LLM assessments.

This module prompts an LLM to assess how strongly different factors indicate specific emotions
in neutral contexts, using a qualitative scale that gets mapped to numerical probabilities.
"""

import json
import torch
import random
from typing import Dict, List, Any, Tuple
from utils import norm_key, pool_logistic, validate_rating, RATING_SCALE
from dial_cache import save_cpt
from config import EmobirdConfig


class NeutralProbabilityExtractor:
    """
    Extracts neutral conditional probabilities for (factor, emotion) pairs using qualitative LLM assessments.
    """
    
    # Qualitative scale mapping to numerical probabilities
    PROBABILITY_SCALE = {
        'very-unlikely': 0.05,
        'unlikely': 0.25,
        'neutral': 0.50,
        'likely': 0.75,
        'very-likely': 0.95
    }
    
    def __init__(self, config: EmobirdConfig):
        """Initialize the neutral probability extractor."""
        self.config = config
        self.vllm_wrapper = None
        
    def set_vllm(self, vllm_wrapper):
        """Set the vLLM wrapper for inference."""
        self.vllm_wrapper = vllm_wrapper
        
    def extract_neutral_probabilities(self, factors: List[Dict[str, Any]], 
                                    emotions: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Extract neutral conditional probabilities for all (factor, emotion) pairs.
        
        Args:
            factors: List of factor dictionaries with 'name' and 'values' keys
            emotions: List of emotion strings
            
        Returns:
            Nested dictionary: {factor_name: {emotion: probability}}
        """
        if not self.vllm_wrapper:
            raise ValueError("vLLM wrapper not set. Call set_vllm() first.")
            
        probabilities = {}
        
        print(f"🎯 Extracting neutral probabilities for {len(factors)} factors and {len(emotions)} emotions...")
        
        for factor in factors:
            factor_name = factor['name']
            factor_values = factor.get('possible_values', factor.get('values', []))
            
            print(f"   📊 Processing factor: {factor_name} (values: {factor_values})")
            
            # Skip factor if no values found
            if not factor_values:
                print(f"   ⚠️ No values found for factor {factor_name}, skipping")
                continue
                
            probabilities[factor_name] = {}
            
            # For each factor value, assess against each emotion
            for factor_value in factor_values:
                for emotion in emotions:
                    prob = self._extract_single_probability(factor_name, factor_value, emotion)
                    
                    # Create a composite key for factor_value
                    factor_key = f"{factor_name}={factor_value}"
                    if factor_key not in probabilities:
                        probabilities[factor_key] = {}
                    
                    probabilities[factor_key][emotion] = prob
                    
        return probabilities
    
    def _extract_single_probability(self, factor_name: str, factor_value: str, emotion: str) -> float:
        """
        Extract neutral probability for a single (factor_value, emotion) pair with strict validation.
        
        Args:
            factor_name: Name of the psychological factor
            factor_value: Specific value of the factor
            emotion: Target emotion
            
        Returns:
            Numerical probability (0.0 to 1.0)
        """
        prompt = self._build_neutral_assessment_prompt(factor_name, factor_value, emotion)
        
        # Define JSON schema for validation
        schema = {
            "required": ["rating"],
            "properties": {
                "rating": {"type": "string"},
                "reasoning": {"type": "string"}
            }
        }
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                print(f"      🔄 Attempt {attempt + 1}: Calling json_call for {factor_name}={factor_value} → {emotion}")
                response_data = self.vllm_wrapper.json_call(
                    prompt=prompt,
                    schema=schema,
                    component="neutral_probability_extractor",
                    interaction_type=f"probability_assessment_attempt_{attempt+1}",
                    max_retries=1  # Let json_call handle its own retries
                )
                
                print(f"      📋 Raw JSON response: {response_data}")
                raw_rating = response_data.get('rating', 'neutral')
                print(f"      🏷️ Extracted rating: '{raw_rating}'")
                
                validated_rating = validate_rating(raw_rating)
                probability = RATING_SCALE[validated_rating]
                
                print(f"      ✅ Successfully extracted: {raw_rating} → {validated_rating} → {probability}")
                return probability
                
            except ValueError as e:
                if "Illegal rating" in str(e):
                    print(f"      ⚠️ Invalid rating on attempt {attempt + 1}: {e}")
                    if attempt < max_retries:
                        # Make prompt stricter for retry
                        prompt = self._build_stricter_assessment_prompt(factor_name, factor_value, emotion)
                        continue
                    else:
                        print(f"      ⚠️ All rating validation attempts failed, falling back to neutral")
                        return 0.50
                else:
                    # Other validation error, continue to next attempt
                    print(f"      ⚠️ JSON validation error on attempt {attempt + 1}: {e}")
                    if attempt >= max_retries:
                        print(f"      ⚠️ All attempts failed, falling back to neutral")
                        return 0.50
                    continue
                    
            except Exception as e:
                print(f"      ❌ Unexpected error on attempt {attempt + 1}: {e}")
                if attempt >= max_retries:
                    print(f"      ❌ All attempts failed, falling back to neutral")
                    return 0.50
                continue
        
        # Should never reach here, but just in case
        return 0.50
    
    def _build_neutral_assessment_prompt(self, factor_name: str, factor_value: str, emotion: str) -> str:
        """
        Build prompt for assessing neutral probability of a (factor_value, emotion) pair.
        
        Args:
            factor_name: Name of the psychological factor
            factor_value: Specific value of the factor
            emotion: Target emotion
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""TASK: Assess how strongly a neutral situation with a specific factor indicates an emotion.

FACTOR: {factor_name} = "{factor_value}"
EMOTION: {emotion.upper()}

INSTRUCTIONS:
Imagine a completely neutral situation where the {factor_name} is "{factor_value}".
How strongly does having this factor value, BY ITSELF, indicate that someone would feel {emotion.upper()}?

Consider ONLY the factor value, not any specific dramatic scenario. Think about neutral, everyday contexts.

Rate using this scale:
- very-unlikely: This factor almost never indicates this emotion
- unlikely: This factor rarely indicates this emotion  
- neutral: This factor has no particular relationship to this emotion
- likely: This factor often indicates this emotion
- very-likely: This factor almost always indicates this emotion

Respond with ONLY this JSON format:
{{
    "rating": "your_rating_here",
    "reasoning": "brief explanation of your assessment"
}}

Use exactly one of these ratings: very-unlikely, unlikely, neutral, likely, very-likely"""

        return prompt
    
    def _build_stricter_assessment_prompt(self, factor_name: str, factor_value: str, emotion: str) -> str:
        """
        Build a stricter prompt for retry attempts with more explicit instructions.
        """
        prompt = f"""CRITICAL: You must respond with EXACTLY the specified JSON format and rating scale.

TASK: Assess how strongly a neutral situation with a specific factor indicates an emotion.

FACTOR: {factor_name} = "{factor_value}"
EMOTION: {emotion.upper()}

INSTRUCTIONS:
Imagine a completely neutral situation where the {factor_name} is "{factor_value}".
How strongly does having this factor value, BY ITSELF, indicate that someone would feel {emotion.upper()}?

Consider ONLY the factor value, not any specific dramatic scenario. Think about neutral, everyday contexts.

You MUST use EXACTLY one of these five ratings:
- very-unlikely
- unlikely  
- neutral
- likely
- very-likely

No other rating words are allowed. Do not use "low", "high", "moderate", or any other terms.

Respond with ONLY this JSON format:
{{
    "rating": "your_exact_rating_here",
    "reasoning": "brief explanation of your assessment"
}}

EXAMPLE RESPONSE:
{{
    "rating": "likely",
    "reasoning": "high stress often contributes to anxiety in neutral situations"
}}

Remember: Use EXACTLY one of: very-unlikely, unlikely, neutral, likely, very-likely"""

        return prompt
    
    def build_cpt_from_probabilities(self, probabilities: Dict[str, Dict[str, float]], 
                                   factors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a CPT table from extracted neutral probabilities.
        
        Args:
            probabilities: Nested dict of {factor_key: {emotion: probability}}
            factors: List of factor definitions
            
        Returns:
            CPT data structure compatible with existing pipeline
        """
        print("🎲 Building CPT from neutral probabilities...")
        
        # Extract all emotions from the probabilities
        all_emotions = set()
        for factor_probs in probabilities.values():
            all_emotions.update(factor_probs.keys())
        all_emotions = sorted(list(all_emotions))
        
        cpt_table = {}
        
        # Generate all possible factor combinations
        factor_names = [f['name'] for f in factors]
        factor_combinations = self._generate_factor_combinations(factors)
        
        for combination in factor_combinations:
            # Create normalized combination key
            combo_parts = []
            for factor_name, factor_value in combination.items():
                combo_parts.append(norm_key(factor_name, factor_value))
            combo_key = "|".join(sorted(combo_parts))  # Sort for consistency
            
            # Calculate emotion probabilities for this combination
            emotion_probs = {}
            
            for emotion in all_emotions:
                # Collect probabilities across factors in the combination for BIRD pooling
                factor_contributions = []
                
                for factor_name, factor_value in combination.items():
                    factor_key = norm_key(factor_name, factor_value)
                    if factor_key in probabilities and emotion in probabilities[factor_key]:
                        factor_contributions.append(probabilities[factor_key][emotion])
                
                # Use logistic pooling (BIRD formula) instead of averaging
                if factor_contributions:
                    emotion_probs[emotion] = pool_logistic(factor_contributions)
                else:
                    emotion_probs[emotion] = 0.50  # Neutral probability
            
            # No normalization needed for logistic pooling - each emotion is independent
            cpt_table[combo_key] = emotion_probs
        
        cpt_data = {
            'factors': factors,
            'emotions': all_emotions,
            'combinations': cpt_table,  # Renamed from 'cpt' to 'combinations' for clarity
            'metadata': {
                'method': 'neutral_probability_extraction_with_logistic_pooling',
                'num_combinations': len(cpt_table),
                'num_factors': len(factors),
                'num_emotions': len(all_emotions),
                'pooling_method': 'logistic_bird_formula'
            }
        }
        
        print(f"   ✅ Built CPT with {len(cpt_table)} factor combinations and {len(all_emotions)} emotions")
        
        # Save CPT to cache for future use
        try:
            save_cpt(cpt_data)
            print(f"   💾 CPT saved to cache successfully")
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to save CPT to cache: {e}")
            # Continue without failing - caching is not critical for functionality
        
        return cpt_data
    
    def _generate_factor_combinations(self, factors: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Generate all possible combinations of factor values.
        
        Args:
            factors: List of factor definitions with 'name' and 'values'
            
        Returns:
            List of dictionaries mapping factor_name to factor_value
        """
        import itertools
        
        if not factors:
            return [{}]
        
        factor_names = [f['name'] for f in factors]
        # Handle both 'values' and 'possible_values' field names
        factor_value_lists = []
        for f in factors:
            values = f.get('values', f.get('possible_values', ['']))
            if not values or values == ['']:  # Fallback if no values found
                print(f"⚠️ Warning: Factor '{f['name']}' has no values, using default ['unknown']")
                values = ['unknown']
            factor_value_lists.append(values)
            print(f"📊 Factor '{f['name']}' has values: {values}")
        
        combinations = []
        for combo in itertools.product(*factor_value_lists):
            combination = dict(zip(factor_names, combo))
            combinations.append(combination)
        
        return combinations
    
    def get_probability_scale_info(self) -> Dict[str, Any]:
        """
        Get information about the probability scale used.
        
        Returns:
            Dictionary with scale information
        """
        return {
            'scale': self.PROBABILITY_SCALE,
            'scale_order': list(self.PROBABILITY_SCALE.keys()),
            'description': 'Qualitative scale for neutral probability assessment',
            'range': [min(self.PROBABILITY_SCALE.values()), max(self.PROBABILITY_SCALE.values())]
        }
