"""
Neutral Probability Extractor: Generate neutral conditional probabilities using qualitative LLM assessments.

This module prompts an LLM to assess how strongly different factors indicate specific emotions
in neutral contexts, using a qualitative scale that gets mapped to numerical probabilities.
"""

import json
from typing import Dict, List, Any, Tuple
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
        Extract neutral probability for a single (factor_value, emotion) pair.
        
        Args:
            factor_name: Name of the psychological factor
            factor_value: Specific value of the factor
            emotion: Target emotion
            
        Returns:
            Numerical probability (0.0 to 1.0)
        """
        prompt = self._build_neutral_assessment_prompt(factor_name, factor_value, emotion)
        
        try:
            # Generate JSON response
            response_data = self.vllm_wrapper.generate_json(prompt)
            qualitative_rating = response_data.get('rating', 'neutral').lower().strip()
            
            # Map qualitative rating to numerical probability
            probability = self.PROBABILITY_SCALE.get(qualitative_rating, 0.50)  # Default to neutral
            
            print(f"      {factor_name}={factor_value} → {emotion}: {qualitative_rating} ({probability})")
            return probability
            
        except Exception as e:
            print(f"      ❌ Error extracting probability for {factor_name}={factor_value} → {emotion}: {e}")
            return 0.50  # Default to neutral probability
    
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
            # Create combination key
            combo_key = str(tuple(combination.values()))
            
            # Calculate emotion probabilities for this combination
            emotion_probs = {}
            
            for emotion in all_emotions:
                # Aggregate probabilities across factors in the combination
                factor_contributions = []
                
                for factor_name, factor_value in combination.items():
                    factor_key = f"{factor_name}={factor_value}"
                    if factor_key in probabilities and emotion in probabilities[factor_key]:
                        factor_contributions.append(probabilities[factor_key][emotion])
                
                # Use average of contributing factors, or neutral if no contributions
                if factor_contributions:
                    emotion_probs[emotion] = sum(factor_contributions) / len(factor_contributions)
                else:
                    emotion_probs[emotion] = 0.50  # Neutral probability
            
            # Normalize probabilities to sum to 1.0
            total_prob = sum(emotion_probs.values())
            if total_prob > 0:
                for emotion in emotion_probs:
                    emotion_probs[emotion] /= total_prob
            
            cpt_table[combo_key] = emotion_probs
        
        cpt_data = {
            'factors': factors,
            'emotions': all_emotions,
            'cpt': cpt_table,
            'metadata': {
                'method': 'neutral_probability_extraction',
                'num_combinations': len(cpt_table),
                'num_factors': len(factors),
                'num_emotions': len(all_emotions)
            }
        }
        
        print(f"   ✅ Built CPT with {len(cpt_table)} factor combinations and {len(all_emotions)} emotions")
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
        factor_value_lists = [f.get('values', ['']) for f in factors]
        
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
