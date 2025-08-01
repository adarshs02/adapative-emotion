"""
FactorEntailment: LLM-based factor value entailment for emotion reasoning.

This module asks the LLM yes/no questions for every value of every factor
and returns exactly one chosen value per factor based on qualitative assessments.
"""

import json
from typing import Dict, List, Any


class FactorEntailment:
    """
    Entails factor values from stories using LLM yes/no assessments.
    """
    
    # Reuse the same qualitative scale mapping from NeutralProbabilityExtractor
    VERBAL2P = {
        'very-unlikely': 0.05,
        'unlikely': 0.25,
        'neutral': 0.50,
        'likely': 0.75,
        'very-likely': 0.95
    }
    
    def __init__(self, vllm_wrapper, factors: List[Dict[str, Any]]):
        """
        Initialize factor entailment.
        
        Args:
            vllm_wrapper: vLLM wrapper for LLM inference
            factors: List of factor definitions with 'name' and 'values'
        """
        self.vllm_wrapper = vllm_wrapper
        self.factors = factors
        
    def entail_values(self, story: str) -> Dict[str, str]:
        """
        Ask the LLM a yes/no question for every value of every factor,
        map the qualitative bucket to bool with the same scale used in
        NeutralProbabilityExtractor (likely/very-likely → True),
        then return exactly ONE chosen value per factor.
        
        Args:
            story: User's story/situation description
            
        Returns:
            Dictionary mapping factor_name to chosen_value
            Example: {"relationship": "close-friend", "fairness": "unfair", "stakes": "high"}
        """
        chosen_values = {}
        
        print(f"🔍 Entailing factor values from story...")
        
        for factor in self.factors:
            factor_name = factor['name']
            factor_values = factor.get('values', [])
            
            if not factor_values:
                continue
                
            print(f"   📊 Processing factor: {factor_name}")
            
            # Ask LLM about each possible value for this factor
            value_scores = {}
            for value in factor_values:
                score = self._assess_factor_value(story, factor_name, value)
                value_scores[value] = score
                print(f"      {value}: {score:.3f}")
            
            # Choose the value with the highest score
            if value_scores:
                chosen_value = max(value_scores.items(), key=lambda x: x[1])[0]
                chosen_values[factor_name] = chosen_value
                print(f"   ✅ Chosen: {factor_name} = {chosen_value}")
            else:
                # Fallback to first value if none scored
                chosen_values[factor_name] = factor_values[0]
                print(f"   ⚠️ Fallback: {factor_name} = {factor_values[0]}")
        
        return chosen_values
    
    def _assess_factor_value(self, story: str, factor_name: str, factor_value: str) -> float:
        """
        Assess how well a specific factor value applies to the story.
        
        Args:
            story: User's story/situation
            factor_name: Name of the factor being assessed
            factor_value: Specific value to assess
            
        Returns:
            Numerical score (0.0 to 1.0) based on LLM assessment
        """
        prompt = self._build_entailment_prompt(story, factor_name, factor_value)
        
        try:
            # Generate JSON response
            response_data = self.vllm_wrapper.generate_json(prompt)
            qualitative_rating = response_data.get('applies', 'neutral').lower().strip()
            
            # Map qualitative rating to numerical score
            score = self.VERBAL2P.get(qualitative_rating, 0.50)  # Default to neutral
            
            return score
            
        except Exception as e:
            print(f"      ❌ Error assessing {factor_name}={factor_value}: {e}")
            return 0.50  # Default to neutral score
    
    def _build_entailment_prompt(self, story: str, factor_name: str, factor_value: str) -> str:
        """
        Build prompt for assessing whether a factor value applies to the story.
        
        Args:
            story: User's story/situation
            factor_name: Name of the factor
            factor_value: Value to assess
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""TASK: Assess whether a specific factor value applies to this story.

STORY: {story}

FACTOR: {factor_name}
VALUE TO ASSESS: "{factor_value}"

INSTRUCTIONS:
Read the story carefully. Does the factor "{factor_name}" have the value "{factor_value}" in this situation?

Consider the story context and determine how likely it is that this factor value applies.

Rate using this scale:
- very-unlikely: This factor value almost certainly does not apply
- unlikely: This factor value probably does not apply  
- neutral: Unclear whether this factor value applies
- likely: This factor value probably applies
- very-likely: This factor value almost certainly applies

Respond with ONLY this JSON format:
{{
    "applies": "your_rating_here",
    "reasoning": "brief explanation of your assessment"
}}

Use exactly one of these ratings: very-unlikely, unlikely, neutral, likely, very-likely"""

        return prompt
    
    def get_factor_definitions(self) -> List[Dict[str, Any]]:
        """
        Get the factor definitions used by this entailment engine.
        
        Returns:
            List of factor definitions
        """
        return self.factors
    
    def get_scale_info(self) -> Dict[str, Any]:
        """
        Get information about the qualitative scale used.
        
        Returns:
            Dictionary with scale information
        """
        return {
            'scale': self.VERBAL2P,
            'threshold_for_true': 0.75,
            'description': 'Qualitative scale for factor value entailment'
        }
