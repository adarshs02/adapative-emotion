"""
LogisticPooler: BIRD pooling formula for emotion probability calculation.

This module applies the BIRD pooling formula to combine individual factor-emotion
probabilities into a final emotion probability.
"""

import math
from typing import Dict, List, Any


class LogisticPooler:
    """
    Applies BIRD pooling formula to combine factor-emotion probabilities.
    """
    
    @staticmethod
    def pool(emotion: str, chosen_values: Dict[str, str], cpt_data: dict) -> float:
        """
        Fetch each P(emotion | factor=value) dial from cpt_data["cpt"].
        Apply the BIRD pooling formula:

            prodP = ∏ P_e|v
            prodN = ∏ (1 - P_e|v)
            return prodP / (prodP + prodN)

        If a dial is missing ⇒ use 0.50.
        
        Args:
            emotion: Target emotion to calculate probability for
            chosen_values: Dictionary mapping factor_name to chosen_value
            cpt_data: CPT data structure with factors and probability tables
            
        Returns:
            Final pooled probability for the emotion
        """
        if not chosen_values:
            return 0.50  # Neutral probability if no factors
        
        cpt_table = cpt_data.get('cpt', {})
        
        # Collect individual probabilities for this emotion
        individual_probs = []
        
        for factor_name, factor_value in chosen_values.items():
            # Create factor key for CPT lookup
            factor_key = f"{factor_name}={factor_value}"
            
            # Look up probability in CPT
            if factor_key in cpt_table and emotion in cpt_table[factor_key]:
                prob = cpt_table[factor_key][emotion]
                individual_probs.append(prob)
            else:
                # Missing dial ⇒ use neutral probability
                individual_probs.append(0.50)
        
        if not individual_probs:
            return 0.50
        
        # Apply BIRD pooling formula
        return LogisticPooler._bird_pooling_formula(individual_probs)
    
    @staticmethod
    def _bird_pooling_formula(probs: List[float]) -> float:
        """
        Apply the BIRD pooling formula to a list of probabilities.
        
        Formula:
            prodP = ∏ P_i
            prodN = ∏ (1 - P_i)
            return prodP / (prodP + prodN)
        
        Args:
            probs: List of individual probabilities
            
        Returns:
            Pooled probability
        """
        if not probs:
            return 0.50
        
        # Handle edge cases to avoid numerical issues
        probs = [max(min(p, 0.999), 0.001) for p in probs]  # Clamp to avoid 0/1
        
        # Calculate products
        prodP = 1.0
        prodN = 1.0
        
        for prob in probs:
            prodP *= prob
            prodN *= (1.0 - prob)
        
        # Apply formula
        denominator = prodP + prodN
        if denominator == 0:
            return 0.50  # Fallback for numerical edge case
        
        pooled_prob = prodP / denominator
        return pooled_prob
    
    @staticmethod
    def pool_all_emotions(chosen_values: Dict[str, str], cpt_data: dict) -> Dict[str, float]:
        """
        Pool probabilities for all emotions in the CPT data.
        
        Args:
            chosen_values: Dictionary mapping factor_name to chosen_value
            cpt_data: CPT data structure with factors and probability tables
            
        Returns:
            Dictionary mapping emotion to pooled probability
        """
        emotions = cpt_data.get('emotions', [])
        
        pooled_probs = {}
        for emotion in emotions:
            pooled_probs[emotion] = LogisticPooler.pool(emotion, chosen_values, cpt_data)
        
        return pooled_probs
    
    @staticmethod
    def pool_soft(emotion: str, factor_scores: Dict[str, Dict[str, float]], cpt_data: dict) -> float:
        """
        Soft pooling using posterior value buckets instead of hard yes/no.
        Each factor contributes softly based on its entailment scores.
        
        Args:
            emotion: Target emotion to calculate probability for
            factor_scores: Dictionary mapping factor_name to {value: score} dict
            cpt_data: CPT data structure
            
        Returns:
            Soft-pooled probability for the emotion
        """
        if not factor_scores:
            return 0.50
        
        cpt_table = cpt_data.get('cpt', {})
        
        # Collect soft-weighted probabilities for this emotion
        soft_contributions = []
        
        for factor_name, value_scores in factor_scores.items():
            # Calculate weighted average probability for this factor
            weighted_prob = 0.0
            total_weight = 0.0
            
            for factor_value, score in value_scores.items():
                factor_key = f"{factor_name}={factor_value}"
                
                if factor_key in cpt_table and emotion in cpt_table[factor_key]:
                    prob = cpt_table[factor_key][emotion]
                    weighted_prob += prob * score
                    total_weight += score
            
            if total_weight > 0:
                avg_prob = weighted_prob / total_weight
                soft_contributions.append(avg_prob)
            else:
                soft_contributions.append(0.50)  # Neutral fallback
        
        if not soft_contributions:
            return 0.50
        
        # Apply BIRD pooling to soft contributions
        return LogisticPooler._bird_pooling_formula(soft_contributions)
    
    @staticmethod
    def get_pooling_info() -> Dict[str, Any]:
        """
        Get information about the pooling method used.
        
        Returns:
            Dictionary with pooling information
        """
        return {
            'method': 'BIRD pooling formula',
            'formula': 'prodP / (prodP + prodN)',
            'missing_dial_default': 0.50,
            'description': 'Logistic pooling of individual factor-emotion probabilities'
        }
