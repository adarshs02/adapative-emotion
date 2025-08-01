"""
EmotionPredictor: End-to-end emotion prediction combining factor entailment and pooling.

This module orchestrates the complete emotion prediction pipeline:
entail values from story → pool probabilities → return top-k emotions.
"""

from typing import Dict, List, Any
from factor_entailment import FactorEntailment
from logistic_pooler import LogisticPooler


class EmotionPredictor:
    """
    End-to-end emotion prediction engine.
    """
    
    def __init__(self, entailment: FactorEntailment, pooler: LogisticPooler, cpt_data: dict):
        """
        Initialize emotion predictor.
        
        Args:
            entailment: FactorEntailment instance for factor value extraction
            pooler: LogisticPooler instance for probability combination
            cpt_data: CPT data structure with factors and probability tables
        """
        self.entailment = entailment
        self.pooler = pooler
        self.cpt_data = cpt_data
    
    def predict(self, story: str, top_k: int = 3) -> Dict[str, float]:
        """
        • entail values from story
        • pool probabilities for every emotion in cpt_data["emotions"]
        • return a dict of the top-k emotions with probs, sorted desc
        
        Args:
            story: User's story/situation description
            top_k: Number of top emotions to return
            
        Returns:
            Dictionary of top-k emotions with their probabilities, sorted descending
        """
        print(f"🎭 Predicting emotions for story (top-{top_k})...")
        print(f"📖 Story: {story[:100]}..." if len(story) > 100 else f"📖 Story: {story}")
        
        # Step 1: Entail factor values from story
        print("\n🔍 Step 1: Entailing factor values...")
        chosen_values = self.entailment.entail_values(story)
        print(f"   ✅ Entailed values: {chosen_values}")
        
        # Step 2: Pool probabilities for all emotions
        print("\n🎲 Step 2: Pooling emotion probabilities...")
        all_emotion_probs = self.pooler.pool_all_emotions(chosen_values, self.cpt_data)
        
        # Step 3: Sort and return top-k emotions
        print("\n📊 Step 3: Selecting top emotions...")
        sorted_emotions = sorted(all_emotion_probs.items(), key=lambda x: x[1], reverse=True)
        top_k_emotions = dict(sorted_emotions[:top_k])
        
        print(f"   🏆 Top-{top_k} emotions:")
        for emotion, prob in top_k_emotions.items():
            print(f"      {emotion}: {prob:.4f}")
        
        return top_k_emotions
    
    def predict_soft(self, story: str, top_k: int = 3, use_soft_pooling: bool = True) -> Dict[str, float]:
        """
        Predict emotions using soft pooling with posterior value buckets.
        Instead of hard yes/no entailment, uses the entailment scores directly.
        
        Args:
            story: User's story/situation description
            top_k: Number of top emotions to return
            use_soft_pooling: Whether to use soft pooling (if False, falls back to hard prediction)
            
        Returns:
            Dictionary of top-k emotions with their soft-pooled probabilities
        """
        if not use_soft_pooling:
            return self.predict(story, top_k)
        
        print(f"🌟 Predicting emotions with soft pooling (top-{top_k})...")
        print(f"📖 Story: {story[:100]}..." if len(story) > 100 else f"📖 Story: {story}")
        
        # Step 1: Get soft factor scores (all values, not just chosen ones)
        print("\n🔍 Step 1: Getting soft factor scores...")
        factor_scores = self._get_soft_factor_scores(story)
        
        # Step 2: Apply soft pooling for all emotions
        print("\n🎲 Step 2: Soft pooling emotion probabilities...")
        all_emotion_probs = {}
        emotions = self.cpt_data.get('emotions', [])
        
        for emotion in emotions:
            soft_prob = self.pooler.pool_soft(emotion, factor_scores, self.cpt_data)
            all_emotion_probs[emotion] = soft_prob
        
        # Step 3: Sort and return top-k emotions
        print("\n📊 Step 3: Selecting top emotions...")
        sorted_emotions = sorted(all_emotion_probs.items(), key=lambda x: x[1], reverse=True)
        top_k_emotions = dict(sorted_emotions[:top_k])
        
        print(f"   🌟 Soft top-{top_k} emotions:")
        for emotion, prob in top_k_emotions.items():
            print(f"      {emotion}: {prob:.4f}")
        
        return top_k_emotions
    
    def _get_soft_factor_scores(self, story: str) -> Dict[str, Dict[str, float]]:
        """
        Get soft scores for all factor values (not just the chosen ones).
        
        Args:
            story: User's story/situation description
            
        Returns:
            Dictionary mapping factor_name to {value: score} dict
        """
        factors = self.entailment.get_factor_definitions()
        factor_scores = {}
        
        for factor in factors:
            factor_name = factor['name']
            factor_values = factor.get('values', [])
            
            if not factor_values:
                continue
            
            print(f"   📊 Getting soft scores for factor: {factor_name}")
            
            value_scores = {}
            for value in factor_values:
                score = self.entailment._assess_factor_value(story, factor_name, value)
                value_scores[value] = score
                print(f"      {value}: {score:.3f}")
            
            factor_scores[factor_name] = value_scores
        
        return factor_scores
    
    def predict_all(self, story: str) -> Dict[str, float]:
        """
        Predict probabilities for all emotions (no top-k filtering).
        
        Args:
            story: User's story/situation description
            
        Returns:
            Dictionary of all emotions with their probabilities
        """
        print(f"🎭 Predicting all emotions...")
        
        # Entail factor values and pool probabilities
        chosen_values = self.entailment.entail_values(story)
        all_emotion_probs = self.pooler.pool_all_emotions(chosen_values, self.cpt_data)
        
        return all_emotion_probs
    
    def get_prediction_info(self) -> Dict[str, Any]:
        """
        Get information about the prediction setup.
        
        Returns:
            Dictionary with prediction configuration info
        """
        return {
            'factors': [f['name'] for f in self.entailment.get_factor_definitions()],
            'emotions': self.cpt_data.get('emotions', []),
            'num_factor_combinations': len(self.cpt_data.get('cpt', {})),
            'entailment_scale': self.entailment.get_scale_info(),
            'pooling_method': self.pooler.get_pooling_info(),
            'supports_soft_pooling': True
        }
