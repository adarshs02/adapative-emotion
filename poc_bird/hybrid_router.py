"""
Hybrid scenario router that combines tag-based and text-based search for improved accuracy.
Uses both tag embeddings and text embeddings, then combines results with weighted scoring.
"""

import json
import os
import numpy as np
from typing import List, Tuple, Dict, Optional

import config
from tag_router import get_tag_router
from atomic_text_router import get_atomic_text_router


class HybridRouter:
    """Router that combines tag-based and text-based scenario matching."""
    
    def __init__(self, tag_weight: float = 0.5, text_weight: float = 0.5):
        """
        Initialize the hybrid router.
        
        Args:
            tag_weight: Weight for tag-based scores (0.0 to 1.0)
            text_weight: Weight for text-based scores (0.0 to 1.0)
        """
        self.tag_weight = tag_weight
        self.text_weight = text_weight
        
        # Ensure weights sum to 1.0
        total_weight = tag_weight + text_weight
        if total_weight > 0:
            self.tag_weight = tag_weight / total_weight
            self.text_weight = text_weight / total_weight
        else:
            self.tag_weight = 0.5
            self.text_weight = 0.5
        
        print(f"Hybrid router initialized with tag_weight={self.tag_weight:.2f}, text_weight={self.text_weight:.2f}")
        
        # Initialize both routers
        self.tag_router = get_tag_router()
        self.text_router = get_atomic_text_router()
    
    def route_top_k(self, text: str, k: int = None, fusion_method: str = "weighted_sum") -> List[Dict]:
        """
        Find top-k matching scenarios using hybrid tag+text approach.
        
        Args:
            text: Input situation text
            k: Number of top scenarios to return
            fusion_method: How to combine scores ("weighted_sum", "max", "min", "rank_fusion")
        
        Returns:
            List of scenario matches with combined confidence scores
        """
        if k is None:
            k = config.TOP_K_SCENARIOS
        
        # Get results from both systems (get more than k to allow for better fusion)
        search_k = min(k * 3, 50)  # Search more scenarios for better fusion
        
        print(f"Getting top-{search_k} from tag system...")
        tag_results = self.tag_router.route_top_k(text, k=search_k)
        
        print(f"Getting top-{search_k} from text system...")
        text_results = self.text_router.route_top_k(text, k=search_k)
        
        # Store last generated tags for benchmark access
        self._last_generated_tags = getattr(self.tag_router, '_last_generated_tags', [])
        
        # Combine results using specified fusion method
        if fusion_method == "weighted_sum":
            combined_results = self._weighted_sum_fusion(tag_results, text_results)
        elif fusion_method == "max":
            combined_results = self._max_fusion(tag_results, text_results)
        elif fusion_method == "min":
            combined_results = self._min_fusion(tag_results, text_results)
        elif fusion_method == "rank_fusion":
            combined_results = self._rank_fusion(tag_results, text_results)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Sort by combined confidence and return top-k
        combined_results.sort(key=lambda x: x['confidence'], reverse=True)
        return combined_results[:k]
    
    def _weighted_sum_fusion(self, tag_results: List[Dict], text_results: List[Dict]) -> List[Dict]:
        """Combine results using weighted sum of confidence scores."""
        scenario_scores = {}
        
        # Process tag results
        for result in tag_results:
            scenario_id = result['scenario_id']
            tag_score = result['confidence']
            
            if scenario_id not in scenario_scores:
                scenario_scores[scenario_id] = {
                    'scenario_id': scenario_id,
                    'description': result['description'],
                    'tag_score': 0.0,
                    'text_score': 0.0,
                    'tag_rank': len(tag_results),
                    'text_rank': len(text_results),
                    'tags': result.get('tags', []),
                    'scenario': result.get('scenario', {})
                }
            
            scenario_scores[scenario_id]['tag_score'] = tag_score
            scenario_scores[scenario_id]['tag_rank'] = tag_results.index(result)
        
        # Process text results
        for result in text_results:
            scenario_id = result['scenario_id']
            text_score = result['confidence']
            
            if scenario_id not in scenario_scores:
                scenario_scores[scenario_id] = {
                    'scenario_id': scenario_id,
                    'description': result['description'],
                    'tag_score': 0.0,
                    'text_score': 0.0,
                    'tag_rank': len(tag_results),
                    'text_rank': len(text_results),
                    'tags': result.get('tags', []),
                    'scenario': result.get('scenario', {})
                }
            
            scenario_scores[scenario_id]['text_score'] = text_score
            scenario_scores[scenario_id]['text_rank'] = text_results.index(result)
        
        # Calculate combined scores
        combined_results = []
        for scenario_id, scores in scenario_scores.items():
            combined_confidence = (
                self.tag_weight * scores['tag_score'] + 
                self.text_weight * scores['text_score']
            )
            
            result = {
                'scenario_id': scenario_id,
                'description': scores['description'],
                'confidence': combined_confidence,
                'score': combined_confidence,  # For compatibility
                'tag_score': scores['tag_score'],
                'text_score': scores['text_score'],
                'tag_rank': scores['tag_rank'],
                'text_rank': scores['text_rank'],
                'fusion_method': 'weighted_sum',
                'tags': scores['tags'],
                'scenario': scores['scenario']
            }
            
            combined_results.append(result)
        
        return combined_results
    
    def _max_fusion(self, tag_results: List[Dict], text_results: List[Dict]) -> List[Dict]:
        """Combine results using maximum confidence score."""
        scenario_scores = {}
        
        # Process both result sets
        for results, score_key in [(tag_results, 'tag'), (text_results, 'text')]:
            for result in results:
                scenario_id = result['scenario_id']
                score = result['confidence']
                
                if scenario_id not in scenario_scores:
                    scenario_scores[scenario_id] = {
                        'scenario_id': scenario_id,
                        'description': result['description'],
                        'max_score': score,
                        'tag_score': 0.0,
                        'text_score': 0.0,
                        'tags': result.get('tags', []),
                        'scenario': result.get('scenario', {})
                    }
                else:
                    scenario_scores[scenario_id]['max_score'] = max(
                        scenario_scores[scenario_id]['max_score'], score
                    )
                
                scenario_scores[scenario_id][f'{score_key}_score'] = score
        
        # Create combined results
        combined_results = []
        for scenario_id, scores in scenario_scores.items():
            result = {
                'scenario_id': scenario_id,
                'description': scores['description'],
                'confidence': scores['max_score'],
                'score': scores['max_score'],
                'tag_score': scores['tag_score'],
                'text_score': scores['text_score'],
                'fusion_method': 'max',
                'tags': scores['tags'],
                'scenario': scores['scenario']
            }
            combined_results.append(result)
        
        return combined_results
    
    def _min_fusion(self, tag_results: List[Dict], text_results: List[Dict]) -> List[Dict]:
        """Combine results using minimum confidence score (conservative approach)."""
        # Only include scenarios that appear in both result sets
        tag_scenarios = {r['scenario_id']: r for r in tag_results}
        text_scenarios = {r['scenario_id']: r for r in text_results}
        
        combined_results = []
        for scenario_id in set(tag_scenarios.keys()) & set(text_scenarios.keys()):
            tag_result = tag_scenarios[scenario_id]
            text_result = text_scenarios[scenario_id]
            
            min_confidence = min(tag_result['confidence'], text_result['confidence'])
            
            result = {
                'scenario_id': scenario_id,
                'description': tag_result['description'],
                'confidence': min_confidence,
                'score': min_confidence,
                'tag_score': tag_result['confidence'],
                'text_score': text_result['confidence'],
                'fusion_method': 'min',
                'tags': tag_result.get('tags', []),
                'scenario': tag_result.get('scenario', {})
            }
            combined_results.append(result)
        
        return combined_results
    
    def _rank_fusion(self, tag_results: List[Dict], text_results: List[Dict]) -> List[Dict]:
        """Combine results using reciprocal rank fusion (RRF)."""
        scenario_scores = {}
        
        # Process tag results
        for rank, result in enumerate(tag_results):
            scenario_id = result['scenario_id']
            rrf_score = 1.0 / (rank + 1)  # Reciprocal rank
            
            if scenario_id not in scenario_scores:
                scenario_scores[scenario_id] = {
                    'scenario_id': scenario_id,
                    'description': result['description'],
                    'rrf_score': 0.0,
                    'tag_score': result['confidence'],
                    'text_score': 0.0,
                    'tag_rank': rank,
                    'text_rank': len(text_results),
                    'tags': result.get('tags', []),
                    'scenario': result.get('scenario', {})
                }
            
            scenario_scores[scenario_id]['rrf_score'] += self.tag_weight * rrf_score
            scenario_scores[scenario_id]['tag_score'] = result['confidence']
            scenario_scores[scenario_id]['tag_rank'] = rank
        
        # Process text results
        for rank, result in enumerate(text_results):
            scenario_id = result['scenario_id']
            rrf_score = 1.0 / (rank + 1)  # Reciprocal rank
            
            if scenario_id not in scenario_scores:
                scenario_scores[scenario_id] = {
                    'scenario_id': scenario_id,
                    'description': result['description'],
                    'rrf_score': 0.0,
                    'tag_score': 0.0,
                    'text_score': result['confidence'],
                    'tag_rank': len(tag_results),
                    'text_rank': rank,
                    'tags': result.get('tags', []),
                    'scenario': result.get('scenario', {})
                }
            
            scenario_scores[scenario_id]['rrf_score'] += self.text_weight * rrf_score
            scenario_scores[scenario_id]['text_score'] = result['confidence']
            scenario_scores[scenario_id]['text_rank'] = rank
        
        # Create combined results
        combined_results = []
        for scenario_id, scores in scenario_scores.items():
            result = {
                'scenario_id': scenario_id,
                'description': scores['description'],
                'confidence': scores['rrf_score'],
                'score': scores['rrf_score'],
                'tag_score': scores['tag_score'],
                'text_score': scores['text_score'],
                'tag_rank': scores['tag_rank'],
                'text_rank': scores['text_rank'],
                'fusion_method': 'rank_fusion',
                'tags': scores['tags'],
                'scenario': scores['scenario']
            }
            combined_results.append(result)
        
        return combined_results


# Global hybrid router instance (lazy-loaded)
_hybrid_router_instance = None


def get_hybrid_router(tag_weight: float = 0.5, text_weight: float = 0.5) -> HybridRouter:
    """Get the global hybrid router instance (singleton pattern)."""
    global _hybrid_router_instance
    if _hybrid_router_instance is None:
        _hybrid_router_instance = HybridRouter(tag_weight, text_weight)
    return _hybrid_router_instance


def hybrid_route_top_k(text: str, k: int = None, fusion_method: str = "weighted_sum", 
                      tag_weight: float = 0.5, text_weight: float = 0.5) -> List[Dict]:
    """
    Convenience function for hybrid routing.
    
    Args:
        text: Input situation text
        k: Number of top scenarios to return
        fusion_method: How to combine scores ("weighted_sum", "max", "min", "rank_fusion")
        tag_weight: Weight for tag-based scores
        text_weight: Weight for text-based scores
    
    Returns:
        List of scenario matches with combined confidence scores
    """
    router = get_hybrid_router(tag_weight, text_weight)
    return router.route_top_k(text, k, fusion_method)
