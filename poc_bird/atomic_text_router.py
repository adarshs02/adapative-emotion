"""
Atomic text-based scenario router using HNSW indexing for efficient similarity search.
This uses atomic-scenarios.json for fair comparison with the tag-based system.
"""

import json
import os
import pickle
from typing import List, Tuple, Dict, Optional

import numpy as np
import hnswlib
import config
from vllm import LLM

from config import (
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE, EMBED_DIM,
    HNSW_INDEX_PATH, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH,
    SCENARIOS_FILE, SCENARIO_MAPPING_PATH
)

# Atomic text-based paths
ATOMIC_TEXT_INDEX_PATH = HNSW_INDEX_PATH.replace('.idx', '_atomic_text.idx')
ATOMIC_TEXT_MAPPING_PATH = HNSW_INDEX_PATH.replace('.idx', '_atomic_text_mapping.json')


class AtomicTextRouter:
    """Router that finds the best matching atomic scenario using text embeddings and HNSW indexing."""
    
    def __init__(self):
        """Initializes the router, loading the model and index."""
        self.index_path = ATOMIC_TEXT_INDEX_PATH
        self.mapping_path = ATOMIC_TEXT_MAPPING_PATH
        self.llm = None
        self.index = None
        self.scenario_mapping = None

        self._load_model()
        self._load_index()

    def _load_model(self):
        """Load the Llama model using vLLM for embedding."""
        print(f"Loading model {EMBED_MODEL_NAME} for embedding...")
        self.llm = LLM(
            model=config.EMBED_MODEL_NAME, 
            trust_remote_code=True,
            gpu_memory_utilization=0.4,  # Use a fraction of GPU memory
            task="embed",
            enforce_eager=True,  # Recommended for embedding models
        )
        print("vLLM model loaded successfully!")
    
    def _load_index(self):
        """Load the HNSW index and scenario mapping."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"Atomic text HNSW index not found at {self.index_path}. "
                f"Please run 'python build_atomic_text_index.py' first."
            )
        
        print(f"Loading atomic text HNSW index from {self.index_path}...")
        self.index = hnswlib.Index(space='cosine', dim=EMBED_DIM)
        self.index.load_index(self.index_path)
        self.index.set_ef(HNSW_EF_SEARCH)
        print("Atomic text HNSW index loaded successfully!")
        
        # Load scenario mapping
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"Atomic text scenario mapping not found at {self.mapping_path}")

        with open(self.mapping_path, 'r') as f:
            self.scenario_mapping = json.load(f)
        print(f"Loaded atomic text scenario mapping for {len(self.scenario_mapping)} scenarios")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for input text using vLLM."""
        if self.llm is None:
            raise RuntimeError("vLLM model not loaded")

        # Use vLLM's embed method
        outputs = self.llm.embed([text])
        
        # Extract the embedding from the output
        embedding = outputs[0].outputs.embedding
        
        return np.array(embedding, dtype=np.float32).reshape(1, -1)
    
    def route(self, text: str, threshold: float = None) -> Tuple[Optional[str], float]:
        """
        Find the best matching atomic scenario for the input text.
        
        Args:
            text: Input situation text
            threshold: Similarity threshold (uses config default if None)
        
        Returns:
            Tuple of (scenario_id, confidence_score)
            Returns (None, score) if no scenario meets threshold
        """
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        
        # Generate embedding for input text
        query_embedding = self.embed_text(text)
        
        # Search HNSW index
        labels, distances = self.index.knn_query(query_embedding, k=1)
        
        # Confidence is 1 - distance for cosine similarity
        confidence = 1 - distances[0][0]
        scenario_idx = labels[0][0]

        # Check threshold
        if confidence < threshold:
            return None, confidence
        
        # Get scenario ID from mapping
        scenario_info = self.scenario_mapping[str(scenario_idx)]
        scenario_id = scenario_info['id']
        
        print(f"Matched atomic scenario '{scenario_id}' with confidence {confidence:.4f}")
        
        return scenario_id, confidence
    
    def route_top_k(self, text: str, k: int = None) -> List[Dict]:
        """
        Find top-k matching atomic scenarios for the input text.
        
        Args:
            text: Input situation text
            k: Number of top scenarios to return (uses config default if None)
        
        Returns:
            List of dictionaries with 'scenario_id', 'description', and 'confidence'
        """
        if k is None:
            k = config.TOP_K_SCENARIOS
        
        # Generate embedding for input text
        query_embedding = self.embed_text(text)
        
        # Search HNSW index
        labels, distances = self.index.knn_query(query_embedding, k=k)
        
        results = []
        for i, idx in enumerate(labels[0]):
            confidence = 1 - distances[0][i]
            scenario_info = self.scenario_mapping[str(idx)]
            
            result = {
                'scenario_id': scenario_info['id'],
                'description': scenario_info['description'],
                'confidence': confidence,
                'score': confidence  # For compatibility with existing code
            }
            
            # Add scenario object for compatibility
            result['scenario'] = {
                'id': scenario_info['id'],
                'description': scenario_info['description']
            }
            
            results.append(result)
        
        return results
    
    def get_scenario_info(self, scenario_id: str) -> Optional[Dict]:
        """Get atomic scenario information by ID."""
        for scenario_info in self.scenario_mapping.values():
            if scenario_info['id'] == scenario_id:
                return scenario_info
        return None


# Global router instance (lazy-loaded)
_atomic_text_router_instance = None


def get_atomic_text_router() -> AtomicTextRouter:
    """Get the global atomic text router instance (singleton pattern)."""
    global _atomic_text_router_instance
    if _atomic_text_router_instance is None:
        _atomic_text_router_instance = AtomicTextRouter()
    return _atomic_text_router_instance


def atomic_text_route(text: str, threshold: float = None) -> Tuple[Optional[str], float]:
    """
    Convenience function for routing text to atomic scenarios using text embeddings.
    
    Args:
        text: Input situation text
        threshold: Similarity threshold
    
    Returns:
        Tuple of (scenario_id, confidence_score)
    """
    router = get_atomic_text_router()
    return router.route(text, threshold)


def atomic_text_route_top_k(text: str, k: int = None) -> List[Dict]:
    """
    Convenience function for getting top-k atomic scenario matches using text embeddings.
    
    Args:
        text: Input situation text
        k: Number of top scenarios to return
    
    Returns:
        List of scenario matches with confidence scores
    """
    router = get_atomic_text_router()
    return router.route_top_k(text, k)
