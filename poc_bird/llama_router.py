"""
Llama-based scenario router using HNSW indexing for efficient similarity search.
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


class LlamaRouter:
    """Router that finds the best matching scenario using Llama embeddings and HNSW indexing."""
    
    def __init__(self):
        """Initializes the router, loading the model and index."""
        self.index_path = HNSW_INDEX_PATH
        self.mapping_path = SCENARIO_MAPPING_PATH
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
                f"HNSW index not found at {self.index_path}. "
                f"Please run 'python build_index.py' first."
            )
        
        print(f"Loading HNSW index from {self.index_path}...")
        self.index = hnswlib.Index(space='cosine', dim=EMBED_DIM)
        self.index.load_index(self.index_path)
        self.index.set_ef(HNSW_EF_SEARCH)
        print("HNSW index loaded successfully!")
        
        # Load scenario mapping
        if not os.path.exists(SCENARIO_MAPPING_PATH):
            raise FileNotFoundError(f"Scenario mapping not found at {SCENARIO_MAPPING_PATH}")

        with open(SCENARIO_MAPPING_PATH, 'r') as f:
            self.scenario_mapping = json.load(f)
        print(f"Loaded scenario mapping for {len(self.scenario_mapping)} scenarios")
    
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
        Find the best matching scenario for the input text.
        
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
        
        return scenario_id, confidence
    
    def route_top_k(self, text: str, k: int = None) -> List[Dict]:
        """
        Find top-k matching scenarios for the input text.
        
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
            
            results.append({
                'scenario_id': scenario_info['id'],
                'description': scenario_info['description'],
                'confidence': confidence
            })
        
        return results
    
    def get_scenario_info(self, scenario_id: str) -> Optional[Dict]:
        """Get scenario information by ID."""
        for scenario_info in self.scenario_mapping.values():
            if scenario_info['id'] == scenario_id:
                return scenario_info
        return None


# Global router instance (lazy-loaded)
_router_instance = None


def get_router() -> LlamaRouter:
    """Get the global router instance (singleton pattern)."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LlamaRouter()
    return _router_instance


def route(text: str, threshold: float = None) -> Tuple[Optional[str], float]:
    """
    Convenience function for routing text to scenarios.
    
    Args:
        text: Input situation text
        threshold: Similarity threshold
    
    Returns:
        Tuple of (scenario_id, confidence_score)
    """
    router = get_router()
    return router.route(text, threshold)


def route_top_k(text: str, k: int = None) -> List[Dict]:
    """
    Convenience function for getting top-k scenario matches.
    
    Args:
        text: Input situation text
        k: Number of top scenarios to return
    
    Returns:
        List of scenario matches with confidence scores
    """
    router = get_router()
    return router.route_top_k(text, k)
