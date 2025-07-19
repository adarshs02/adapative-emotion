"""
Tag-based scenario router using HNSW indexing for efficient similarity search.
This replaces llama_router.py to work with tags instead of raw text.
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
from tag_generator import get_tag_generator

# Tag-based paths
TAG_INDEX_PATH = HNSW_INDEX_PATH.replace('.idx', '_tags.idx')
TAG_MAPPING_PATH = HNSW_INDEX_PATH.replace('.idx', '_tags_mapping.json')


class TagRouter:
    """Router that finds the best matching scenario using tag embeddings and HNSW indexing."""
    
    def __init__(self):
        """Initializes the router, loading the model and index."""
        self.index_path = TAG_INDEX_PATH
        self.mapping_path = TAG_MAPPING_PATH
        self.llm = None
        self.index = None
        self.scenario_mapping = None
        self.tag_generator = None

        self._load_model()
        self._load_index()
        self._load_tag_generator()

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
                f"Tag HNSW index not found at {self.index_path}. "
                f"Please run 'python build_tag_index.py' first."
            )
        
        print(f"Loading tag HNSW index from {self.index_path}...")
        self.index = hnswlib.Index(space='cosine', dim=EMBED_DIM)
        self.index.load_index(self.index_path)
        self.index.set_ef(HNSW_EF_SEARCH)
        print("Tag HNSW index loaded successfully!")
        
        # Load scenario mapping
        if not os.path.exists(self.mapping_path):
            raise FileNotFoundError(f"Tag scenario mapping not found at {self.mapping_path}")

        with open(self.mapping_path, 'r') as f:
            self.scenario_mapping = json.load(f)
        print(f"Loaded tag scenario mapping for {len(self.scenario_mapping)} scenarios")
    
    def _load_tag_generator(self):
        """Load the tag generator for processing user input."""
        self.tag_generator = get_tag_generator()
        print("Tag generator loaded successfully!")
    
    def _tags_to_text(self, tags: List[str]) -> str:
        """Convert a list of tags to a text string for embedding."""
        return " ".join(tags)
    
    def embed_tags(self, tags: List[str]) -> np.ndarray:
        """Generate embedding for tags using vLLM."""
        if self.llm is None:
            raise RuntimeError("vLLM model not loaded")

        # Convert tags to text
        tag_text = self._tags_to_text(tags)
        
        # Use vLLM's embed method
        outputs = self.llm.embed([tag_text])
        
        # Extract the embedding from the output
        embedding = outputs[0].outputs.embedding
        
        return np.array(embedding, dtype=np.float32).reshape(1, -1)
    
    def route(self, text: str, threshold: float = None) -> Tuple[Optional[str], float]:
        """
        Find the best matching scenario for the input text using tags.
        
        Args:
            text: Input situation text
            threshold: Similarity threshold (uses config default if None)
        
        Returns:
            Tuple of (scenario_id, confidence_score)
            Returns (None, score) if no scenario meets threshold
        """
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        
        # Generate tags for input text
        input_tags = self.tag_generator.generate_tags(text)
        print(f"Generated tags for input: {input_tags}")
        
        # Generate embedding for input tags
        query_embedding = self.embed_tags(input_tags)
        
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
        
        print(f"Matched scenario '{scenario_id}' with confidence {confidence:.4f}")
        print(f"Matched scenario tags: {scenario_info.get('tags', [])}")
        
        return scenario_id, confidence
    
    def route_top_k(self, text: str, k: int = None) -> List[Dict]:
        """
        Find top-k matching scenarios for the input text using tags.
        
        Args:
            text: Input situation text
            k: Number of top scenarios to return (uses config default if None)
        
        Returns:
            List of dictionaries with 'scenario_id', 'description', 'confidence', and 'tags'
        """
        if k is None:
            k = config.TOP_K_SCENARIOS
        
        # Generate tags for input text
        input_tags = self.tag_generator.generate_tags(text)
        print(f"Generated tags for input: {input_tags}")
        
        # Store last generated tags for benchmark access
        self._last_generated_tags = input_tags
        
        # Generate embedding for input tags
        query_embedding = self.embed_tags(input_tags)
        
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
                'tags': scenario_info.get('tags', []),
                'score': confidence  # For compatibility with existing code
            }
            
            # Add scenario object for compatibility
            result['scenario'] = {
                'id': scenario_info['id'],
                'description': scenario_info['description'],
                'tags': scenario_info.get('tags', [])
            }
            
            results.append(result)
        
        return results
    
    def get_scenario_info(self, scenario_id: str) -> Optional[Dict]:
        """Get scenario information by ID."""
        for scenario_info in self.scenario_mapping.values():
            if scenario_info['id'] == scenario_id:
                return scenario_info
        return None


# Global router instance (lazy-loaded)
_tag_router_instance = None


def get_tag_router() -> TagRouter:
    """Get the global tag router instance (singleton pattern)."""
    global _tag_router_instance
    if _tag_router_instance is None:
        _tag_router_instance = TagRouter()
    return _tag_router_instance


def route(text: str, threshold: float = None) -> Tuple[Optional[str], float]:
    """
    Convenience function for routing text to scenarios using tags.
    
    Args:
        text: Input situation text
        threshold: Similarity threshold
    
    Returns:
        Tuple of (scenario_id, confidence_score)
    """
    router = get_tag_router()
    return router.route(text, threshold)


def route_top_k(text: str, k: int = None) -> List[Dict]:
    """
    Convenience function for getting top-k scenario matches using tags.
    
    Args:
        text: Input situation text
        k: Number of top scenarios to return
    
    Returns:
        List of scenario matches with confidence scores
    """
    router = get_tag_router()
    return router.route_top_k(text, k)
