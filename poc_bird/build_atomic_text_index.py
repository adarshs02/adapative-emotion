#!/usr/bin/env python3
"""
Build HNSW index for atomic scenario text embeddings using Llama-3.1-8B.
This creates a text-based index using atomic-scenarios.json for fair comparison with tag-based system.
Usage: python build_atomic_text_index.py [--force]
"""

import argparse
import json
import os
from typing import List, Dict

import numpy as np
import hnswlib
from vllm import LLM
from tqdm import tqdm

from config import (
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE, EMBED_DIM,
    HNSW_INDEX_PATH, HNSW_M, HNSW_EF_CONSTRUCTION,
    SCENARIOS_FILE
)

# Paths for atomic text-based index
ATOMIC_SCENARIOS_FILE = "atomic-scenarios.json"
ATOMIC_TEXT_INDEX_PATH = HNSW_INDEX_PATH.replace('.idx', '_atomic_text.idx')
ATOMIC_TEXT_MAPPING_PATH = HNSW_INDEX_PATH.replace('.idx', '_atomic_text_mapping.json')


def load_model():
    """Load the embedding model using vLLM."""
    print(f"Loading model {EMBED_MODEL_NAME} for embedding...")
    model = LLM(
        model=EMBED_MODEL_NAME,
        task="embed",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
    )
    print("Model loaded successfully!")
    return model


def load_atomic_scenarios() -> List[dict]:
    """Load atomic scenarios from JSON file."""
    if not os.path.exists(ATOMIC_SCENARIOS_FILE):
        raise FileNotFoundError(f"Atomic scenarios file not found: {ATOMIC_SCENARIOS_FILE}")
    
    with open(ATOMIC_SCENARIOS_FILE, 'r') as f:
        data = json.load(f)
    
    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError(f"No scenarios found in {ATOMIC_SCENARIOS_FILE}")
    
    print(f"Loaded {len(scenarios)} atomic scenarios")
    return scenarios


def generate_text_embeddings(model, scenarios: List[dict], batch_size: int = 16) -> np.ndarray:
    """Generate embeddings for scenario descriptions using vLLM."""
    # Extract descriptions for embedding
    descriptions = [scenario['description'] for scenario in scenarios]
    
    print(f"Generating text embeddings for {len(descriptions)} atomic scenarios with batch size {batch_size}...")
    
    # vLLM's embed method handles batching internally
    outputs = model.embed(descriptions)
    
    # Extract embeddings from the output
    all_embeddings = [o.outputs.embedding for o in outputs]
    
    return np.array(all_embeddings, dtype=np.float32)


def build_hnsw_index(embeddings: np.ndarray, force: bool = False) -> hnswlib.Index:
    """Build and save HNSW index for atomic text embeddings."""
    if os.path.exists(ATOMIC_TEXT_INDEX_PATH) and not force:
        print(f"Atomic text index already exists at {ATOMIC_TEXT_INDEX_PATH}. Use --force to rebuild.")
        print(f"Loading existing index from {ATOMIC_TEXT_INDEX_PATH}...")
        index = hnswlib.Index(space='cosine', dim=EMBED_DIM)
        index.load_index(ATOMIC_TEXT_INDEX_PATH)
        return index

    print(f"Building HNSW index with {len(embeddings)} atomic text embeddings...")
    # Create HNSW index
    index = hnswlib.Index(space='cosine', dim=EMBED_DIM)
    index.init_index(max_elements=len(embeddings), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)

    # Add embeddings to the index
    index.add_items(embeddings, np.arange(len(embeddings)))

    # Save the index
    print(f"Saving atomic text index to {ATOMIC_TEXT_INDEX_PATH}...")
    index.save_index(ATOMIC_TEXT_INDEX_PATH)
    print("Atomic text index saved successfully!")

    return index


def save_atomic_text_mapping(scenarios: List[dict]):
    """Save scenario ID to index mapping for atomic text-based system."""
    mapping = {
        i: {
            'id': scenario['id'],
            'description': scenario['description']
        }
        for i, scenario in enumerate(scenarios)
    }
    
    with open(ATOMIC_TEXT_MAPPING_PATH, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Atomic text scenario mapping saved to {ATOMIC_TEXT_MAPPING_PATH}")


def main():
    """Main function to build HNSW index from atomic scenario descriptions."""
    parser = argparse.ArgumentParser(description="Build HNSW index for atomic text-based scenario matching")
    parser.add_argument('--force', action='store_true', 
                       help='Force rebuild even if index exists')
    args = parser.parse_args()
    
    # Check if index already exists
    if os.path.exists(ATOMIC_TEXT_INDEX_PATH) and not args.force:
        print(f"Atomic text index already exists at {ATOMIC_TEXT_INDEX_PATH}")
        print("Use --force to rebuild")
        return
    
    # Load atomic scenarios
    scenarios = load_atomic_scenarios()
    print(f"Loaded {len(scenarios)} atomic scenarios")
    
    # Load embedding model
    model = load_model()

    # Generate embeddings for descriptions
    embeddings = generate_text_embeddings(model, scenarios, batch_size=EMBED_BATCH_SIZE)
    print(f"Generated {len(embeddings)} text embeddings with dimension {embeddings.shape[1]}")
    
    # Verify embedding dimension
    if embeddings.shape[1] != EMBED_DIM:
        print(f"Warning: Expected embedding dimension {EMBED_DIM}, got {embeddings.shape[1]}")
    
    # Build HNSW index
    index = build_hnsw_index(embeddings, force=args.force)
    
    # Save atomic text mapping
    save_atomic_text_mapping(scenarios)
    
    print("\nAtomic text-based index building completed successfully!")
    print(f"Atomic text index: {ATOMIC_TEXT_INDEX_PATH}")
    print(f"Atomic text mapping: {ATOMIC_TEXT_MAPPING_PATH}")


if __name__ == "__main__":
    main()
