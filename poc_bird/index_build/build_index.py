#!/usr/bin/env python3
"""
Build HNSW index for scenario embeddings using Llama-3.1-8B.
Usage: python build_index.py [--force]
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


def load_model():
    """Load the embedding model using vLLM."""
    print(f"Loading model {EMBED_MODEL_NAME} for embedding...")
    # Use vLLM for embedding tasks
    model = LLM(
        model=EMBED_MODEL_NAME,
        task="embed",
        enforce_eager=True,  # Recommended for embedding models
    )
    print("Model loaded successfully!")
    return model


def embed_text(model, tokenizer, text: str) -> np.ndarray:
    """Generate embedding for a single text using transformers."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling over the sequence dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy().astype(np.float32)


def load_scenarios() -> List[dict]:
    """Load scenarios from JSON file."""
    if not os.path.exists(SCENARIOS_FILE):
        raise FileNotFoundError(f"Scenarios file not found: {SCENARIOS_FILE}")
    
    with open(SCENARIOS_FILE, 'r') as f:
        data = json.load(f)
    
    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError(f"No scenarios found in {SCENARIOS_FILE}")
    
    print(f"Loaded {len(scenarios)} scenarios")
    return scenarios


def build_hnsw_index(embeddings: np.ndarray, force: bool = False) -> hnswlib.Index:
    """Build and save HNSW index."""
    if os.path.exists(HNSW_INDEX_PATH) and not force:
        print(f"Index already exists at {HNSW_INDEX_PATH}. Use --force to rebuild.")
        # Instead of exiting, let's load the existing index
        print(f"Loading existing index from {HNSW_INDEX_PATH}...")
        index = hnswlib.Index(space='cosine', dim=EMBED_DIM)
        index.load_index(HNSW_INDEX_PATH)
        return index

    print(f"Building HNSW index with {len(embeddings)} embeddings...")
    # Create HNSW index
    index = hnswlib.Index(space='cosine', dim=EMBED_DIM)
    index.init_index(max_elements=len(embeddings), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)

    # Add embeddings to the index
    index.add_items(embeddings, np.arange(len(embeddings)))

    # Save the index
    print(f"Saving index to {HNSW_INDEX_PATH}...")
    index.save_index(HNSW_INDEX_PATH)
    print("Index saved successfully!")

    return index


def generate_embeddings(model, texts: list, batch_size: int = 16) -> np.ndarray:
    """Generate embeddings for a list of texts using vLLM."""
    print(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}...")
    
    # vLLM's embed method handles batching internally
    outputs = model.embed(texts)
    
    # Extract embeddings from the output
    all_embeddings = [o.outputs.embedding for o in outputs]
    
    return np.array(all_embeddings, dtype=np.float32)


def save_scenario_mapping(scenarios: List[dict]):
    """Save scenario ID to index mapping."""
    mapping = {
        i: {
            'id': scenario['id'],
            'description': scenario['description']
        }
        for i, scenario in enumerate(scenarios)
    }
    
    mapping_file = config.HNSW_INDEX_PATH.replace('.idx', '_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Scenario mapping saved to {mapping_file}")


def main():
    """Main function to build HNSW index from scenarios."""
    parser = argparse.ArgumentParser(description="Build HNSW index for scenario matching")
    parser.add_argument('--force', action='store_true', 
                       help='Force rebuild even if index exists')
    args = parser.parse_args()
    
    # Check if index already exists
    if os.path.exists(HNSW_INDEX_PATH) and not args.force:
        print(f"Index already exists at {HNSW_INDEX_PATH}")
        print("Use --force to rebuild")
        return
    
    # Load scenarios
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")
    
    # Load embedding model
    model = load_model()

    # Generate embeddings
    descriptions = [scenario['description'] for scenario in scenarios]
    embeddings = generate_embeddings(model, descriptions, batch_size=EMBED_BATCH_SIZE)
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Verify embedding dimension
    if embeddings.shape[1] != EMBED_DIM:
        print(f"Warning: Expected embedding dimension {EMBED_DIM}, got {embeddings.shape[1]}")
    
    # Build HNSW index
    index = build_hnsw_index(embeddings)
    
    # Save index
    index.save_index(HNSW_INDEX_PATH)
    print(f"Saved HNSW index to {HNSW_INDEX_PATH}")
    
    # Save scenario mapping
    mapping_file = HNSW_INDEX_PATH.replace('.idx', '_mapping.json')
    scenario_mapping = {i: scenario for i, scenario in enumerate(scenarios)}
    
    with open(mapping_file, 'w') as f:
        json.dump(scenario_mapping, f, indent=2)
    print(f"Saved scenario mapping to {mapping_file}")
    
    print("\nIndex building completed successfully!")


if __name__ == "__main__":
    main()
