#!/usr/bin/env python3
"""
Build HNSW index for scenario tag embeddings using Llama-3.1-8B.
This replaces the original build_index.py to work with tags instead of raw descriptions.
Usage: python build_tag_index.py [--force]
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
from tag_generator import get_tag_generator

# New paths for tag-based index
TAG_INDEX_PATH = HNSW_INDEX_PATH.replace('.idx', '_tags.idx')
TAG_MAPPING_PATH = HNSW_INDEX_PATH.replace('.idx', '_tags_mapping.json')
SCENARIO_TAGS_PATH = SCENARIOS_FILE.replace('.json', '_with_tags.json')


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


def generate_scenario_tags(scenarios: List[dict], force: bool = False) -> List[dict]:
    """
    Generate tags for all scenarios and save to file.
    
    Args:
        scenarios: List of scenario dictionaries
        force: Whether to regenerate tags even if file exists
        
    Returns:
        List of scenarios with tags added
    """
    # Check if we already have scenarios with tags
    if os.path.exists(SCENARIO_TAGS_PATH) and not force:
        print(f"Loading existing scenario tags from {SCENARIO_TAGS_PATH}")
        with open(SCENARIO_TAGS_PATH, 'r') as f:
            data = json.load(f)
        return data.get("scenarios", [])
    
    print("Generating tags for all scenarios...")
    tag_generator = get_tag_generator()
    
    scenarios_with_tags = []
    for scenario in tqdm(scenarios, desc="Generating tags"):
        # Generate tags for the scenario description
        tags = tag_generator.generate_tags(scenario['description'])
        
        # Create new scenario dict with tags
        scenario_with_tags = scenario.copy()
        scenario_with_tags['tags'] = tags
        scenarios_with_tags.append(scenario_with_tags)
        
        print(f"Scenario: {scenario['description'][:50]}...")
        print(f"Tags: {tags}")
        print()
    
    # Save scenarios with tags
    scenarios_data = {"scenarios": scenarios_with_tags}
    with open(SCENARIO_TAGS_PATH, 'w') as f:
        json.dump(scenarios_data, f, indent=2)
    
    print(f"Saved scenarios with tags to {SCENARIO_TAGS_PATH}")
    return scenarios_with_tags


def tags_to_text(tags: List[str]) -> str:
    """Convert a list of tags to a text string for embedding."""
    # Join tags with spaces to create a text representation
    return " ".join(tags)


def generate_tag_embeddings(model, scenarios_with_tags: List[dict], batch_size: int = 16) -> np.ndarray:
    """Generate embeddings for scenario tags using vLLM."""
    # Convert tags to text for embedding
    tag_texts = [tags_to_text(scenario['tags']) for scenario in scenarios_with_tags]
    
    print(f"Generating embeddings for {len(tag_texts)} tag sets with batch size {batch_size}...")
    
    # vLLM's embed method handles batching internally
    outputs = model.embed(tag_texts)
    
    # Extract embeddings from the output
    all_embeddings = [o.outputs.embedding for o in outputs]
    
    return np.array(all_embeddings, dtype=np.float32)


def build_hnsw_index(embeddings: np.ndarray, force: bool = False) -> hnswlib.Index:
    """Build and save HNSW index for tag embeddings."""
    if os.path.exists(TAG_INDEX_PATH) and not force:
        print(f"Tag index already exists at {TAG_INDEX_PATH}. Use --force to rebuild.")
        print(f"Loading existing index from {TAG_INDEX_PATH}...")
        index = hnswlib.Index(space='cosine', dim=EMBED_DIM)
        index.load_index(TAG_INDEX_PATH)
        return index

    print(f"Building HNSW index with {len(embeddings)} tag embeddings...")
    # Create HNSW index
    index = hnswlib.Index(space='cosine', dim=EMBED_DIM)
    index.init_index(max_elements=len(embeddings), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)

    # Add embeddings to the index
    index.add_items(embeddings, np.arange(len(embeddings)))

    # Save the index
    print(f"Saving tag index to {TAG_INDEX_PATH}...")
    index.save_index(TAG_INDEX_PATH)
    print("Tag index saved successfully!")

    return index


def save_tag_mapping(scenarios_with_tags: List[dict]):
    """Save scenario ID to index mapping for tag-based system."""
    mapping = {
        i: {
            'id': scenario['id'],
            'description': scenario['description'],
            'tags': scenario['tags']
        }
        for i, scenario in enumerate(scenarios_with_tags)
    }
    
    with open(TAG_MAPPING_PATH, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Tag scenario mapping saved to {TAG_MAPPING_PATH}")


def main():
    """Main function to build HNSW index from scenario tags."""
    parser = argparse.ArgumentParser(description="Build HNSW index for tag-based scenario matching")
    parser.add_argument('--force', action='store_true', 
                       help='Force rebuild even if index exists')
    parser.add_argument('--force-tags', action='store_true',
                       help='Force regeneration of tags even if they exist')
    args = parser.parse_args()
    
    # Check if index already exists
    if os.path.exists(TAG_INDEX_PATH) and not args.force:
        print(f"Tag index already exists at {TAG_INDEX_PATH}")
        print("Use --force to rebuild")
        return
    
    # Load scenarios
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")
    
    # Generate tags for scenarios
    scenarios_with_tags = generate_scenario_tags(scenarios, force=args.force_tags)
    
    # Load embedding model
    model = load_model()

    # Generate embeddings for tags
    embeddings = generate_tag_embeddings(model, scenarios_with_tags, batch_size=EMBED_BATCH_SIZE)
    print(f"Generated {len(embeddings)} tag embeddings with dimension {embeddings.shape[1]}")
    
    # Verify embedding dimension
    if embeddings.shape[1] != EMBED_DIM:
        print(f"Warning: Expected embedding dimension {EMBED_DIM}, got {embeddings.shape[1]}")
    
    # Build HNSW index
    index = build_hnsw_index(embeddings, force=args.force)
    
    # Save tag mapping
    save_tag_mapping(scenarios_with_tags)
    
    print("\nTag-based index building completed successfully!")
    print(f"Tag index: {TAG_INDEX_PATH}")
    print(f"Tag mapping: {TAG_MAPPING_PATH}")
    print(f"Scenarios with tags: {SCENARIO_TAGS_PATH}")


if __name__ == "__main__":
    main()
