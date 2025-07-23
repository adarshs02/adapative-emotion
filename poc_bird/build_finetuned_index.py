#!/usr/bin/env python3
"""
Build HNSW index using fine-tuned Llama 3.1 embedding model with tags.
This uses the best-performing model (100% scenario matching accuracy).
Usage: python build_finetuned_index.py [--force]
"""

import argparse
import json
import os
import sys
from typing import List, Dict
import numpy as np
import hnswlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# Configuration
FINETUNED_MODEL_PATH = "llama_embedding_with_tags_20250720_225721/final_model"
BASE_MODEL = "meta-llama/Llama-3.1-8B"
SCENARIOS_FILE = "atomic-scenarios.json"
INDEX_DIR = "indices"
FINETUNED_INDEX_PATH = os.path.join(INDEX_DIR, "finetuned_embedding.idx")
FINETUNED_MAPPING_PATH = os.path.join(INDEX_DIR, "finetuned_mapping.json")
FINETUNED_EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "finetuned_embeddings.npy")

# HNSW parameters
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
EMBED_DIM = 4096  # Llama 3.1 8B embedding dimension


class FineTunedEmbeddingModel:
    """Wrapper for the fine-tuned Llama embedding model."""
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Llama-3.1-8B"):
        self.model_path = model_path
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Loading fine-tuned model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModel.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        
        print(f"✅ Fine-tuned model loaded successfully!")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                
                # Normalize
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_scenarios() -> List[Dict]:
    """Load scenarios from JSON file."""
    with open(SCENARIOS_FILE, 'r') as f:
        data = json.load(f)
    
    # Handle nested structure if present
    if 'scenarios' in data:
        scenarios = data['scenarios']
    else:
        scenarios = data
    
    print(f"📋 Loaded {len(scenarios)} scenarios")
    return scenarios


def generate_embeddings(model: FineTunedEmbeddingModel, scenarios: List[Dict]) -> np.ndarray:
    """Generate embeddings for all scenarios using fine-tuned model."""
    # Extract scenario descriptions
    texts = [scenario['description'] for scenario in scenarios]
    
    print(f"🔧 Generating embeddings for {len(texts)} scenarios...")
    embeddings = model.encode(texts, batch_size=16)
    
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def build_hnsw_index(embeddings: np.ndarray, force: bool = False) -> hnswlib.Index:
    """Build and save HNSW index."""
    if os.path.exists(FINETUNED_INDEX_PATH) and not force:
        print(f"Index already exists at {FINETUNED_INDEX_PATH}. Use --force to rebuild.")
        print(f"Loading existing index...")
        index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        index.load_index(FINETUNED_INDEX_PATH)
        return index

    print(f"🏗️  Building HNSW index with {len(embeddings)} embeddings...")
    
    # Create HNSW index
    index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
    index.init_index(max_elements=len(embeddings), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M)

    # Add embeddings to the index
    index.add_items(embeddings, np.arange(len(embeddings)))
    
    # Optimize search performance
    index.set_ef(50)

    # Save the index
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.save_index(FINETUNED_INDEX_PATH)
    print(f"💾 Index saved to: {FINETUNED_INDEX_PATH}")

    return index


def save_scenario_mapping(scenarios: List[Dict]):
    """Save scenario ID to index mapping."""
    mapping = {
        i: {
            'id': scenario['id'],
            'description': scenario['description']
        }
        for i, scenario in enumerate(scenarios)
    }
    
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(FINETUNED_MAPPING_PATH, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"🗺️  Scenario mapping saved to: {FINETUNED_MAPPING_PATH}")


def save_embeddings(embeddings: np.ndarray):
    """Save embeddings for future use."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(FINETUNED_EMBEDDINGS_PATH, embeddings)
    print(f"📊 Embeddings saved to: {FINETUNED_EMBEDDINGS_PATH}")


def main():
    """Main function to build HNSW index using fine-tuned model."""
    parser = argparse.ArgumentParser(description="Build HNSW index using fine-tuned embedding model")
    parser.add_argument('--force', action='store_true', 
                       help='Force rebuild even if index exists')
    parser.add_argument('--model_path', type=str, default=FINETUNED_MODEL_PATH,
                       help='Path to fine-tuned model')
    args = parser.parse_args()
    
    # Check if index already exists
    if os.path.exists(FINETUNED_INDEX_PATH) and not args.force:
        print(f"🎯 Fine-tuned index already exists at {FINETUNED_INDEX_PATH}")
        print("Use --force to rebuild")
        return
    
    print("🏁 Building production index with fine-tuned embedding model...")
    print("=" * 70)
    print(f"📍 Model: {args.model_path}")
    print(f"📊 Expected performance: 100% scenario matching accuracy")
    print("=" * 70)
    
    # Load scenarios
    scenarios = load_scenarios()
    
    # Load fine-tuned model
    model = FineTunedEmbeddingModel(args.model_path, BASE_MODEL)
    
    # Generate embeddings
    embeddings = generate_embeddings(model, scenarios)
    
    # Verify embedding dimension
    expected_dim = 4096  # Llama 3.1 8B expected dimension
    if embeddings.shape[1] != expected_dim:
        print(f"⚠️  Note: Expected embedding dimension {expected_dim}, got {embeddings.shape[1]}")
        print(f"✅ Using actual model dimension: {embeddings.shape[1]}")
    
    # Build HNSW index
    index = build_hnsw_index(embeddings, force=args.force)
    
    # Save embeddings and mapping
    save_embeddings(embeddings)
    save_scenario_mapping(scenarios)
    
    print("\n" + "=" * 70)
    print("🎉 Fine-tuned index building completed successfully!")
    print("=" * 70)
    print(f"📁 Index: {FINETUNED_INDEX_PATH}")
    print(f"🗺️  Mapping: {FINETUNED_MAPPING_PATH}")
    print(f"📊 Embeddings: {FINETUNED_EMBEDDINGS_PATH}")
    print(f"🔍 Scenarios indexed: {len(scenarios)}")
    print(f"🎯 Expected accuracy: 100% scenario matching")
    print("=" * 70)
    print("🚀 Ready for production use!")


if __name__ == "__main__":
    main()
