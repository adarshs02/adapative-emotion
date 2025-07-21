#!/usr/bin/env python3
"""
Build HNSW index for fine-tuned Llama 3.1 embedding model.
Creates a production-ready index for fast scenario retrieval.
"""

import json
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import hnswlib
from datetime import datetime
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
os.chdir(project_root)

from config import HNSW_M, HNSW_EF_CONSTRUCTION


class FineTunedEmbeddingModel:
    """Wrapper for the fine-tuned Llama embedding model."""
    
    def __init__(self, model_path: str, base_model: str = "meta-llama/Llama-3.1-8B"):
        self.model_path = model_path
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔥 Loading fine-tuned model from {model_path}...")
        
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
        
        print(f"  Encoding {len(texts)} texts in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"    Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
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


def load_scenarios(scenarios_file: str = "atomic-scenarios_with_tags.json") -> List[Dict]:
    """Load scenarios from JSON file."""
    print(f"📚 Loading scenarios from {scenarios_file}...")
    
    with open(scenarios_file, 'r') as f:
        data = json.load(f)
    
    scenarios = data['scenarios'] if 'scenarios' in data else data
    print(f"✅ Loaded {len(scenarios)} scenarios")
    
    return scenarios


def build_finetuned_index(model_path: str, scenarios_file: str = "atomic-scenarios_with_tags.json", 
                         output_prefix: str = "scenario_finetuned", batch_size: int = 16):
    """Build HNSW index using fine-tuned model embeddings."""
    
    print("🚀 Building Fine-tuned Model HNSW Index")
    print("=" * 60)
    
    # Load scenarios
    scenarios = load_scenarios(scenarios_file)
    
    # Load fine-tuned model
    finetuned_model = FineTunedEmbeddingModel(model_path)
    
    # Generate embeddings for scenarios + tags (matching training format)
    print(f"\n🧮 Generating embeddings for {len(scenarios)} scenarios with tags...")
    scenario_texts = []
    
    for scenario in scenarios:
        # Combine scenario description with tags (matching training data format)
        tags = scenario.get('tags', [])
        
        # Tags are already a list of strings
        if isinstance(tags, list):
            tag_text = ", ".join(tags) if tags else ""
        else:
            # Fallback for dictionary format (if any scenarios use that)
            tag_parts = []
            for category, values in tags.items():
                if isinstance(values, list):
                    tag_parts.extend(values)
                else:
                    tag_parts.append(values)
            tag_text = ", ".join(tag_parts) if tag_parts else ""
        
        # Combine description with tags
        combined_text = f"{scenario['description']}. Tags: {tag_text}" if tag_text else scenario['description']
        scenario_texts.append(combined_text)
    
    embeddings = finetuned_model.encode(scenario_texts, batch_size=batch_size)
    
    print(f"✅ Generated embeddings: shape {embeddings.shape}")
    
    # Get embedding dimension
    embed_dim = embeddings.shape[1]
    print(f"📏 Embedding dimension: {embed_dim}")
    
    # Build HNSW index
    print(f"\n🏗️  Building HNSW index...")
    print(f"   M = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION}")
    
    index = hnswlib.Index(space='cosine', dim=embed_dim)
    index.init_index(
        max_elements=len(scenarios), 
        ef_construction=HNSW_EF_CONSTRUCTION, 
        M=HNSW_M
    )
    
    # Add embeddings to index
    scenario_ids = list(range(len(scenarios)))
    index.add_items(embeddings, scenario_ids)
    
    # Save index
    index_path = f"{output_prefix}.idx"
    index.save_index(index_path)
    print(f"💾 Saved HNSW index to: {index_path}")
    
    # Create scenario mapping
    scenario_mapping = {}
    for i, scenario in enumerate(scenarios):
        scenario_mapping[str(i)] = {
            'id': scenario['id'],
            'description': scenario['description'],
            'tags': scenario.get('tags', {}),
            'index': i
        }
    
    # Save scenario mapping
    mapping_path = f"{output_prefix}_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(scenario_mapping, f, indent=2)
    print(f"💾 Saved scenario mapping to: {mapping_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'scenarios_file': scenarios_file,
        'total_scenarios': len(scenarios),
        'embedding_dimension': int(embed_dim),
        'index_path': index_path,
        'mapping_path': mapping_path,
        'hnsw_parameters': {
            'M': HNSW_M,
            'ef_construction': HNSW_EF_CONSTRUCTION
        },
        'batch_size': batch_size
    }
    
    metadata_path = f"{output_prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to: {metadata_path}")
    
    print(f"\n🎉 Fine-tuned index building completed!")
    print("=" * 60)
    print(f"📁 Generated files:")
    print(f"   • Index:    {index_path}")
    print(f"   • Mapping:  {mapping_path}")
    print(f"   • Metadata: {metadata_path}")
    
    # Test the index
    print(f"\n🧪 Testing index...")
    test_query = scenarios[0]['description']
    query_embedding = finetuned_model.encode([test_query])
    
    labels, distances = index.knn_query(query_embedding, k=5)
    
    print(f"Test query: '{test_query[:60]}...'")
    print(f"Top match: {scenarios[labels[0][0]]['id']} (distance: {distances[0][0]:.4f})")
    print(f"✅ Index is working correctly!")
    
    return index_path, mapping_path, metadata_path


def main():
    """Main index building function."""
    parser = argparse.ArgumentParser(description="Build HNSW index for fine-tuned embedding model")
    parser.add_argument("model_path", type=str, help="Path to fine-tuned model directory")
    parser.add_argument("--scenarios", type=str, default="atomic-scenarios_with_tags.json", 
                       help="Scenarios JSON file")
    parser.add_argument("--output-prefix", type=str, default="scenario_finetuned", 
                       help="Prefix for output files")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size for embedding generation")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path '{args.model_path}' does not exist")
        sys.exit(1)
    
    # Validate scenarios file
    if not os.path.exists(args.scenarios):
        print(f"❌ Error: Scenarios file '{args.scenarios}' does not exist")
        sys.exit(1)
    
    try:
        # Build index
        index_path, mapping_path, metadata_path = build_finetuned_index(
            model_path=args.model_path,
            scenarios_file=args.scenarios,
            output_prefix=args.output_prefix,
            batch_size=args.batch_size
        )
        
        print(f"\n🚀 Ready for production deployment!")
        print(f"Use these files in your production router:")
        print(f"   • Index: {index_path}")
        print(f"   • Mapping: {mapping_path}")
        
    except Exception as e:
        print(f"❌ Index building failed: {e}")
        raise


if __name__ == "__main__":
    main()
