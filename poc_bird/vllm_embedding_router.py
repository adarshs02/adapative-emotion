"""
Scenario router using vLLM in embedding mode for efficient inference.
"""

import json
import numpy as np
from typing import List, Dict, Any
import os
import hnswlib
from vllm import LLM
import torch


class VLLMEmbeddingRouter:
    """Scenario router using vLLM for embedding generation."""
    
    def __init__(self, model_path: str, scenarios_file: str = "atomic-scenarios.json"):
        self.model_path = model_path
        self.scenarios_file = scenarios_file
        
        # Initialize vLLM for embeddings
        print(f"🚀 Initializing vLLM embedding model: {model_path}")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            max_model_len=512,
            enforce_eager=True
        )
        
        # Load scenarios
        with open(scenarios_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'scenarios' in data:
                self.scenarios = data['scenarios']
            else:
                self.scenarios = data
        
        # Build or load index
        self.index_file = f"vllm_{os.path.basename(model_path).replace('/', '_')}.index"
        self.embeddings_file = f"vllm_{os.path.basename(model_path).replace('/', '_')}_embeddings.npy"
        
        if self._index_exists():
            self._load_index()
        else:
            self._build_index()
    
    def _index_exists(self) -> bool:
        """Check if index files exist."""
        return os.path.exists(self.index_file) and os.path.exists(self.embeddings_file)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using vLLM."""
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        # Use vLLM to encode texts
        # Note: This uses the model's hidden states as embeddings
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Generate embeddings using vLLM
            # We'll use the model's last hidden state as embeddings
            outputs = self.llm.encode(batch_texts)
            
            # Extract embeddings (this may need adjustment based on vLLM API)
            batch_embeddings = []
            for output in outputs:
                # Get the mean of the last hidden states
                embedding = np.mean(output.outputs[0].hidden_states[-1], axis=0)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _build_index(self):
        """Build HNSW index using vLLM embeddings."""
        print(f"🔧 Building vLLM embedding index...")
        
        # Create scenario texts
        scenario_texts = [scenario['description'] for scenario in self.scenarios]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(scenario_texts)
        
        # Build HNSW index
        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        self.index.add_items(embeddings, list(range(len(embeddings))))
        self.index.set_ef(50)
        
        # Save index and embeddings
        self.index.save_index(self.index_file)
        np.save(self.embeddings_file, embeddings)
        
        print(f"💾 Saved vLLM embedding index")
    
    def _load_index(self):
        """Load existing index."""
        print(f"📂 Loading vLLM embedding index...")
        
        embeddings = np.load(self.embeddings_file)
        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.load_index(self.index_file, max_elements=len(embeddings))
        self.index.set_ef(50)
        
        print(f"✅ Loaded vLLM embedding index")
    
    def route_top_k(self, text: str, k: int = 5) -> List[Dict]:
        """Find top-k matching scenarios using vLLM embeddings."""
        # Generate embedding for input
        input_embedding = self.generate_embeddings([text])
        
        # Search in index
        labels, distances = self.index.knn_query(input_embedding, k=k)
        
        # Convert to results
        results = []
        for label, distance in zip(labels[0], distances[0]):
            scenario = self.scenarios[label]
            confidence = 1.0 - distance
            
            result = {
                'scenario_id': scenario['id'],
                'description': scenario['description'],
                'confidence': confidence,
                'score': confidence,
                'distance': distance,
                'scenario': scenario
            }
            results.append(result)
        
        return results


# Simplified version that doesn't rely on vLLM's encode method (which may not exist)
class SimpleVLLMEmbeddingRouter:
    """Simplified vLLM embedding router using model inference."""
    
    def __init__(self, model_path: str, scenarios_file: str = "atomic-scenarios.json"):
        self.model_path = model_path
        self.scenarios_file = scenarios_file
        
        # For now, we'll use the existing embedding generator from config
        # This is a placeholder until we have proper vLLM embedding integration
        from embedding_generator import EmbeddingGenerator
        self.embedder = EmbeddingGenerator()
        
        # Load scenarios
        with open(scenarios_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'scenarios' in data:
                self.scenarios = data['scenarios']
            else:
                self.scenarios = data
        
        # Build or load index
        self.index_file = f"simple_vllm_{os.path.basename(model_path).replace('/', '_')}.index"
        self.embeddings_file = f"simple_vllm_{os.path.basename(model_path).replace('/', '_')}_embeddings.npy"
        
        if self._index_exists():
            self._load_index()
        else:
            self._build_index()
    
    def _index_exists(self) -> bool:
        """Check if index files exist."""
        return os.path.exists(self.index_file) and os.path.exists(self.embeddings_file)
    
    def _build_index(self):
        """Build HNSW index using existing embedding generator."""
        print(f"🔧 Building simple vLLM-style embedding index...")
        
        # Create scenario texts
        scenario_texts = [scenario['description'] for scenario in self.scenarios]
        
        # Generate embeddings using existing embedder
        embeddings = self.embedder.generate_embeddings(scenario_texts)
        
        # Build HNSW index
        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        self.index.add_items(embeddings, list(range(len(embeddings))))
        self.index.set_ef(50)
        
        # Save index and embeddings
        self.index.save_index(self.index_file)
        np.save(self.embeddings_file, embeddings)
        
        print(f"💾 Saved simple vLLM embedding index")
    
    def _load_index(self):
        """Load existing index."""
        print(f"📂 Loading simple vLLM embedding index...")
        
        embeddings = np.load(self.embeddings_file)
        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.load_index(self.index_file, max_elements=len(embeddings))
        self.index.set_ef(50)
        
        print(f"✅ Loaded simple vLLM embedding index")
    
    def route_top_k(self, text: str, k: int = 5) -> List[Dict]:
        """Find top-k matching scenarios."""
        # Generate embedding for input
        input_embedding = self.embedder.generate_embeddings([text])
        
        # Search in index
        labels, distances = self.index.knn_query(input_embedding, k=k)
        
        # Convert to results
        results = []
        for label, distance in zip(labels[0], distances[0]):
            scenario = self.scenarios[label]
            confidence = 1.0 - distance
            
            result = {
                'scenario_id': scenario['id'],
                'description': scenario['description'],
                'confidence': confidence,
                'score': confidence,
                'distance': distance,
                'scenario': scenario
            }
            results.append(result)
        
        return results
