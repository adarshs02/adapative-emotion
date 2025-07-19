"""
Improved text-based router using enhanced embedding strategies.
"""

import json
import numpy as np
import hnswlib
from typing import List, Dict, Any
import os

import config
from improved_embeddings import ImprovedEmbeddingGenerator


class ImprovedTextRouter:
    """Text-based router with improved embedding strategies."""
    
    def __init__(self, scenarios_file: str = "atomic-scenarios.json"):
        self.scenarios_file = scenarios_file
        self.embedder = ImprovedEmbeddingGenerator()
        
        # Load scenarios
        with open(scenarios_file, 'r') as f:
            data = json.load(f)
            # Handle both direct array and wrapped structure
            if isinstance(data, dict) and 'scenarios' in data:
                self.scenarios = data['scenarios']
            else:
                self.scenarios = data
        
        self.scenario_lookup = {scenario['id']: scenario for scenario in self.scenarios}
        
        # Try to load improved index, build if not exists
        self.index_file = "improved_atomic_text.index"
        self.embeddings_file = "improved_atomic_embeddings.npy"
        self.enhanced_scenarios_file = "enhanced_atomic_scenarios.json"
        
        if self._index_exists():
            self._load_index()
        else:
            self._build_index()
    
    def _index_exists(self) -> bool:
        """Check if improved index files exist."""
        return (os.path.exists(self.index_file) and 
                os.path.exists(self.embeddings_file) and
                os.path.exists(self.enhanced_scenarios_file))
    
    def _build_index(self):
        """Build improved HNSW index with enhanced embeddings."""
        print("🔧 Building improved text index...")
        
        # Generate enhanced text representations
        enhanced_texts = []
        for scenario in self.scenarios:
            enhanced_text = self.embedder.create_enhanced_text(scenario)
            enhanced_texts.append(enhanced_text)
        
        print("🚀 Generating improved embeddings...")
        
        # Generate ensemble embeddings
        embeddings = self.embedder.generate_ensemble_embeddings(enhanced_texts)
        
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        
        # Build HNSW index
        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        self.index.add_items(embeddings, list(range(len(embeddings))))
        self.index.set_ef(50)
        
        # Save index and embeddings
        self.index.save_index(self.index_file)
        np.save(self.embeddings_file, embeddings)
        
        # Save enhanced scenarios for reference
        enhanced_data = []
        for i, scenario in enumerate(self.scenarios):
            enhanced_data.append({
                'scenario_id': scenario['id'],
                'original_description': scenario['description'],
                'enhanced_text': enhanced_texts[i],
                'embedding_index': i
            })
        
        with open(self.enhanced_scenarios_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        print("💾 Saved improved index and enhanced scenarios")
    
    def _load_index(self):
        """Load existing improved index."""
        print("📂 Loading improved text index...")
        
        # Load embeddings to get dimensions
        embeddings = np.load(self.embeddings_file)
        
        # Load index
        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.load_index(self.index_file, max_elements=len(embeddings))
        self.index.set_ef(50)
        
        print(f"✅ Loaded improved index with {len(embeddings)} scenarios")
    
    def route_top_k(self, text: str, k: int = None) -> List[Dict]:
        """Find top-k matching scenarios using improved embeddings."""
        if k is None:
            k = config.TOP_K_SCENARIOS
        
        # Generate enhanced representation of input text
        enhanced_text = self.embedder.create_enhanced_text({'description': text})
        
        # Generate ensemble embedding for input
        input_embedding = self.embedder.generate_ensemble_embeddings([enhanced_text])
        
        # Search in index
        labels, distances = self.index.knn_query(input_embedding, k=k)
        
        # Convert to results
        results = []
        for label, distance in zip(labels[0], distances[0]):
            scenario = self.scenarios[label]
            confidence = 1.0 - distance  # Convert distance to confidence
            
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


# Global improved router instance
_improved_router_instance = None


def get_improved_text_router() -> ImprovedTextRouter:
    """Get the global improved text router instance."""
    global _improved_router_instance
    if _improved_router_instance is None:
        _improved_router_instance = ImprovedTextRouter()
    return _improved_router_instance


def improved_text_route_top_k(text: str, k: int = None) -> List[Dict]:
    """Convenience function for improved text routing."""
    router = get_improved_text_router()
    return router.route_top_k(text, k)
