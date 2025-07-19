"""
Joint embedding router that combines scenario descriptions and tags into unified embeddings.
This approach embeds the concatenated text+tags together rather than separate fusion.
"""

import json
import numpy as np
import hnswlib
from typing import List, Dict, Any
import os

import config
from tag_generator import TagGenerator
from improved_embeddings import ImprovedEmbeddingGenerator


class JointEmbeddingRouter:
    """Router that creates joint embeddings of scenario descriptions + tags."""
    
    def __init__(self, scenarios_file: str = "atomic-scenarios.json"):
        self.scenarios_file = scenarios_file
        self.tag_generator = TagGenerator()
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
        
        # Try to load joint index, build if not exists
        self.index_file = "joint_embedding.index"
        self.embeddings_file = "joint_embeddings.npy"
        self.enhanced_scenarios_file = "joint_enhanced_scenarios.json"
        
        if self._index_exists():
            self._load_index()
        else:
            self._build_index()
    
    def _index_exists(self) -> bool:
        """Check if joint embedding index files exist."""
        return (os.path.exists(self.index_file) and 
                os.path.exists(self.embeddings_file) and
                os.path.exists(self.enhanced_scenarios_file))
    
    def _create_joint_text(self, scenario: Dict, tags: List[str] = None) -> str:
        """Create joint text representation combining description and tags."""
        description = scenario.get('description', '')
        
        # Use provided tags or generate them
        if tags is None:
            tags = self.tag_generator.generate_tags(description)
        
        # Create structured joint representation
        joint_parts = [description]
        
        if tags:
            # Convert tags to natural language
            tag_text = self._tags_to_natural_language(tags)
            joint_parts.append(f"Context: {tag_text}")
            
            # Also add raw tags for exact matching
            joint_parts.append(f"Tags: {', '.join(tags)}")
        
        return ". ".join(joint_parts)
    
    def _tags_to_natural_language(self, tags: List[str]) -> str:
        """Convert tags to natural language description."""
        if not tags:
            return ""
        
        # Group tags by type (if they follow a pattern)
        tag_groups = {}
        for tag in tags:
            if '_' in tag:
                prefix = tag.split('_')[0]
                if prefix not in tag_groups:
                    tag_groups[prefix] = []
                tag_groups[prefix].append(tag.replace('_', ' '))
            else:
                if 'general' not in tag_groups:
                    tag_groups['general'] = []
                tag_groups['general'].append(tag)
        
        # Convert to natural language
        descriptions = []
        for group, group_tags in tag_groups.items():
            if group == 'general':
                descriptions.append(f"involving {', '.join(group_tags)}")
            else:
                descriptions.append(f"{group}: {', '.join(group_tags)}")
        
        return "; ".join(descriptions)
    
    def _build_index(self):
        """Build joint embedding HNSW index."""
        print("🔧 Building joint embedding index...")
        
        # Generate tags for all scenarios first
        print("🏷️  Generating tags for all scenarios...")
        scenario_tags = {}
        for i, scenario in enumerate(self.scenarios):
            print(f"  Generating tags for scenario {i+1}/{len(self.scenarios)}: {scenario['id']}")
            tags = self.tag_generator.generate_tags(scenario['description'])
            scenario_tags[scenario['id']] = tags
        
        # Create joint text representations
        print("📝 Creating joint text representations...")
        joint_texts = []
        enhanced_scenarios = []
        
        for scenario in self.scenarios:
            tags = scenario_tags[scenario['id']]
            joint_text = self._create_joint_text(scenario, tags)
            joint_texts.append(joint_text)
            
            enhanced_scenarios.append({
                'scenario_id': scenario['id'],
                'original_description': scenario['description'],
                'generated_tags': tags,
                'joint_text': joint_text,
                'embedding_index': len(enhanced_scenarios)
            })
        
        print("🚀 Generating joint embeddings...")
        
        # Generate embeddings for joint texts
        embeddings = self.embedder.generate_ensemble_embeddings(joint_texts)
        
        print(f"✅ Generated joint embeddings shape: {embeddings.shape}")
        
        # Build HNSW index
        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        self.index.add_items(embeddings, list(range(len(embeddings))))
        self.index.set_ef(50)
        
        # Save index and embeddings
        self.index.save_index(self.index_file)
        np.save(self.embeddings_file, embeddings)
        
        # Save enhanced scenarios for reference
        with open(self.enhanced_scenarios_file, 'w') as f:
            json.dump(enhanced_scenarios, f, indent=2)
        
        print("💾 Saved joint embedding index and enhanced scenarios")
    
    def _load_index(self):
        """Load existing joint embedding index."""
        print("📂 Loading joint embedding index...")
        
        # Load embeddings to get dimensions
        embeddings = np.load(self.embeddings_file)
        
        # Load index
        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.load_index(self.index_file, max_elements=len(embeddings))
        self.index.set_ef(50)
        
        print(f"✅ Loaded joint embedding index with {len(embeddings)} scenarios")
    
    def route_top_k(self, text: str, k: int = None) -> List[Dict]:
        """Find top-k matching scenarios using joint embeddings."""
        if k is None:
            k = config.TOP_K_SCENARIOS
        
        # Generate tags for input text
        input_tags = self.tag_generator.generate_tags(text)
        
        # Store last generated tags for benchmark access
        self._last_generated_tags = input_tags
        
        # Create joint representation of input
        input_scenario = {'description': text}
        joint_input_text = self._create_joint_text(input_scenario, input_tags)
        
        print(f"Joint input representation: {joint_input_text[:100]}...")
        
        # Generate embedding for joint input
        input_embedding = self.embedder.generate_ensemble_embeddings([joint_input_text])
        
        # Search in joint index
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
                'input_tags': input_tags,
                'joint_representation': joint_input_text,
                'scenario': scenario
            }
            results.append(result)
        
        return results


# Global joint embedding router instance
_joint_router_instance = None


def get_joint_embedding_router() -> JointEmbeddingRouter:
    """Get the global joint embedding router instance."""
    global _joint_router_instance
    if _joint_router_instance is None:
        _joint_router_instance = JointEmbeddingRouter()
    return _joint_router_instance


def joint_embedding_route_top_k(text: str, k: int = None) -> List[Dict]:
    """Convenience function for joint embedding routing."""
    router = get_joint_embedding_router()
    return router.route_top_k(text, k)
