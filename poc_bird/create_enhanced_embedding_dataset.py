"""
Enhanced Training Dataset Generator for Llama 3.1 Embedding Model
Optimized for scenario matching and tag-based retrieval with advanced data augmentation.
"""

import json
import random
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import itertools


class EnhancedEmbeddingDatasetGenerator:
    """Generate comprehensive training dataset optimized for Llama 3.1 embedding fine-tuning."""
    
    def __init__(self, scenarios_file: str = "atomic-scenarios_with_tags.json"):
        self.scenarios_file = scenarios_file
        
        # Load scenarios with tags
        with open(scenarios_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'scenarios' in data:
                self.scenarios = data['scenarios']
            else:
                self.scenarios = data
        
        # Build tag taxonomy and similarity groups
        self.tag_similarity_groups = self._build_tag_similarity_groups()
        self.domain_groups = self._group_scenarios_by_domain()
        
        print(f"✅ Loaded {len(self.scenarios)} scenarios with tags")
        print(f"📊 Found {len(self.tag_similarity_groups)} tag similarity groups")
        print(f"🏷️ Created {len(self.domain_groups)} domain groups")
    
    def _build_tag_similarity_groups(self) -> Dict[str, List[str]]:
        """Build groups of similar tags for better negative sampling."""
        tag_groups = defaultdict(list)
        
        # Group tags by semantic categories
        categories = {
            'actors': ['self', 'stranger', 'coworker_peer', 'coworker_superior', 'family_member', 
                      'sibling', 'friend', 'customer', 'service_worker', 'authority_figure'],
            'domains': ['work', 'personal', 'financial', 'health', 'social'],
            'settings': ['home_private', 'workplace_private', 'workplace_shared', 'public', 
                        'street_public', 'public_transport', 'online_group', 'library'],
            'timing': ['sudden', 'ongoing', 'past', 'unanticipated'],
            'impact': ['minor_inconvenience', 'major_inconvenience', 'material_loss_small', 
                      'material_loss_large', 'physical_harm', 'emotional_damage', 'social_damage'],
            'actions': ['taking_property', 'breaking_promise', 'betrayal', 'neglect', 'criticism',
                       'interrupting', 'blocked_goal', 'ignored'],
            'circumstances': ['uncontrollable', 'unforeseen_circumstances', 'technical_issue',
                            'property_damaged', 'safety_risk', 'confusion']
        }
        
        # Build reverse mapping
        for category, tags in categories.items():
            for tag in tags:
                tag_groups[tag].append(category)
        
        return dict(tag_groups)
    
    def _group_scenarios_by_domain(self) -> Dict[str, List[Dict]]:
        """Group scenarios by domain for better sampling."""
        domain_groups = defaultdict(list)
        
        for scenario in self.scenarios:
            tags = scenario.get('tags', [])
            
            # Determine primary domain
            domain = 'general'
            for tag in tags:
                if tag in ['work']:
                    domain = 'work'
                    break
                elif tag in ['personal']:
                    domain = 'personal'
                    break
                elif tag in ['financial']:
                    domain = 'financial'
                    break
                elif tag in ['health']:
                    domain = 'health'
                    break
            
            domain_groups[domain].append(scenario)
        
        return dict(domain_groups)
    
    def create_text_variations(self, text: str, scenario_tags: List[str]) -> List[str]:
        """Create diverse text variations using scenario context."""
        variations = [text]
        lower_text = text.lower()
        
        # Basic variations
        if not any(lower_text.startswith(prefix) for prefix in ['i ', 'my ', 'me ', 'help']):
            variations.extend([
                f"I am experiencing {lower_text}",
                f"I'm dealing with {lower_text}",
                f"Help me with {lower_text}",
                f"Can you help with {lower_text}?",
                f"What should I do about {lower_text}?"
            ])
        
        # Context-aware variations based on tags
        if any(tag in scenario_tags for tag in ['work', 'coworker_peer', 'workplace_private']):
            variations.extend([
                f"At work, {lower_text}",
                f"My workplace situation: {lower_text}",
                f"Work problem: {lower_text}"
            ])
        
        if any(tag in scenario_tags for tag in ['personal', 'home_private', 'family_member']):
            variations.extend([
                f"At home, {lower_text}",
                f"Personal issue: {lower_text}",
                f"Family situation: {lower_text}"
            ])
        
        if any(tag in scenario_tags for tag in ['sudden', 'unexpected']):
            variations.extend([
                f"Suddenly, {lower_text}",
                f"Unexpectedly, {lower_text}",
                f"Out of nowhere, {lower_text}"
            ])
        
        # Emotional variations
        if any(tag in scenario_tags for tag in ['major_inconvenience', 'betrayal', 'emotional_damage']):
            variations.extend([
                f"I'm really frustrated because {lower_text}",
                f"I'm upset that {lower_text}",
                f"This is really bothering me: {lower_text}"
            ])
        
        # Remove duplicates and limit
        unique_variations = list(dict.fromkeys(variations))  # Preserve order
        return unique_variations[:8]  # Limit to 8 variations per scenario
    
    def create_tag_based_queries(self, scenario: Dict[str, Any]) -> List[str]:
        """Generate queries based on tag combinations."""
        tags = scenario.get('tags', [])
        queries = []
        
        # Create queries that mention key tags directly
        if 'coworker_peer' in tags and 'betrayal' in tags:
            queries.append("My colleague betrayed my trust")
        
        if 'financial' in tags and 'sudden' in tags:
            queries.append("I have an unexpected financial problem")
        
        if 'public_transport' in tags and 'major_inconvenience' in tags:
            queries.append("Public transport is causing me major problems")
        
        if 'property_damaged' in tags:
            queries.append("Something I own got damaged")
        
        if 'taking_property' in tags:
            queries.append("Someone took something that belongs to me")
        
        # Create domain-specific queries
        if 'work' in tags:
            queries.extend([
                "I have a work-related issue",
                "Something happened at my job",
                "Workplace problem I'm facing"
            ])
        
        if 'home_private' in tags:
            queries.extend([
                "Issue at home",
                "Problem in my personal space"
            ])
        
        return queries[:4]  # Limit to 4 tag-based queries
    
    def generate_positive_pairs(self) -> List[Dict[str, Any]]:
        """Generate comprehensive positive training pairs."""
        print("1️⃣ Generating positive pairs...")
        positive_pairs = []
        
        for i, scenario in enumerate(self.scenarios):
            if i % 20 == 0:
                print(f"  Processing scenario {i+1}/{len(self.scenarios)}")
            
            scenario_id = scenario['id']
            scenario_desc = scenario['description']
            scenario_tags = scenario.get('tags', [])
            
            # Create text variations
            text_variations = self.create_text_variations(scenario_desc, scenario_tags)
            
            # Create tag-based queries
            tag_queries = self.create_tag_based_queries(scenario)
            
            # Combine all query types
            all_queries = text_variations + tag_queries
            
            # Create positive pairs for each query
            for query in all_queries:
                positive_pairs.append({
                    'query': query,
                    'scenario_id': scenario_id,
                    'scenario_description': scenario_desc,
                    'scenario_tags': scenario_tags,
                    'label': 1,
                    'pair_type': 'positive',
                    'similarity_type': 'exact_match'
                })
        
        print(f"Generated {len(positive_pairs)} positive pairs")
        return positive_pairs
    
    def generate_hard_negatives(self, positive_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hard negative pairs (similar but incorrect scenarios)."""
        print("2️⃣ Generating hard negative pairs...")
        hard_negatives = []
        
        # Group scenarios by tag similarity for hard negative mining
        tag_to_scenarios = defaultdict(list)
        for scenario in self.scenarios:
            for tag in scenario.get('tags', []):
                tag_to_scenarios[tag].append(scenario)
        
        for pos_pair in positive_pairs[:len(positive_pairs)//3]:  # Generate for subset
            current_scenario_id = pos_pair['scenario_id']
            current_tags = pos_pair['scenario_tags']
            
            # Find scenarios with overlapping tags but different IDs
            candidate_scenarios = []
            for tag in current_tags:
                for candidate in tag_to_scenarios[tag]:
                    if (candidate['id'] != current_scenario_id and 
                        candidate not in candidate_scenarios):
                        candidate_scenarios.append(candidate)
            
            # Select hard negatives (scenarios with some tag overlap)
            if candidate_scenarios:
                # Calculate overlap scores
                scored_candidates = []
                for candidate in candidate_scenarios:
                    candidate_tags = set(candidate.get('tags', []))
                    current_tags_set = set(current_tags)
                    overlap = len(candidate_tags & current_tags_set)
                    # We want some overlap but not too much (hard negatives)
                    if 1 <= overlap <= len(current_tags_set) // 2:
                        scored_candidates.append((candidate, overlap))
                
                # Select top hard negatives
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                for candidate, _ in scored_candidates[:2]:  # Top 2 hard negatives
                    hard_negatives.append({
                        'query': pos_pair['query'],
                        'scenario_id': candidate['id'],
                        'scenario_description': candidate['description'],
                        'scenario_tags': candidate.get('tags', []),
                        'label': 0,
                        'pair_type': 'hard_negative',
                        'similarity_type': 'tag_overlap'
                    })
        
        print(f"Generated {len(hard_negatives)} hard negative pairs")
        return hard_negatives
    
    def generate_random_negatives(self, positive_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate random negative pairs for balanced training."""
        print("3️⃣ Generating random negative pairs...")
        random_negatives = []
        
        # Create mapping for efficient random sampling
        scenario_pool = [s for s in self.scenarios]
        
        for pos_pair in positive_pairs:
            current_scenario_id = pos_pair['scenario_id']
            
            # Select random scenario that's different from current
            wrong_scenario = random.choice([s for s in scenario_pool if s['id'] != current_scenario_id])
            
            random_negatives.append({
                'query': pos_pair['query'],
                'scenario_id': wrong_scenario['id'],
                'scenario_description': wrong_scenario['description'],
                'scenario_tags': wrong_scenario.get('tags', []),
                'label': 0,
                'pair_type': 'random_negative',
                'similarity_type': 'random'
            })
        
        print(f"Generated {len(random_negatives)} random negative pairs")
        return random_negatives
    
    def generate_cross_domain_pairs(self) -> List[Dict[str, Any]]:
        """Generate pairs across different domains for better generalization."""
        print("4️⃣ Generating cross-domain pairs...")
        cross_domain_pairs = []
        
        # Create queries that could apply to multiple domains
        cross_domain_queries = [
            "Someone didn't keep their promise to me",
            "I'm dealing with an unexpected delay",
            "My plans got disrupted suddenly",
            "Someone is not responding when I need them to",
            "I'm facing a technical issue",
            "Something important to me got damaged",
            "I'm experiencing interference with my work",
            "Someone interrupted what I was doing"
        ]
        
        for query in cross_domain_queries:
            # Find scenarios that could match this query across domains
            matching_scenarios = []
            
            for scenario in self.scenarios:
                scenario_desc = scenario['description'].lower()
                query_lower = query.lower()
                
                # Simple keyword matching for demonstration
                if (any(word in scenario_desc for word in ['delay', 'wait', 'disrupt']) and 
                    'delay' in query_lower):
                    matching_scenarios.append(scenario)
                elif (any(word in scenario_desc for word in ['promise', 'commitment', 'agree']) and 
                      'promise' in query_lower):
                    matching_scenarios.append(scenario)
                elif (any(word in scenario_desc for word in ['technical', 'technology', 'device']) and 
                      'technical' in query_lower):
                    matching_scenarios.append(scenario)
                elif (any(word in scenario_desc for word in ['damaged', 'break', 'broken']) and 
                      'damaged' in query_lower):
                    matching_scenarios.append(scenario)
            
            # Create positive pairs
            for scenario in matching_scenarios[:3]:  # Limit to 3 per query
                cross_domain_pairs.append({
                    'query': query,
                    'scenario_id': scenario['id'],
                    'scenario_description': scenario['description'],
                    'scenario_tags': scenario.get('tags', []),
                    'label': 1,
                    'pair_type': 'cross_domain_positive',
                    'similarity_type': 'semantic_match'
                })
        
        print(f"Generated {len(cross_domain_pairs)} cross-domain pairs")
        return cross_domain_pairs
    
    def create_comprehensive_dataset(self, output_dir: str = "enhanced_training_data") -> Dict[str, str]:
        """Create the comprehensive training dataset."""
        print("🚀 Creating comprehensive training dataset for Llama 3.1 embedding...")
        print("=" * 70)
        
        # Generate all types of pairs
        positive_pairs = self.generate_positive_pairs()
        hard_negatives = self.generate_hard_negatives(positive_pairs)
        random_negatives = self.generate_random_negatives(positive_pairs)
        cross_domain_pairs = self.generate_cross_domain_pairs()
        
        # Combine all pairs
        all_pairs = positive_pairs + hard_negatives + random_negatives + cross_domain_pairs
        
        # Shuffle the dataset
        random.shuffle(all_pairs)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total pairs: {len(all_pairs)}")
        print(f"   Positive pairs: {len(positive_pairs + [p for p in cross_domain_pairs if p['label'] == 1])}")
        print(f"   Hard negatives: {len(hard_negatives)}")
        print(f"   Random negatives: {len(random_negatives)}")
        print(f"   Cross-domain pairs: {len(cross_domain_pairs)}")
        
        # Split into train/validation/test (70/15/15)
        train_size = int(0.70 * len(all_pairs))
        val_size = int(0.15 * len(all_pairs))
        
        train_data = all_pairs[:train_size]
        val_data = all_pairs[train_size:train_size + val_size]
        test_data = all_pairs[train_size + val_size:]
        
        print(f"\n📂 Data Splits:")
        print(f"   Training: {len(train_data)} pairs")
        print(f"   Validation: {len(val_data)} pairs")
        print(f"   Test: {len(test_data)} pairs")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save datasets with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_created = {}
        
        for split_name, split_data in [("train", train_data), ("validation", val_data), ("test", test_data)]:
            filename = os.path.join(output_dir, f"{split_name}_enhanced_{timestamp}.json")
            
            # Calculate split statistics
            pos_count = len([p for p in split_data if p['label'] == 1])
            neg_count = len([p for p in split_data if p['label'] == 0])
            
            dataset = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "dataset_version": "enhanced_v1.0",
                    "total_pairs": len(split_data),
                    "positive_pairs": pos_count,
                    "negative_pairs": neg_count,
                    "balance_ratio": f"{pos_count}:{neg_count}",
                    "scenarios_file": self.scenarios_file,
                    "split": split_name,
                    "generation_method": "enhanced_embedding_optimized",
                    "pair_types": {
                        "positive": len([p for p in split_data if p['pair_type'] == 'positive']),
                        "hard_negative": len([p for p in split_data if p['pair_type'] == 'hard_negative']),
                        "random_negative": len([p for p in split_data if p['pair_type'] == 'random_negative']),
                        "cross_domain_positive": len([p for p in split_data if p['pair_type'] == 'cross_domain_positive'])
                    }
                },
                "data": split_data
            }
            
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            files_created[split_name] = filename
            print(f"💾 Saved {split_name}: {filename}")
        
        # Save dataset summary
        summary_filename = os.path.join(output_dir, f"dataset_summary_{timestamp}.json")
        summary = {
            "dataset_info": {
                "version": "enhanced_v1.0",
                "total_scenarios": len(self.scenarios),
                "total_pairs": len(all_pairs),
                "positive_pairs": len(positive_pairs + [p for p in cross_domain_pairs if p['label'] == 1]),
                "negative_pairs": len(hard_negatives) + len(random_negatives),
                "train_pairs": len(train_data),
                "val_pairs": len(val_data),
                "test_pairs": len(test_data),
                "tag_groups": len(self.tag_similarity_groups),
                "domain_groups": len(self.domain_groups)
            },
            "files": files_created,
            "training_recommendations": {
                "embedding_model": "Llama-3.1",
                "batch_size": 32,
                "learning_rate": 5e-5,
                "max_sequence_length": 512,
                "training_objective": "contrastive_loss",
                "hard_negative_ratio": 0.3
            },
            "created_at": datetime.now().isoformat()
        }
        
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        files_created['summary'] = summary_filename
        
        print(f"\n✅ Enhanced training dataset created successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Summary file: {summary_filename}")
        
        return files_created


def main():
    """Generate enhanced training dataset for Llama 3.1 embedding fine-tuning."""
    print("🎯 ENHANCED EMBEDDING TRAINING DATASET GENERATOR")
    print("🦙 Optimized for Llama 3.1 Embedding Model")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create generator and generate dataset
    generator = EnhancedEmbeddingDatasetGenerator("atomic-scenarios_with_tags.json")
    files_created = generator.create_comprehensive_dataset()
    
    print(f"\n🎉 Enhanced dataset generation completed!")
    print(f"\n🚀 Next Steps for Llama 3.1 Fine-tuning:")
    print(f"1. Use the generated training files:")
    print(f"   --train_file {files_created['train']}")
    print(f"   --val_file {files_created['validation']}")
    print(f"   --test_file {files_created['test']}")
    print(f"\n2. Recommended training parameters:")
    print(f"   --model_name meta-llama/Llama-3.1-8B")
    print(f"   --batch_size 32")
    print(f"   --learning_rate 5e-5")
    print(f"   --max_length 512")
    print(f"   --num_epochs 3")
    print(f"\n3. Training objective: Contrastive loss with hard negatives")
    print(f"4. The dataset includes diverse pair types for robust training")


if __name__ == "__main__":
    main()
