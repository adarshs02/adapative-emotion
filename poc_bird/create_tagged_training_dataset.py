"""
Create training dataset with proper tag generation for scenario + tag joint embeddings.
"""

import json
import random
import os
from datetime import datetime
from typing import List, Dict, Any
from tag_generator import TagGenerator


class TaggedTrainingDatasetGenerator:
    """Generate training dataset with proper tag integration."""
    
    def __init__(self, scenarios_file: str = "atomic-scenarios.json"):
        self.scenarios_file = scenarios_file
        self.tag_generator = TagGenerator()
        
        # Load scenarios
        with open(scenarios_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'scenarios' in data:
                self.scenarios = data['scenarios']
            else:
                self.scenarios = data
        
        print(f"Loaded {len(self.scenarios)} scenarios for tagged training dataset generation")
    
    def create_joint_representation_with_tags(self, text: str, scenario_id: str = None) -> Dict[str, Any]:
        """Create joint representation with text + tags."""
        # Generate tags
        print(f"  Generating tags for: {text[:50]}...")
        tags = self.tag_generator.generate_tags(text)
        
        # Create joint representation
        joint_parts = [text]
        if tags:
            # Add natural language context
            tag_text = self._tags_to_natural_language(tags)
            joint_parts.append(f"Context: {tag_text}")
            # Add raw tags
            joint_parts.append(f"Tags: {', '.join(tags)}")
        
        joint_text = ". ".join(joint_parts)
        
        return {
            'original_text': text,
            'tags': tags,
            'tag_context': self._tags_to_natural_language(tags) if tags else "",
            'joint_representation': joint_text
        }
    
    def _tags_to_natural_language(self, tags: List[str]) -> str:
        """Convert tags to natural language description."""
        if not tags:
            return ""
        
        # Group tags by type
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
        
        descriptions = []
        for group, group_tags in tag_groups.items():
            if group == 'general':
                descriptions.append(f"involving {', '.join(group_tags)}")
            else:
                descriptions.append(f"{group}: {', '.join(group_tags)}")
        
        return "; ".join(descriptions)
    
    def create_text_variations(self, text: str) -> List[str]:
        """Create rule-based variations of text."""
        variations = [text]  # Original text
        
        # Simple rule-based transformations
        lower_text = text.lower()
        
        # Add first-person variations (limit to 2 to reduce dataset size)
        if not any(lower_text.startswith(prefix) for prefix in ['i ', 'my ', 'me ']):
            variations.extend([
                f"I am experiencing {lower_text}",
                f"Help me with {lower_text}"
            ])
        
        # Add question variation
        variations.append(f"What should I do about {lower_text}?")
        
        # Remove duplicates and limit
        unique_variations = []
        seen = set()
        for var in variations:
            if var not in seen:
                unique_variations.append(var)
                seen.add(var)
        
        return unique_variations[:3]  # Limit to 3 variations per scenario
    
    def generate_positive_pairs(self) -> List[Dict[str, Any]]:
        """Generate positive pairs (query matches correct scenario) with tags."""
        print("1️⃣ Generating positive pairs with tags...")
        positive_pairs = []
        
        for i, scenario in enumerate(self.scenarios):
            print(f"Processing scenario {i+1}/{len(self.scenarios)}: {scenario['id']}")
            
            scenario_id = scenario['id']
            scenario_desc = scenario['description']
            
            # Create joint representation of the scenario with tags
            scenario_joint_data = self.create_joint_representation_with_tags(scenario_desc, scenario_id)
            
            # Create rule-based variations
            query_variations = self.create_text_variations(scenario_desc)
            
            # Create positive pairs
            for query in query_variations:
                query_joint_data = self.create_joint_representation_with_tags(query)
                
                positive_pairs.append({
                    'query': query,
                    'query_tags': query_joint_data['tags'],
                    'query_joint': query_joint_data['joint_representation'],
                    'scenario_id': scenario_id,
                    'scenario_description': scenario_desc,
                    'scenario_tags': scenario_joint_data['tags'],
                    'scenario_joint': scenario_joint_data['joint_representation'],
                    'label': 1,  # Positive pair
                    'pair_type': 'positive'
                })
            
            # Print progress every 10 scenarios
            if (i + 1) % 10 == 0:
                print(f"  ✅ Completed {i + 1}/{len(self.scenarios)} scenarios")
        
        print(f"Generated {len(positive_pairs)} positive pairs with tags")
        return positive_pairs
    
    def generate_negative_pairs(self, positive_pairs: List[Dict[str, Any]], 
                               negative_ratio: float = 0.5) -> List[Dict[str, Any]]:
        """Generate negative pairs (query doesn't match scenario) with tags."""
        print("2️⃣ Generating negative pairs...")
        
        negative_pairs = []
        num_negatives = int(len(positive_pairs) * negative_ratio)
        
        for i in range(num_negatives):
            if i % 100 == 0:
                print(f"  Generated {i}/{num_negatives} negative pairs...")
            
            # Pick a random positive pair
            pos_pair = random.choice(positive_pairs)
            
            # Pick a different scenario
            wrong_scenario = random.choice(self.scenarios)
            while wrong_scenario['id'] == pos_pair['scenario_id']:
                wrong_scenario = random.choice(self.scenarios)
            
            # Use existing scenario joint representation if available
            wrong_scenario_joint_data = self.create_joint_representation_with_tags(
                wrong_scenario['description'], wrong_scenario['id']
            )
            
            negative_pairs.append({
                'query': pos_pair['query'],
                'query_tags': pos_pair['query_tags'],
                'query_joint': pos_pair['query_joint'],
                'scenario_id': wrong_scenario['id'],
                'scenario_description': wrong_scenario['description'],
                'scenario_tags': wrong_scenario_joint_data['tags'],
                'scenario_joint': wrong_scenario_joint_data['joint_representation'],
                'label': 0,  # Negative pair
                'pair_type': 'negative'
            })
        
        print(f"Generated {len(negative_pairs)} negative pairs")
        return negative_pairs
    
    def create_training_dataset(self) -> List[str]:
        """Create complete training dataset with tags."""
        print("🚀 Creating tagged training dataset...")
        
        # Generate positive and negative pairs
        positive_pairs = self.generate_positive_pairs()
        negative_pairs = self.generate_negative_pairs(positive_pairs)
        
        # Combine all pairs
        all_pairs = positive_pairs + negative_pairs
        
        # Shuffle the dataset
        random.shuffle(all_pairs)
        
        print(f"3️⃣ Total dataset size: {len(all_pairs)} pairs")
        print(f"   Positive pairs: {len(positive_pairs)}")
        print(f"   Negative pairs: {len(negative_pairs)}")
        
        # Split into train/validation/test
        train_size = int(0.7 * len(all_pairs))
        val_size = int(0.15 * len(all_pairs))
        
        train_data = all_pairs[:train_size]
        val_data = all_pairs[train_size:train_size + val_size]
        test_data = all_pairs[train_size + val_size:]
        
        print(f"   Train: {len(train_data)} pairs")
        print(f"   Validation: {len(val_data)} pairs")
        print(f"   Test: {len(test_data)} pairs")
        
        # Create output directory
        output_dir = "training_data_tagged"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save datasets
        timestamp = int(datetime.now().timestamp())
        files_created = []
        
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            filename = f"{output_dir}/{split_name}_dataset_tagged_{timestamp}.json"
            
            dataset = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_pairs": len(split_data),
                    "positive_pairs": len([p for p in split_data if p['label'] == 1]),
                    "negative_pairs": len([p for p in split_data if p['label'] == 0]),
                    "scenarios_file": self.scenarios_file,
                    "split": split_name,
                    "generation_method": "tagged_joint_representation",
                    "includes_tags": True
                },
                "data": split_data
            }
            
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            files_created.append(filename)
            print(f"💾 Saved {split_name} dataset: {filename}")
        
        print(f"✅ Tagged training dataset creation completed!")
        print(f"📁 Files created: {files_created}")
        
        return files_created


def main():
    """Main function to create tagged training dataset."""
    print("🎯 TAGGED TRAINING DATASET GENERATOR")
    print("=" * 50)
    print("Creating dataset with scenario descriptions + tags")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create generator and generate dataset
    generator = TaggedTrainingDatasetGenerator()
    files_created = generator.create_training_dataset()
    
    print("\n🎉 Tagged dataset generation completed successfully!")
    print("\n📊 Dataset Statistics:")
    
    # Show sample data
    if files_created:
        train_file = [f for f in files_created if 'train_' in f][0]
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        print(f"   Total training pairs: {train_data['metadata']['total_pairs']}")
        print(f"   Positive pairs: {train_data['metadata']['positive_pairs']}")
        print(f"   Negative pairs: {train_data['metadata']['negative_pairs']}")
        
        # Show a few examples with tags
        print("\n🔍 Sample positive pair with tags:")
        pos_example = next(p for p in train_data['data'] if p['label'] == 1)
        print(f"   Query: {pos_example['query'][:100]}...")
        print(f"   Query Tags: {pos_example['query_tags']}")
        print(f"   Scenario: {pos_example['scenario_id']}")
        print(f"   Scenario Tags: {pos_example['scenario_tags']}")
        print(f"   Joint Query: {pos_example['query_joint'][:150]}...")
        
        print("\n🔍 Sample negative pair with tags:")
        neg_example = next(p for p in train_data['data'] if p['label'] == 0)
        print(f"   Query: {neg_example['query'][:100]}...")
        print(f"   Query Tags: {neg_example['query_tags']}")
        print(f"   Wrong Scenario: {neg_example['scenario_id']}")
        print(f"   Wrong Scenario Tags: {neg_example['scenario_tags']}")
    
    print("\n🚀 Next steps:")
    print("1. Use these tagged files for fine-tuning:")
    for file in files_created:
        if 'train_' in file:
            print(f"   --train_file {file}")
        elif 'val_' in file:
            print(f"   --val_file {file}")
    print("2. The model will learn joint embeddings of text + tags")
    print("3. This should improve scenario matching accuracy significantly")


if __name__ == "__main__":
    main()
