"""
Create a simplified training dataset for scenario matching without slow LLM generation.
Focus on speed and essential functionality.
"""

import json
import random
import os
from datetime import datetime
from typing import List, Dict, Any
from tag_generator import TagGenerator


class SimpleTrainingDatasetGenerator:
    """Generate training dataset efficiently without extensive LLM calls."""
    
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
        
        print(f"Loaded {len(self.scenarios)} scenarios for training dataset generation")
    
    def create_joint_representation(self, text: str, scenario_id: str = None) -> str:
        """Create joint representation with text + tags."""
        # Generate tags
        tags = self.tag_generator.generate_tags(text)
        
        # Create joint representation
        joint_parts = [text]
        if tags:
            # Add natural language context
            tag_text = self._tags_to_natural_language(tags)
            joint_parts.append(f"Context: {tag_text}")
            # Add raw tags
            joint_parts.append(f"Tags: {', '.join(tags)}")
        
        return ". ".join(joint_parts)
    
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
    
    def create_simple_variations(self, text: str) -> List[str]:
        """Create simple variations of text without LLM calls."""
        variations = []
        
        # Add the original text
        variations.append(text)
        
        # Create simple variations by:
        # 1. Adding "I am experiencing" prefix
        if not text.lower().startswith(('i ', 'my ', 'me ')):
            variations.append(f"I am experiencing {text.lower()}")
        
        # 2. Adding "Help with" prefix
        variations.append(f"Help with {text.lower()}")
        
        # 3. Adding question format
        variations.append(f"What should I do about {text.lower()}?")
        
        # 4. Adding emotional context
        variations.append(f"I'm dealing with {text.lower()}")
        
        return variations[:3]  # Limit to 3 variations
    
    def generate_positive_pairs(self) -> List[Dict[str, Any]]:
        """Generate positive pairs (query matches correct scenario)."""
        print("1️⃣ Generating positive pairs...")
        positive_pairs = []
        
        for i, scenario in enumerate(self.scenarios):
            print(f"Generating positive pairs for scenario {i+1}/{len(self.scenarios)}: {scenario['id']}")
            
            scenario_id = scenario['id']
            scenario_desc = scenario['description']
            
            # Create joint representation of the scenario
            scenario_joint = self.create_joint_representation(scenario_desc, scenario_id)
            
            # Create simple variations instead of LLM-generated paraphrases
            query_variations = self.create_simple_variations(scenario_desc)
            
            # Create positive pairs
            for query in query_variations:
                query_joint = self.create_joint_representation(query)
                
                positive_pairs.append({
                    'query': query,
                    'query_joint': query_joint,
                    'scenario_id': scenario_id,
                    'scenario_description': scenario_desc,
                    'scenario_joint': scenario_joint,
                    'label': 1,  # Positive pair
                    'pair_type': 'positive'
                })
        
        print(f"Generated {len(positive_pairs)} positive pairs")
        return positive_pairs
    
    def generate_negative_pairs(self, positive_pairs: List[Dict[str, Any]], 
                               negative_ratio: float = 1.0) -> List[Dict[str, Any]]:
        """Generate negative pairs (query doesn't match scenario)."""
        print("2️⃣ Generating negative pairs...")
        
        negative_pairs = []
        num_negatives = int(len(positive_pairs) * negative_ratio)
        
        # Create scenario lookup
        scenario_lookup = {s['id']: s for s in self.scenarios}
        
        for i in range(num_negatives):
            if i % 100 == 0:
                print(f"  Generated {i}/{num_negatives} negative pairs...")
            
            # Pick a random positive pair
            pos_pair = random.choice(positive_pairs)
            
            # Pick a different scenario
            wrong_scenario = random.choice(self.scenarios)
            while wrong_scenario['id'] == pos_pair['scenario_id']:
                wrong_scenario = random.choice(self.scenarios)
            
            # Create negative pair
            wrong_scenario_joint = self.create_joint_representation(
                wrong_scenario['description'], wrong_scenario['id']
            )
            
            negative_pairs.append({
                'query': pos_pair['query'],
                'query_joint': pos_pair['query_joint'],
                'scenario_id': wrong_scenario['id'],
                'scenario_description': wrong_scenario['description'],
                'scenario_joint': wrong_scenario_joint,
                'label': 0,  # Negative pair
                'pair_type': 'negative'
            })
        
        print(f"Generated {len(negative_pairs)} negative pairs")
        return negative_pairs
    
    def create_training_dataset(self) -> List[str]:
        """Create complete training dataset."""
        print("🚀 Creating simple training dataset...")
        
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
        output_dir = "training_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save datasets
        timestamp = int(datetime.now().timestamp())
        files_created = []
        
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            filename = f"{output_dir}/{split_name}_dataset_{timestamp}.json"
            
            dataset = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_pairs": len(split_data),
                    "positive_pairs": len([p for p in split_data if p['label'] == 1]),
                    "negative_pairs": len([p for p in split_data if p['label'] == 0]),
                    "scenarios_file": self.scenarios_file,
                    "split": split_name
                },
                "data": split_data
            }
            
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            files_created.append(filename)
            print(f"💾 Saved {split_name} dataset: {filename}")
        
        print(f"✅ Training dataset creation completed!")
        print(f"📁 Files created: {files_created}")
        
        return files_created


def main():
    """Main function to create training dataset."""
    print("🎯 SIMPLE TRAINING DATASET GENERATOR")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create generator and generate dataset
    generator = SimpleTrainingDatasetGenerator()
    files_created = generator.create_training_dataset()
    
    print("\n🎉 Dataset generation completed successfully!")
    print("Next steps:")
    print("1. Use the generated files for fine-tuning:")
    for file in files_created:
        if 'train_' in file:
            print(f"   --train_file {file}")
        elif 'val_' in file:
            print(f"   --val_file {file}")
    print("2. Run the fine-tuning script with these files")


if __name__ == "__main__":
    main()
