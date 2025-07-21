"""
Create a practical training dataset without relying on LLM generation.
Use rule-based variations and existing data.
"""

import json
import random
import os
from datetime import datetime
from typing import List, Dict, Any


class PracticalTrainingDatasetGenerator:
    """Generate training dataset using rule-based approaches, no LLM generation."""
    
    def __init__(self, scenarios_file: str = "atomic-scenarios.json"):
        self.scenarios_file = scenarios_file
        
        # Load scenarios
        with open(scenarios_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'scenarios' in data:
                self.scenarios = data['scenarios']
            else:
                self.scenarios = data
        
        print(f"Loaded {len(self.scenarios)} scenarios for training dataset generation")
    
    def create_simple_joint_representation(self, text: str, scenario_id: str = None) -> str:
        """Create joint representation without tag generation - just use the text."""
        # For now, just use the text as-is since tag generation is slow
        # In a real fine-tuning scenario, we'd want to include tags, but let's start simple
        return text
    
    def create_text_variations(self, text: str) -> List[str]:
        """Create rule-based variations of text."""
        variations = [text]  # Original text
        
        # Simple rule-based transformations
        lower_text = text.lower()
        
        # Add first-person variations
        if not any(lower_text.startswith(prefix) for prefix in ['i ', 'my ', 'me ']):
            variations.extend([
                f"I am experiencing {lower_text}",
                f"I'm dealing with {lower_text}",
                f"My situation involves {lower_text}",
                f"Help me with {lower_text}"
            ])
        
        # Add question variations
        variations.extend([
            f"What should I do about {lower_text}?",
            f"How do I handle {lower_text}?",
            f"Can you help with {lower_text}?"
        ])
        
        # Add emotional variations
        variations.extend([
            f"I'm frustrated by {lower_text}",
            f"I'm upset about {lower_text}",
            f"This {lower_text} is bothering me"
        ])
        
        # Remove duplicates and limit
        unique_variations = []
        seen = set()
        for var in variations:
            if var not in seen:
                unique_variations.append(var)
                seen.add(var)
        
        return unique_variations[:5]  # Limit to 5 variations per scenario
    
    def generate_positive_pairs(self) -> List[Dict[str, Any]]:
        """Generate positive pairs (query matches correct scenario)."""
        print("1️⃣ Generating positive pairs...")
        positive_pairs = []
        
        for i, scenario in enumerate(self.scenarios):
            if i % 50 == 0:
                print(f"  Processing scenario {i+1}/{len(self.scenarios)}")
            
            scenario_id = scenario['id']
            scenario_desc = scenario['description']
            
            # Create joint representation of the scenario
            scenario_joint = self.create_simple_joint_representation(scenario_desc, scenario_id)
            
            # Create rule-based variations
            query_variations = self.create_text_variations(scenario_desc)
            
            # Create positive pairs
            for query in query_variations:
                query_joint = self.create_simple_joint_representation(query)
                
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
        
        for i in range(num_negatives):
            if i % 500 == 0:
                print(f"  Generated {i}/{num_negatives} negative pairs...")
            
            # Pick a random positive pair
            pos_pair = random.choice(positive_pairs)
            
            # Pick a different scenario
            wrong_scenario = random.choice(self.scenarios)
            while wrong_scenario['id'] == pos_pair['scenario_id']:
                wrong_scenario = random.choice(self.scenarios)
            
            # Create negative pair
            wrong_scenario_joint = self.create_simple_joint_representation(
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
        print("🚀 Creating practical training dataset (no LLM generation)...")
        
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
                    "split": split_name,
                    "generation_method": "rule_based_no_llm"
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
    print("🎯 PRACTICAL TRAINING DATASET GENERATOR")
    print("=" * 50)
    print("Using rule-based variations, no LLM generation")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create generator and generate dataset
    generator = PracticalTrainingDatasetGenerator()
    files_created = generator.create_training_dataset()
    
    print("\n🎉 Dataset generation completed successfully!")
    print("\n📊 Dataset Statistics:")
    
    # Show sample data
    if files_created:
        train_file = [f for f in files_created if 'train_' in f][0]
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        print(f"   Total training pairs: {train_data['metadata']['total_pairs']}")
        print(f"   Positive pairs: {train_data['metadata']['positive_pairs']}")
        print(f"   Negative pairs: {train_data['metadata']['negative_pairs']}")
        
        # Show a few examples
        print("\n🔍 Sample positive pair:")
        pos_example = next(p for p in train_data['data'] if p['label'] == 1)
        print(f"   Query: {pos_example['query'][:100]}...")
        print(f"   Scenario: {pos_example['scenario_id']}")
        
        print("\n🔍 Sample negative pair:")
        neg_example = next(p for p in train_data['data'] if p['label'] == 0)
        print(f"   Query: {neg_example['query'][:100]}...")
        print(f"   Wrong Scenario: {neg_example['scenario_id']}")
    
    print("\n🚀 Next steps:")
    print("1. Use these files for fine-tuning:")
    for file in files_created:
        if 'train_' in file:
            print(f"   --train_file {file}")
        elif 'val_' in file:
            print(f"   --val_file {file}")
    print("2. Run the fine-tuning script with these files")
    print("3. The model will learn to distinguish between matching and non-matching pairs")


if __name__ == "__main__":
    main()
