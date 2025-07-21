"""
Create a comprehensive training dataset for fine-tuning embedding models on scenario matching.
Generates positive and negative pairs with various augmentations.
"""

import json
import torch
import numpy as np
from typing import List, Dict, Any
import os
from datetime import datetime
import random
import re
from tag_generator import TagGenerator


class ScenarioTrainingDatasetGenerator:
    """Generate training dataset for scenario matching with embeddings."""
    
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
    
    def generate_paraphrases(self, text: str, num_paraphrases: int = 2) -> List[str]:
        """Generate paraphrases of the given text."""
        try:
            prompt = f"""Generate {num_paraphrases} different paraphrases of the following text. Each paraphrase should convey the same meaning but use different words and sentence structure.

Original text: {text}

Provide the paraphrases as a simple list, one per line:"""
            
            # Use TagGenerator's model for paraphrasing
            messages = [{"role": "user", "content": prompt}]
            full_prompt = self.tag_generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tag_generator.tokenizer(full_prompt, return_tensors="pt").to(self.tag_generator.device)
            
            with torch.no_grad():
                outputs = self.tag_generator.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tag_generator.tokenizer.eos_token_id
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tag_generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Parse paraphrases from response
            paraphrases = []
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering and bullet points
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^[-*]\s*', '', line)
                if line and len(line) > 10:  # Reasonable minimum length
                    paraphrases.append(line)
                    if len(paraphrases) >= num_paraphrases:
                        break
            
            return paraphrases[:num_paraphrases]
            
        except Exception as e:
            print(f"Error generating paraphrases: {e}")
            return []
    
    def generate_user_queries(self, scenario_description: str, num_queries: int = 2) -> List[str]:
        """Generate realistic user queries that would match this scenario."""
        try:
            prompt = f"""Generate {num_queries} realistic user queries or questions that someone might ask when experiencing this situation:

Scenario: {scenario_description}

The queries should be natural, conversational, and something a real person might say when seeking help or describing their situation. Provide them as a simple list, one per line:"""
            
            # Use TagGenerator's model for query generation
            messages = [{"role": "user", "content": prompt}]
            full_prompt = self.tag_generator.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tag_generator.tokenizer(full_prompt, return_tensors="pt").to(self.tag_generator.device)
            
            with torch.no_grad():
                outputs = self.tag_generator.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tag_generator.tokenizer.eos_token_id
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tag_generator.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Parse queries from response
            queries = []
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering and bullet points
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^[-*]\s*', '', line)
                if line and len(line) > 5:  # Reasonable minimum length
                    queries.append(line)
                    if len(queries) >= num_queries:
                        break
            
            return queries[:num_queries]
            
        except Exception as e:
            print(f"Error generating user queries: {e}")
            return []
    
    def create_joint_representation(self, text: str, scenario_id: str = None) -> str:
        """Create joint representation of text + tags (same as joint embedding router)."""
        # Generate tags for the text
        tags = self.tag_generator.generate_tags(text)
        
        # Create structured joint representation
        joint_parts = [text]
        
        if tags:
            # Convert tags to natural language
            tag_text = self._tags_to_natural_language(tags)
            joint_parts.append(f"Context: {tag_text}")
            
            # Also add raw tags for exact matching
            joint_parts.append(f"Tags: {', '.join(tags)}")
        
        return ". ".join(joint_parts)
    
    def _tags_to_natural_language(self, tags: List[str]) -> str:
        """Convert tags to natural language description (same as joint embedding router)."""
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
    
    def generate_positive_pairs(self) -> List[Dict[str, Any]]:
        """Generate positive training pairs (query, scenario) that should match."""
        positive_pairs = []
        
        for i, scenario in enumerate(self.scenarios):
            print(f"Generating positive pairs for scenario {i+1}/{len(self.scenarios)}: {scenario['id']}")
            
            scenario_id = scenario['id']
            scenario_desc = scenario['description']
            
            # Create joint representation of the scenario
            scenario_joint = self.create_joint_representation(scenario_desc, scenario_id)
            
            # Generate paraphrases and user queries for this scenario (reduced for speed)
            paraphrases = self.generate_paraphrases(scenario_desc, num_paraphrases=1)
            user_queries = self.generate_user_queries(scenario_desc, num_queries=1)
            
            # Create positive pairs
            all_queries = [scenario_desc] + paraphrases + user_queries
            
            for query in all_queries:
                if query and len(query.strip()) > 10:
                    query_joint = self.create_joint_representation(query)
                    
                    positive_pairs.append({
                        'query': query.strip(),
                        'query_joint': query_joint,
                        'scenario_id': scenario_id,
                        'scenario_description': scenario_desc,
                        'scenario_joint': scenario_joint,
                        'label': 1,  # Positive pair
                        'pair_type': 'positive'
                    })
        
        print(f"Generated {len(positive_pairs)} positive pairs")
        return positive_pairs
    
    def generate_negative_pairs(self, positive_pairs: List[Dict], num_negatives_per_positive: int = 2) -> List[Dict[str, Any]]:
        """Generate negative training pairs (query, wrong_scenario) that should not match."""
        negative_pairs = []
        
        # Create a mapping of scenario_id to scenario data
        scenario_lookup = {s['id']: s for s in self.scenarios}
        
        for pos_pair in positive_pairs:
            query = pos_pair['query']
            query_joint = pos_pair['query_joint']
            correct_scenario_id = pos_pair['scenario_id']
            
            # Select random scenarios that are NOT the correct one
            other_scenarios = [s for s in self.scenarios if s['id'] != correct_scenario_id]
            negative_scenarios = random.sample(other_scenarios, min(num_negatives_per_positive, len(other_scenarios)))
            
            for neg_scenario in negative_scenarios:
                neg_scenario_joint = self.create_joint_representation(neg_scenario['description'], neg_scenario['id'])
                
                negative_pairs.append({
                    'query': query,
                    'query_joint': query_joint,
                    'scenario_id': neg_scenario['id'],
                    'scenario_description': neg_scenario['description'],
                    'scenario_joint': neg_scenario_joint,
                    'label': 0,  # Negative pair
                    'pair_type': 'negative'
                })
        
        print(f"Generated {len(negative_pairs)} negative pairs")
        return negative_pairs
    
    def create_training_dataset(self, output_dir: str = "training_data") -> Dict[str, str]:
        """Create complete training dataset with positive and negative pairs."""
        print("🚀 Creating comprehensive training dataset...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate positive pairs
        print("\n1️⃣ Generating positive pairs...")
        positive_pairs = self.generate_positive_pairs()
        
        # Generate negative pairs
        print("\n2️⃣ Generating negative pairs...")
        negative_pairs = self.generate_negative_pairs(positive_pairs, num_negatives_per_positive=2)
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Split into train/validation/test
        total_pairs = len(all_pairs)
        train_size = int(0.7 * total_pairs)
        val_size = int(0.15 * total_pairs)
        
        train_pairs = all_pairs[:train_size]
        val_pairs = all_pairs[train_size:train_size + val_size]
        test_pairs = all_pairs[train_size + val_size:]
        
        # Save datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files_created = {}
        
        # Save training data
        train_file = os.path.join(output_dir, f"train_dataset_{timestamp}.json")
        with open(train_file, 'w') as f:
            json.dump({
                'metadata': {
                    'total_pairs': len(train_pairs),
                    'positive_pairs': len([p for p in train_pairs if p['label'] == 1]),
                    'negative_pairs': len([p for p in train_pairs if p['label'] == 0]),
                    'created_at': datetime.now().isoformat(),
                    'split': 'train'
                },
                'data': train_pairs
            }, f, indent=2)
        files_created['train'] = train_file
        
        # Save validation data
        val_file = os.path.join(output_dir, f"val_dataset_{timestamp}.json")
        with open(val_file, 'w') as f:
            json.dump({
                'metadata': {
                    'total_pairs': len(val_pairs),
                    'positive_pairs': len([p for p in val_pairs if p['label'] == 1]),
                    'negative_pairs': len([p for p in val_pairs if p['label'] == 0]),
                    'created_at': datetime.now().isoformat(),
                    'split': 'validation'
                },
                'data': val_pairs
            }, f, indent=2)
        files_created['validation'] = val_file
        
        # Save test data
        test_file = os.path.join(output_dir, f"test_dataset_{timestamp}.json")
        with open(test_file, 'w') as f:
            json.dump({
                'metadata': {
                    'total_pairs': len(test_pairs),
                    'positive_pairs': len([p for p in test_pairs if p['label'] == 1]),
                    'negative_pairs': len([p for p in test_pairs if p['label'] == 0]),
                    'created_at': datetime.now().isoformat(),
                    'split': 'test'
                },
                'data': test_pairs
            }, f, indent=2)
        files_created['test'] = test_file
        
        # Save summary
        summary_file = os.path.join(output_dir, f"dataset_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump({
                'dataset_info': {
                    'total_scenarios': len(self.scenarios),
                    'total_pairs': total_pairs,
                    'positive_pairs': len(positive_pairs),
                    'negative_pairs': len(negative_pairs),
                    'train_pairs': len(train_pairs),
                    'val_pairs': len(val_pairs),
                    'test_pairs': len(test_pairs)
                },
                'files': files_created,
                'created_at': datetime.now().isoformat()
            }, f, indent=2)
        files_created['summary'] = summary_file
        
        print(f"\n✅ Training dataset created successfully!")
        print(f"📊 Dataset Statistics:")
        print(f"   Total pairs: {total_pairs}")
        print(f"   Positive pairs: {len(positive_pairs)} ({len(positive_pairs)/total_pairs:.1%})")
        print(f"   Negative pairs: {len(negative_pairs)} ({len(negative_pairs)/total_pairs:.1%})")
        print(f"   Train: {len(train_pairs)} pairs")
        print(f"   Validation: {len(val_pairs)} pairs")
        print(f"   Test: {len(test_pairs)} pairs")
        print(f"\n📁 Files created:")
        for split, filepath in files_created.items():
            print(f"   {split.capitalize()}: {filepath}")
        
        return files_created


def main():
    """Generate training dataset for scenario matching."""
    generator = ScenarioTrainingDatasetGenerator()
    files_created = generator.create_training_dataset()
    
    print("\n🎯 Next steps:")
    print("1. Review the generated training data")
    print("2. Fine-tune Llama embedding model using this data")
    print("3. Fine-tune Qwen embedding model using this data")
    print("4. Compare performance of both fine-tuned models")


if __name__ == "__main__":
    main()
