"""
Fine-tune embedding model for improved scenario matching performance.
Uses contrastive learning to train embeddings that better distinguish between scenarios.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import random
from tqdm import tqdm
import os
from datetime import datetime

import config
from vllm import LLM


class ScenarioEmbeddingDataset(Dataset):
    """Dataset for training scenario embeddings with contrastive learning."""
    
    def __init__(self, scenarios: List[Dict], tokenizer, max_length: int = 512):
        self.scenarios = scenarios
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create scenario ID to index mapping
        self.id_to_idx = {scenario['id']: idx for idx, scenario in enumerate(scenarios)}
        
        # Pre-generate training pairs
        self.training_pairs = self._generate_training_pairs()
        
    def _generate_training_pairs(self) -> List[Tuple[int, int, int]]:
        """Generate positive and negative pairs for contrastive learning."""
        pairs = []
        
        for i, scenario in enumerate(self.scenarios):
            # Positive pairs: same scenario with variations
            # For now, we'll use the scenario with itself as positive
            # In practice, you'd want variations/paraphrases
            pairs.append((i, i, 1))  # (anchor, positive, label)
            
            # Negative pairs: different scenarios
            for _ in range(3):  # 3 negative samples per positive
                neg_idx = random.choice([j for j in range(len(self.scenarios)) if j != i])
                pairs.append((i, neg_idx, 0))  # (anchor, negative, label)
        
        return pairs
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        anchor_idx, other_idx, label = self.training_pairs[idx]
        
        anchor_text = self.scenarios[anchor_idx]['description']
        other_text = self.scenarios[other_idx]['description']
        
        # Tokenize texts
        anchor_tokens = self.tokenizer(
            anchor_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        other_tokens = self.tokenizer(
            other_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor_tokens['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_tokens['attention_mask'].squeeze(),
            'other_input_ids': other_tokens['input_ids'].squeeze(),
            'other_attention_mask': other_tokens['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class ContrastiveEmbeddingModel(nn.Module):
    """Wrapper around base model for contrastive learning."""
    
    def __init__(self, base_model, embedding_dim: int = 4096):
        super().__init__()
        self.base_model = base_model
        self.embedding_dim = embedding_dim
        
        # Add a projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, 256)  # Final embedding size
        )
        
    def forward(self, input_ids, attention_mask):
        # Get embeddings from base model
        # Note: This is a simplified version - you'd need to adapt based on your model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use mean pooling of last hidden states
        embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Apply projection head
        projected = self.projection_head(mean_embeddings)
        
        return projected


class ContrastiveLoss(nn.Module):
    """Contrastive loss for training embeddings."""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, anchor_emb, other_emb, labels):
        # Compute cosine similarity
        anchor_norm = torch.nn.functional.normalize(anchor_emb, p=2, dim=1)
        other_norm = torch.nn.functional.normalize(other_emb, p=2, dim=1)
        
        similarity = torch.sum(anchor_norm * other_norm, dim=1)
        
        # Contrastive loss
        positive_loss = labels * (1 - similarity)
        negative_loss = (1 - labels) * torch.clamp(similarity - self.margin, min=0)
        
        loss = positive_loss + negative_loss
        return loss.mean()


def load_scenarios(file_path: str) -> List[Dict]:
    """Load scenarios from JSON file."""
    with open(file_path, 'r') as f:
        scenarios = json.load(f)
    
    # Ensure each scenario has required fields
    processed_scenarios = []
    for scenario in scenarios:
        if 'id' in scenario and 'description' in scenario:
            processed_scenarios.append(scenario)
    
    return processed_scenarios


def create_training_variations(scenarios: List[Dict], llm_model) -> List[Dict]:
    """Generate training variations of scenarios using LLM."""
    print("🔄 Generating training variations...")
    
    variations = []
    
    for scenario in tqdm(scenarios[:10]):  # Limit for demo
        # Create prompt for generating variations
        prompt = f"""Generate 3 different ways to describe this emotional scenario while keeping the core meaning:

Original: {scenario['description']}

Generate 3 variations that describe the same situation but with different wording:
1. 
2. 
3. """

        try:
            # Generate variations (simplified - you'd use your LLM here)
            # For now, just create simple variations
            base_desc = scenario['description']
            
            variations.extend([
                {
                    'id': f"{scenario['id']}_var1",
                    'description': base_desc,  # Original
                    'original_id': scenario['id']
                },
                {
                    'id': f"{scenario['id']}_var2", 
                    'description': f"A situation where {base_desc.lower()}",
                    'original_id': scenario['id']
                },
                {
                    'id': f"{scenario['id']}_var3",
                    'description': f"When {base_desc.lower()}",
                    'original_id': scenario['id']
                }
            ])
            
        except Exception as e:
            print(f"Error generating variations for {scenario['id']}: {e}")
            # Fallback to original
            variations.append(scenario)
    
    return variations


def fine_tune_embeddings(
    scenarios_file: str = "atomic-scenarios.json",
    output_dir: str = "fine_tuned_embeddings",
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-5
):
    """Fine-tune embeddings for scenario matching."""
    
    print("🚀 Starting embedding fine-tuning...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load scenarios
    print("📚 Loading scenarios...")
    scenarios = load_scenarios(scenarios_file)
    print(f"Loaded {len(scenarios)} scenarios")
    
    # Initialize model (simplified - you'd load your actual model here)
    print("🤖 Loading base model...")
    # Note: This is a placeholder - you'd need to adapt for your specific model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For demonstration, we'll create a simple training loop structure
    # You'd need to adapt this for your specific model architecture
    
    print("📊 Creating training dataset...")
    # Create dataset with variations
    training_scenarios = create_training_variations(scenarios, None)
    
    # Save training configuration
    config_data = {
        'timestamp': datetime.now().isoformat(),
        'scenarios_file': scenarios_file,
        'num_scenarios': len(scenarios),
        'num_training_samples': len(training_scenarios),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': str(device)
    }
    
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"✅ Training configuration saved to {output_dir}/training_config.json")
    print(f"📈 Ready to train on {len(training_scenarios)} samples")
    
    # TODO: Implement actual training loop
    print("⚠️  Note: This is a framework - actual training implementation depends on your model architecture")
    
    return output_dir


def evaluate_fine_tuned_model(model_path: str, test_scenarios: List[Dict]):
    """Evaluate the fine-tuned model on test scenarios."""
    print("🧪 Evaluating fine-tuned model...")
    
    # Load fine-tuned model
    # TODO: Implement model loading and evaluation
    
    print("📊 Evaluation complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune embeddings for scenario matching")
    parser.add_argument('--scenarios', default='atomic-scenarios.json', help='Scenarios file')
    parser.add_argument('--output-dir', default='fine_tuned_embeddings', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    fine_tune_embeddings(
        scenarios_file=args.scenarios,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
