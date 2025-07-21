"""
Fine-tune embedding models (Llama and Qwen) for scenario matching using contrastive learning.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Any, Tuple
import os
from datetime import datetime
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse


class ScenarioMatchingDataset(Dataset):
    """Dataset for scenario matching with contrastive learning."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r') as f:
            dataset = json.load(f)
            self.data = dataset['data']
        
        print(f"Loaded {len(self.data)} pairs from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query and scenario joint representations
        query_encoding = self.tokenizer(
            item['query_joint'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        scenario_encoding = self.tokenizer(
            item['scenario_joint'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'scenario_input_ids': scenario_encoding['input_ids'].squeeze(),
            'scenario_attention_mask': scenario_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(item['label'], dtype=torch.float),
            'scenario_id': item['scenario_id']
        }


class ContrastiveEmbeddingModel(nn.Module):
    """Model for contrastive learning of embeddings."""
    
    def __init__(self, model_name: str, embedding_dim: int = 768):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get actual hidden size from model
        hidden_size = self.encoder.config.hidden_size
        
        # Projection head to desired embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def forward(self, input_ids, attention_mask):
        """Forward pass to get embeddings."""
        # Get model outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use mean pooling over sequence length
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        
        # Apply projection
        embeddings = self.projection(embeddings)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def contrastive_loss(self, query_embeddings, scenario_embeddings, labels):
        """Contrastive loss for positive and negative pairs."""
        # Compute cosine similarity
        similarities = torch.cosine_similarity(query_embeddings, scenario_embeddings, dim=1)
        similarities = similarities / self.temperature
        
        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(similarities, labels)
        
        return loss, similarities


class ScenarioMatchingTrainer:
    """Trainer for scenario matching models."""
    
    def __init__(self, model_name: str, train_file: str, val_file: str, 
                 output_dir: str, embedding_dim: int = 768):
        self.model_name = model_name
        self.train_file = train_file
        self.val_file = val_file
        self.output_dir = output_dir
        self.embedding_dim = embedding_dim
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = ContrastiveEmbeddingModel(model_name, embedding_dim)
        
        # Create datasets
        self.train_dataset = ScenarioMatchingDataset(train_file, self.tokenizer)
        self.val_dataset = ScenarioMatchingDataset(val_file, self.tokenizer)
        
        print(f"Initialized trainer for {model_name}")
        print(f"Train dataset: {len(self.train_dataset)} pairs")
        print(f"Validation dataset: {len(self.val_dataset)} pairs")
    
    def train(self, num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(num_epochs):
            print(f"\n🚀 Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                query_embeddings = self.model(batch['query_input_ids'], batch['query_attention_mask'])
                scenario_embeddings = self.model(batch['scenario_input_ids'], batch['scenario_attention_mask'])
                
                # Compute loss
                loss, similarities = self.model.contrastive_loss(
                    query_embeddings, scenario_embeddings, batch['label']
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Statistics
                train_loss += loss.item()
                predictions = (torch.sigmoid(similarities) > 0.5).float()
                train_correct += (predictions == batch['label']).sum().item()
                train_total += len(batch['label'])
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader, device)
            
            # Statistics
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"{self.output_dir}/best_model")
                print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")
            
            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
        
        # Save final model and history
        self.save_model(f"{self.output_dir}/final_model")
        with open(f"{self.output_dir}/training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.4f}")
        return training_history
    
    def evaluate(self, data_loader, device):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # Forward pass
                query_embeddings = self.model(batch['query_input_ids'], batch['query_attention_mask'])
                scenario_embeddings = self.model(batch['scenario_input_ids'], batch['scenario_attention_mask'])
                
                # Compute loss
                loss, similarities = self.model.contrastive_loss(
                    query_embeddings, scenario_embeddings, batch['label']
                )
                
                total_loss += loss.item()
                
                # Predictions
                predictions = (torch.sigmoid(similarities) > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def save_model(self, save_path):
        """Save the trained model."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), f"{save_path}/model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'model_class': 'ContrastiveEmbeddingModel'
        }
        with open(f"{save_path}/config.json", 'w') as f:
            json.dump(config, f, indent=2)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Fine-tune embedding models for scenario matching')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct or Qwen/Qwen3-Embedding-8B)')
    parser.add_argument('--train_file', type=str, required=True, help='Training dataset file')
    parser.add_argument('--val_file', type=str, required=True, help='Validation dataset file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for trained model')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    print("🚀 Starting embedding model fine-tuning...")
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Output dir: {args.output_dir}")
    
    # Initialize trainer
    trainer = ScenarioMatchingTrainer(
        model_name=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("✅ Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
