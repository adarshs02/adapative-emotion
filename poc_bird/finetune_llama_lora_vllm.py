"""
Fine-tune Llama model using LoRA with vLLM embedding mode for scenario matching.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import List, Dict, Any, Tuple
import os
from datetime import datetime
from sklearn.metrics import accuracy_score
import argparse
from vllm import LLM, SamplingParams
import time


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


class LoRAEmbeddingModel(nn.Module):
    """LoRA-enhanced embedding model for contrastive learning."""
    
    def __init__(self, model_name: str, embedding_dim: int = 768, lora_r: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        # Apply LoRA to the model
        self.encoder = get_peft_model(self.encoder, lora_config)
        
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
        
        print(f"✅ LoRA model initialized with r={lora_r}, alpha={lora_alpha}")
        print(f"📊 Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
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


class VLLMEmbeddingGenerator:
    """Generate embeddings using vLLM in embedding mode."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        
        print(f"🚀 Initializing vLLM embedding generator...")
        
        # Initialize vLLM for embeddings
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            max_model_len=512,
            enforce_eager=True  # For embedding mode
        )
        
        print(f"✅ vLLM embedding generator initialized")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts using vLLM."""
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        
        # Use vLLM to generate embeddings
        # Note: This is a placeholder - vLLM embedding API might be different
        # You may need to adjust based on the actual vLLM embedding interface
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # For now, we'll use a simple approach
            # In practice, you'd use vLLM's embedding generation capabilities
            # This is a placeholder that would need to be replaced with actual vLLM embedding calls
            
            # Placeholder: Generate random embeddings (replace with actual vLLM embedding calls)
            batch_embeddings = np.random.randn(len(batch_texts), 768).astype(np.float32)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


class LoRAScenarioMatchingTrainer:
    """Trainer for LoRA-enhanced scenario matching models."""
    
    def __init__(self, model_name: str, train_file: str, val_file: str, 
                 output_dir: str, embedding_dim: int = 768, lora_r: int = 16, lora_alpha: int = 32):
        self.model_name = model_name
        self.train_file = train_file
        self.val_file = val_file
        self.output_dir = output_dir
        self.embedding_dim = embedding_dim
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = LoRAEmbeddingModel(model_name, embedding_dim, lora_r, lora_alpha)
        
        # Create datasets
        self.train_dataset = ScenarioMatchingDataset(train_file, self.tokenizer)
        self.val_dataset = ScenarioMatchingDataset(val_file, self.tokenizer)
        
        print(f"🎯 Initialized LoRA trainer for {model_name}")
        print(f"📊 Train dataset: {len(self.train_dataset)} pairs")
        print(f"📊 Validation dataset: {len(self.val_dataset)} pairs")
    
    def train(self, num_epochs: int = 3, batch_size: int = 8, learning_rate: float = 1e-4):
        """Train the LoRA model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler (higher learning rate for LoRA)
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
            
            print(f"  📊 Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"  📊 Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"{self.output_dir}/best_model")
                print(f"  💾 Saved best LoRA model (val_loss: {val_loss:.4f})")
            
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
        
        print(f"\n✅ LoRA training completed! Best validation loss: {best_val_loss:.4f}")
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
        """Save the trained LoRA model."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save LoRA adapter
        self.model.encoder.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save projection head and other components
        torch.save({
            'projection': self.model.projection.state_dict(),
            'temperature': self.model.temperature.item(),
            'config': {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'lora_r': self.lora_r,
                'lora_alpha': self.lora_alpha
            }
        }, f"{save_path}/model_components.pt")
        
        print(f"💾 LoRA model saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Fine-tune Llama with LoRA for scenario matching')
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                       help='Model name')
    parser.add_argument('--train_file', type=str, required=True, help='Training dataset file')
    parser.add_argument('--val_file', type=str, required=True, help='Validation dataset file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for trained model')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    print("🚀 Starting LoRA fine-tuning for embedding model...")
    print(f"📋 Model: {args.model_name}")
    print(f"📋 Train file: {args.train_file}")
    print(f"📋 Val file: {args.val_file}")
    print(f"📋 Output dir: {args.output_dir}")
    print(f"📋 LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
    
    # Initialize trainer
    trainer = LoRAScenarioMatchingTrainer(
        model_name=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("✅ LoRA fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
