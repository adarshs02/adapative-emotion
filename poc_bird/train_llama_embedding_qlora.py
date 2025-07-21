"""
QLoRA Fine-tuning Script for Llama 3.1 Embedding Model
Optimized for scenario matching and embedding quality with memory-efficient training.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, 
    TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training
)
import os
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration for QLoRA fine-tuning."""
    model_name: str = "meta-llama/Llama-3.1-8B"  # Base Llama model
    max_length: int = 512
    batch_size: int = 32  # Reduced for memory efficiency with QLoRA
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4  # Higher LR typical for LoRA
    num_epochs: int = 3
    warmup_steps: int = 100
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "llama_embedding_qlora"
    
    # QLoRA specific
    lora_r: int = 64  # LoRA rank
    lora_alpha: int = 128  # LoRA scaling parameter
    lora_dropout: float = 0.1
    use_4bit: bool = True  # 4-bit quantization
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # Training specific
    temperature: float = 0.05  # For contrastive loss
    margin: float = 0.5  # Margin for triplet loss
    use_wandb: bool = True
    wandb_project: str = "llama-embedding-qlora"


class ScenarioEmbeddingDataset(Dataset):
    """Dataset for scenario embedding training with contrastive learning."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load enhanced training data
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        
        self.data = dataset['data']
        self.metadata = dataset['metadata']
        
        logger.info(f"Loaded {len(self.data)} training pairs from {data_path}")
        logger.info(f"Dataset metadata: {self.metadata}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query and scenario description
        query_encoding = self.tokenizer(
            item['query'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        scenario_encoding = self.tokenizer(
            item['scenario_description'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'scenario_input_ids': scenario_encoding['input_ids'].squeeze(),
            'scenario_attention_mask': scenario_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(item['label'], dtype=torch.float),
            'pair_type': item['pair_type'],
            'scenario_id': item['scenario_id']
        }


class LlamaEmbeddingModel(nn.Module):
    """Llama model adapted for embedding generation with contrastive learning."""
    
    def __init__(self, model_name: str, config: TrainingConfig):
        super().__init__()
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.use_4bit,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
        
        # Load base model with quantization
        self.llama_model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Prepare model for k-bit training
        self.llama_model = prepare_model_for_kbit_training(self.llama_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Apply LoRA
        self.llama_model = get_peft_model(self.llama_model, lora_config)
        
        # Embedding projection layer
        self.embedding_dim = self.llama_model.config.hidden_size
        self.projection = nn.Linear(self.embedding_dim, 768)  # Project to standard embedding size
        
        # Normalize embeddings for similarity computation
        self.normalize = True
        
        logger.info(f"Initialized LlamaEmbeddingModel with LoRA")
        logger.info(f"Trainable parameters: {self.llama_model.print_trainable_parameters()}")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass to generate embeddings."""
        # Get hidden states from Llama
        outputs = self.llama_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        
        # Project to target dimension
        embeddings = self.projection(embeddings)
        
        # Normalize if specified
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class ContrastiveLoss(nn.Module):
    """Contrastive loss for embedding training."""
    
    def __init__(self, temperature: float = 0.05, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, query_embeddings, scenario_embeddings, labels):
        """
        Compute contrastive loss.
        Args:
            query_embeddings: [batch_size, embedding_dim]
            scenario_embeddings: [batch_size, embedding_dim] 
            labels: [batch_size] - 1 for positive pairs, 0 for negative pairs
        """
        # Compute similarity scores
        similarities = F.cosine_similarity(query_embeddings, scenario_embeddings, dim=1)
        
        # Scale by temperature
        similarities = similarities / self.temperature
        
        # Separate positive and negative pairs
        positive_mask = (labels == 1).float()
        negative_mask = (labels == 0).float()
        
        # Contrastive loss
        positive_loss = positive_mask * torch.clamp(self.margin - similarities, min=0) ** 2
        negative_loss = negative_mask * torch.clamp(similarities - (-self.margin), min=0) ** 2
        
        loss = (positive_loss + negative_loss).mean()
        
        return loss, similarities


class EmbeddingTrainer:
    """Custom trainer for embedding model with QLoRA."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=f"llama-embedding-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(config)
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = LlamaEmbeddingModel(config.model_name, config)
        
        # Initialize loss function
        self.criterion = ContrastiveLoss(config.temperature, config.margin)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        logger.info("EmbeddingTrainer initialized successfully")
    
    def train(self, train_data_path: str, val_data_path: str):
        """Main training loop."""
        
        # Create datasets
        train_dataset = ScenarioEmbeddingDataset(train_data_path, self.tokenizer, self.config.max_length)
        val_dataset = ScenarioEmbeddingDataset(val_data_path, self.tokenizer, self.config.max_length)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Training setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        total_steps = len(train_dataloader) * self.config.num_epochs
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Device: {device}")
        
        # Training loop
        global_step = 0
        best_val_accuracy = 0.0
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for step, batch in enumerate(train_dataloader):
                # Move to device
                query_input_ids = batch['query_input_ids'].to(device)
                query_attention_mask = batch['query_attention_mask'].to(device)
                scenario_input_ids = batch['scenario_input_ids'].to(device)
                scenario_attention_mask = batch['scenario_attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                query_embeddings = self.model(query_input_ids, query_attention_mask)
                scenario_embeddings = self.model(scenario_input_ids, scenario_attention_mask)
                
                # Compute loss
                loss, similarities = self.criterion(query_embeddings, scenario_embeddings, labels)
                
                # Backward pass
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += loss.item()
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, Step {global_step}, Loss: {avg_loss:.4f}")
                    
                    if self.config.use_wandb:
                        wandb.log({
                            "train_loss": avg_loss,
                            "epoch": epoch,
                            "global_step": global_step
                        })
                
                # Validation
                if global_step % self.config.eval_steps == 0:
                    val_accuracy = self.evaluate(val_dataloader, device)
                    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
                    
                    if self.config.use_wandb:
                        wandb.log({
                            "val_accuracy": val_accuracy,
                            "global_step": global_step
                        })
                    
                    # Save best model
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        self.save_model(f"{self.config.output_dir}/best_model")
                        logger.info(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_model(f"{self.config.output_dir}/checkpoint-{global_step}")
        
        # Final save
        self.save_model(f"{self.config.output_dir}/final_model")
        logger.info("Training completed successfully!")
        
        return best_val_accuracy
    
    def evaluate(self, val_dataloader, device):
        """Evaluate model on validation set."""
        self.model.eval()
        
        all_similarities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move to device
                query_input_ids = batch['query_input_ids'].to(device)
                query_attention_mask = batch['query_attention_mask'].to(device)
                scenario_input_ids = batch['scenario_input_ids'].to(device)
                scenario_attention_mask = batch['scenario_attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                query_embeddings = self.model(query_input_ids, query_attention_mask)
                scenario_embeddings = self.model(scenario_input_ids, scenario_attention_mask)
                
                # Compute similarities
                similarities = F.cosine_similarity(query_embeddings, scenario_embeddings, dim=1)
                
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to binary predictions using threshold
        predictions = (np.array(all_similarities) > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, predictions)
        
        self.model.train()
        return accuracy
    
    def save_model(self, path: str):
        """Save model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA adapter
        self.model.llama_model.save_pretrained(path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save config
        with open(os.path.join(path, 'training_config.json'), 'w') as f:
            json.dump(vars(self.config), f, indent=2)
        
        logger.info(f"Model saved to {path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Llama embedding model with QLoRA")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="llama_embedding_qlora", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for contrastive loss")
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", help="Base model name")
    
    args = parser.parse_args()
    
    # Create training config
    config = TrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        temperature=args.temperature,
        output_dir=args.output_dir,
        use_wandb=args.wandb
    )
    
    # Initialize trainer
    trainer = EmbeddingTrainer(config)
    
    # Start training
    best_accuracy = trainer.train(args.train_file, args.val_file)
    
    logger.info(f"Training completed! Best validation accuracy: {best_accuracy:.4f}")
    
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
