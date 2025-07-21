"""
Orchestrate the complete training and comparison pipeline for embedding models.
"""

import os
import subprocess
import json
from datetime import datetime


def run_command(command: str, description: str):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main training and comparison pipeline."""
    print("🎯 COMPLETE EMBEDDING MODEL TRAINING & COMPARISON PIPELINE")
    print("=" * 70)
    
    # Step 1: Create training dataset
    print("\n" + "="*50)
    print("STEP 1: CREATING TRAINING DATASET")
    print("="*50)
    
    if not run_command("python create_training_dataset.py", "Creating training dataset"):
        print("❌ Failed to create training dataset. Exiting.")
        return
    
    # Find the latest dataset files
    training_data_dir = "training_data"
    if not os.path.exists(training_data_dir):
        print(f"❌ Training data directory {training_data_dir} not found")
        return
    
    # Get the latest dataset files
    dataset_files = [f for f in os.listdir(training_data_dir) if f.startswith('train_dataset_')]
    if not dataset_files:
        print("❌ No training dataset files found")
        return
    
    latest_timestamp = max([f.split('_')[-1].replace('.json', '') for f in dataset_files])
    train_file = f"{training_data_dir}/train_dataset_{latest_timestamp}.json"
    val_file = f"{training_data_dir}/val_dataset_{latest_timestamp}.json"
    
    print(f"📁 Using training files:")
    print(f"   Train: {train_file}")
    print(f"   Validation: {val_file}")
    
    # Step 2: Fine-tune Llama model
    print("\n" + "="*50)
    print("STEP 2: FINE-TUNING LLAMA MODEL")
    print("="*50)
    
    llama_output_dir = "models/llama_finetuned"
    llama_command = f"""python finetune_embedding_models.py \\
        --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --train_file {train_file} \\
        --val_file {val_file} \\
        --output_dir {llama_output_dir} \\
        --embedding_dim 768 \\
        --num_epochs 3 \\
        --batch_size 4 \\
        --learning_rate 2e-5"""
    
    llama_success = run_command(llama_command, "Fine-tuning Llama model")
    
    # Step 3: Fine-tune Qwen model
    print("\n" + "="*50)
    print("STEP 3: FINE-TUNING QWEN MODEL")
    print("="*50)
    
    qwen_output_dir = "models/qwen_finetuned"
    qwen_command = f"""python finetune_embedding_models.py \\
        --model_name Qwen/Qwen3-Embedding-8B \\
        --train_file {train_file} \\
        --val_file {val_file} \\
        --output_dir {qwen_output_dir} \\
        --embedding_dim 768 \\
        --num_epochs 3 \\
        --batch_size 4 \\
        --learning_rate 2e-5"""
    
    qwen_success = run_command(qwen_command, "Fine-tuning Qwen model")
    
    # Step 4: Benchmark and compare models
    print("\n" + "="*50)
    print("STEP 4: BENCHMARKING FINE-TUNED MODELS")
    print("="*50)
    
    if llama_success or qwen_success:
        if not run_command("python benchmark_finetuned_models.py", "Benchmarking fine-tuned models"):
            print("⚠️  Benchmarking failed, but models were trained")
    else:
        print("❌ No models were successfully trained, skipping benchmark")
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("🏆 TRAINING PIPELINE SUMMARY")
    print("="*70)
    
    print(f"✅ Dataset creation: Success")
    print(f"{'✅' if llama_success else '❌'} Llama fine-tuning: {'Success' if llama_success else 'Failed'}")
    print(f"{'✅' if qwen_success else '❌'} Qwen fine-tuning: {'Success' if qwen_success else 'Failed'}")
    
    if llama_success or qwen_success:
        print("\n🎯 Next steps:")
        print("1. Check the benchmark results in logs/test/")
        print("2. Compare fine-tuned model performance vs original joint embedding system")
        print("3. Use the best performing model in your production system")
        
        if llama_success:
            print(f"   📁 Llama model saved in: {llama_output_dir}/")
        if qwen_success:
            print(f"   📁 Qwen model saved in: {qwen_output_dir}/")
    else:
        print("\n❌ Training failed. Check error messages above.")
    
    print("\n🎉 Pipeline completed!")


if __name__ == "__main__":
    main()
