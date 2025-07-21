"""
Run LoRA fine-tuning on the generated training dataset.
"""

import os
import subprocess
import glob
from datetime import datetime


def find_latest_dataset_files():
    """Find the latest generated dataset files."""
    training_data_dir = "training_data"
    
    if not os.path.exists(training_data_dir):
        raise FileNotFoundError(f"Training data directory {training_data_dir} not found")
    
    # Find all dataset files
    train_files = glob.glob(f"{training_data_dir}/train_dataset_*.json")
    val_files = glob.glob(f"{training_data_dir}/val_dataset_*.json")
    
    if not train_files or not val_files:
        raise FileNotFoundError("No training dataset files found")
    
    # Get the latest files (by timestamp in filename)
    latest_train = max(train_files, key=lambda x: x.split('_')[-1])
    latest_val = max(val_files, key=lambda x: x.split('_')[-1])
    
    return latest_train, latest_val


def run_lora_training():
    """Run LoRA training with the latest dataset."""
    print("🎯 LLAMA LORA FINE-TUNING")
    print("=" * 50)
    
    # Find dataset files
    try:
        train_file, val_file = find_latest_dataset_files()
        print(f"📁 Using training files:")
        print(f"   Train: {train_file}")
        print(f"   Validation: {val_file}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run create_practical_training_dataset.py first")
        return False
    
    # Set up output directory
    timestamp = int(datetime.now().timestamp())
    output_dir = f"models/llama_lora_{timestamp}"
    
    # Training command
    command = f"""python finetune_llama_lora_vllm.py \\
        --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --train_file {train_file} \\
        --val_file {val_file} \\
        --output_dir {output_dir} \\
        --embedding_dim 768 \\
        --lora_r 16 \\
        --lora_alpha 32 \\
        --num_epochs 3 \\
        --batch_size 4 \\
        --learning_rate 1e-4"""
    
    print(f"\n🚀 Starting LoRA training...")
    print(f"📋 Output directory: {output_dir}")
    print(f"📋 Command: {command}")
    
    try:
        # Run the training
        result = subprocess.run(command, shell=True, check=True, text=True)
        print(f"\n✅ LoRA training completed successfully!")
        print(f"📁 Model saved in: {output_dir}")
        return True, output_dir
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return False, None


def main():
    """Main function."""
    success, model_dir = run_lora_training()
    
    if success:
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 LoRA model saved in: {model_dir}")
        print(f"\n🚀 Next steps:")
        print(f"1. Test the fine-tuned model with benchmark scripts")
        print(f"2. Compare performance with the base model")
        print(f"3. Use the fine-tuned model in your production system")
    else:
        print(f"\n❌ Training failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
