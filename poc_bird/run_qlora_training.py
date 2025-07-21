#!/usr/bin/env python3
"""
Launch script for QLoRA training of Llama 3.1 embedding model.
Automatically uses the generated enhanced training data.
"""

import os
import subprocess
import sys
import argparse
from datetime import datetime

def check_gpu_availability():
    """Check GPU availability and memory."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU detected:")
            print(result.stdout)
            return True
        else:
            print("❌ No GPU detected or nvidia-smi not available")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found. GPU may not be available.")
        return False

def find_latest_dataset():
    """Find the latest enhanced training dataset."""
    enhanced_dir = "enhanced_training_data"
    if not os.path.exists(enhanced_dir):
        print(f"❌ Enhanced training data directory '{enhanced_dir}' not found!")
        print("Please run create_enhanced_embedding_dataset.py first.")
        return None, None, None
    
    # Find files with timestamps
    train_files = [f for f in os.listdir(enhanced_dir) if f.startswith('train_enhanced_')]
    val_files = [f for f in os.listdir(enhanced_dir) if f.startswith('validation_enhanced_')]
    test_files = [f for f in os.listdir(enhanced_dir) if f.startswith('test_enhanced_')]
    
    if not (train_files and val_files and test_files):
        print("❌ Complete dataset not found!")
        print(f"Found: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")
        return None, None, None
    
    # Get the latest (assuming same timestamp)
    latest_train = max(train_files)
    latest_val = max(val_files)
    latest_test = max(test_files)
    
    train_path = os.path.join(enhanced_dir, latest_train)
    val_path = os.path.join(enhanced_dir, latest_val)
    test_path = os.path.join(enhanced_dir, latest_test)
    
    print(f"✅ Found latest dataset:")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")
    print(f"   Test: {test_path}")
    
    return train_path, val_path, test_path

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Launch QLoRA training for Llama 3.1 embedding")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank (default: 64)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="Show command without running")
    
    args = parser.parse_args()
    
    print("🚀 Llama 3.1 Embedding QLoRA Training Launcher")
    print("=" * 60)
    
    # Check GPU
    gpu_available = check_gpu_availability()
    if not gpu_available:
        print("⚠️  Warning: No GPU detected. Training will be very slow on CPU.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Find dataset
    print("\n📊 Locating training data...")
    train_path, val_path, test_path = find_latest_dataset()
    if not train_path:
        return
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"llama_embedding_qlora_{timestamp}"
    
    print(f"\n🎯 Training Configuration:")
    print(f"   GPU: {args.gpu}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   LoRA Rank: {args.lora_r}")
    print(f"   Output Dir: {output_dir}")
    print(f"   Wandb: {'Enabled' if args.wandb else 'Disabled'}")
    
    # Build training command
    cmd = [
        "python", "train_llama_embedding_qlora.py",
        "--train_file", train_path,
        "--val_file", val_path,
        "--output_dir", output_dir,
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.epochs),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_r * 2),  # Typical: alpha = 2 * rank
        "--model_name", "meta-llama/Llama-3.1-8B"
    ]
    
    if args.wandb:
        cmd.append("--wandb")
    
    # Set CUDA device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print(f"\n🔧 Training Command:")
    print(f"CUDA_VISIBLE_DEVICES={args.gpu} {' '.join(cmd)}")
    
    if args.dry_run:
        print("\n🏃 Dry run completed. Use --run to actually start training.")
        return
    
    print(f"\n⏳ Starting training...")
    print("=" * 60)
    
    try:
        # Run training
        result = subprocess.run(cmd, env=env, check=True)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        
        # Show next steps
        print(f"\n🚀 Next Steps:")
        print(f"1. Evaluate the trained model:")
        print(f"   python evaluate_embedding_model.py --model_path {output_dir}/best_model")
        print(f"2. Update your config.py to use the fine-tuned model")
        print(f"3. Test with your router system")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        print("Check the logs above for errors.")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
