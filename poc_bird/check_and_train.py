"""
Check dependencies and run LoRA training.
"""

import os
import sys
import subprocess
import importlib


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'transformers', 
        'peft',
        'datasets',
        'accelerate',
        'bitsandbytes',
        'scipy',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    print("🔍 Checking dependencies...")
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def check_dataset():
    """Check if training dataset exists."""
    training_data_dir = "training_data"
    
    if not os.path.exists(training_data_dir):
        print(f"❌ Training data directory {training_data_dir} not found")
        return False
    
    train_files = [f for f in os.listdir(training_data_dir) if f.startswith('train_dataset_')]
    val_files = [f for f in os.listdir(training_data_dir) if f.startswith('val_dataset_')]
    
    if not train_files or not val_files:
        print("❌ No training dataset files found")
        print("Please run: python create_practical_training_dataset.py")
        return False
    
    latest_train = max(train_files, key=lambda x: x.split('_')[-1])
    latest_val = max(val_files, key=lambda x: x.split('_')[-1])
    
    print(f"✅ Found training dataset:")
    print(f"   Train: {training_data_dir}/{latest_train}")
    print(f"   Val: {training_data_dir}/{latest_val}")
    
    return True


def run_training():
    """Run the LoRA training."""
    print("\n🚀 Starting LoRA training...")
    
    # Use CUDA_VISIBLE_DEVICES=0 to use only the first GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        result = subprocess.run(
            ["python", "run_lora_training.py"],
            env=env,
            check=True,
            text=True,
            capture_output=True
        )
        
        print("✅ Training completed successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Training failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Main function."""
    print("🎯 LLAMA LORA FINE-TUNING SETUP")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Check dataset
    if not check_dataset():
        print("\n❌ Please generate training dataset first")
        return
    
    # Run training
    print("\n" + "=" * 50)
    print("🚀 STARTING TRAINING")
    print("=" * 50)
    
    success = run_training()
    
    if success:
        print("\n🎉 Training pipeline completed successfully!")
    else:
        print("\n❌ Training pipeline failed")


if __name__ == "__main__":
    main()
