import torch

def print_gpu_info():
    """Checks for CUDA availability and prints GPU information if available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Running on GPU: {gpu_name}")
    else:
        print("ℹ️ CUDA not available. Running on CPU.")

if __name__ == '__main__':
    # For testing the utility function directly
    print_gpu_info() 