import torch

def print_gpu_info():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Running on GPU: {device_name}")
        # Print CUDA version
        print(f"CUDA Version: {torch.version.cuda}")
        # Print available memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        free_memory_gb = free_memory / (1024 ** 3)
        total_memory_gb = total_memory / (1024 ** 3)
        print(f"GPU Memory: {free_memory_gb:.2f}GB free / {total_memory_gb:.2f}GB total")
    else:
        print("❌ No GPU available, running on CPU")