import torch
import math
from typing import List

# Rating scale for probability mapping
RATING_SCALE = {
    "very-unlikely": 0.05,
    "unlikely": 0.25,
    "neutral": 0.50,
    "likely": 0.75,
    "very-likely": 0.95
}

def norm_key(name: str, value: str) -> str:
    """
    Normalize factor name and value into a consistent key format.
    
    Args:
        name: Factor name (e.g., "Attachment Style")
        value: Factor value (e.g., "Strong")
        
    Returns:
        Normalized key string (e.g., "attachment_style=strong")
    """
    normalized_name = name.strip().lower().replace(' ', '_')
    normalized_value = value.strip().lower().replace(' ', '_')
    return f"{normalized_name}={normalized_value}"

def validate_rating(rating: str) -> str:
    """
    Validate and normalize a rating string.
    
    Args:
        rating: Raw rating string from LLM
        
    Returns:
        Validated rating string
        
    Raises:
        ValueError: If rating is not in allowed scale
    """
    normalized_rating = rating.strip().lower()
    
    if normalized_rating not in RATING_SCALE:
        raise ValueError(f"Illegal rating: {rating}. Must be one of: {list(RATING_SCALE.keys())}")
    
    return normalized_rating

def pool_logistic(probabilities: List[float]) -> float:
    """
    Pool probabilities using the logistic (BIRD) formula.
    
    Args:
        probabilities: List of probabilities to pool
        
    Returns:
        Pooled probability using BIRD formula
    """
    if not probabilities:
        return 0.5  # Neutral if no probabilities
    
    if len(probabilities) == 1:
        return probabilities[0]
    
    # BIRD formula: prodP / (prodP + prodN)
    # where prodP = product of probabilities, prodN = product of (1-p)
    prod_p = math.prod(probabilities)
    prod_n = math.prod(1 - p for p in probabilities)
    
    # Avoid division by zero
    denominator = prod_p + prod_n
    if denominator == 0:
        return 0.5
    
    return prod_p / denominator

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