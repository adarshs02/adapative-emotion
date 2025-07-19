"""Configuration settings for the BIRD emotion prediction system."""

import os

# Embedding model configuration for Llama 3.1 with vLLM
EMBED_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
EMBED_BATCH_SIZE = 64  # Batch size for embedding generation (vLLM can handle larger batches)
EMBED_DIM = 4096  # Llama 3.1 8B embedding dimension

# HNSW index configuration
HNSW_INDEX_PATH = "scenario.idx"
HNSW_M = 16  # Number of connections per node
HNSW_EF_CONSTRUCTION = 200  # Size of dynamic candidate list
HNSW_EF_SEARCH = 100  # Size of dynamic candidate list during search

# Scenario matching configuration
SIMILARITY_THRESHOLD = 0.70
TOP_K_SCENARIOS = 5

# File paths
SCENARIOS_FILE = "atomic-scenarios.json"
SCENARIO_MAPPING_PATH = "scenario_mapping.json"
CPT_DIR = "cpts/cpts-atomic"

# LLM configuration (for factor extraction)
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.6

def get_device():
    """Get the appropriate device for computation."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def ensure_model_dir():
    """Ensure the models directory exists."""
    model_dir = os.path.dirname(EMBED_MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created models directory: {model_dir}")

def validate_model_path():
    """Validate that the embedding model file exists."""
    if not os.path.exists(EMBED_MODEL_PATH):
        raise FileNotFoundError(
            f"Embedding model not found at {EMBED_MODEL_PATH}. "
            f"Please download meta-llama-3-8b-instruct.Q4_K_M.gguf and place it in the models/ directory."
        )
