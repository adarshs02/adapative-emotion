"""
Configuration module for Emobird system.
"""

import os
import json
import torch
from typing import Dict, Any, Optional

# EmoBIRD Configurations
LLM_MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
LLM_MAX_NEW_TOKENS=128
LLM_TEMPERATURE=0.6

# vLLM specific configurations
USE_VLLM=True
VLLM_GPU_MEMORY_UTILIZATION=0.8
VLLM_MAX_MODEL_LEN=4096
VLLM_TENSOR_PARALLEL_SIZE=1


class EmobirdConfig:
    """Configuration class for Emobird system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration with defaults and optional config file."""
        
        # Default configuration
        self._set_defaults()
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            self._load_config_file(config_path)
        
        # Override with environment variables if present
        self._load_env_variables()
    
    def _set_defaults(self):
        """Set default configuration values."""
        
        # Model configuration
        self.llm_model_name = LLM_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Generation parameters
        self.max_new_tokens = LLM_MAX_NEW_TOKENS
        self.temperature = LLM_TEMPERATURE
        self.do_sample = True
        
        # CPT generation limits
        self.max_cpt_entries = 50  # Limit CPT size to avoid exponential explosion
        self.max_factors = 5
        self.max_factor_values = 4
        
        # Bayesian calibration
        self.use_bayesian_calibration = False
        self.bayesian_smoothing = 0.1
        
        # Default emotions
        self.default_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        
        # Cache settings
        self.use_caching = False
        self.cache_dir = "emobird_cache"
        
        # Logging
        self.log_level = "INFO"
        self.log_file = None
        
        # Performance
        self.batch_size = 1
        self.max_sequence_length = 2048
        
        # vLLM specific settings
        self.use_vllm = USE_VLLM
        self.vllm_gpu_memory_utilization = VLLM_GPU_MEMORY_UTILIZATION
        self.vllm_max_model_len = VLLM_MAX_MODEL_LEN
        self.vllm_tensor_parallel_size = VLLM_TENSOR_PARALLEL_SIZE
    
    def _load_config_file(self, config_path: str):
        """Load configuration from JSON file."""
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration with file values
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Failed to load config file {config_path}: {e}")
    
    def _load_env_variables(self):
        """Load configuration from environment variables."""
        
        env_mappings = {
            'EMOBIRD_MODEL': 'llm_model_name',
            'EMOBIRD_DEVICE': 'device',
            'EMOBIRD_MAX_TOKENS': 'max_new_tokens',
            'EMOBIRD_TEMPERATURE': 'temperature',
            'EMOBIRD_MAX_CPT_ENTRIES': 'max_cpt_entries',
            'EMOBIRD_USE_BAYESIAN': 'use_bayesian_calibration',
            'EMOBIRD_LOG_LEVEL': 'log_level',
            'EMOBIRD_CACHE_DIR': 'cache_dir'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if config_key in ['max_new_tokens', 'max_cpt_entries']:
                    setattr(self, config_key, int(env_value))
                elif config_key in ['temperature', 'bayesian_smoothing']:
                    setattr(self, config_key, float(env_value))
                elif config_key in ['use_bayesian_calibration', 'do_sample', 'use_caching']:
                    setattr(self, config_key, env_value.lower() in ['true', '1', 'yes'])
                else:
                    setattr(self, config_key, env_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        
        config_dict = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                config_dict[key] = getattr(self, key)
        
        return config_dict
    
    def save_config(self, config_path: str):
        """Save current configuration to file."""
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            print(f"✅ Configuration saved to {config_path}")
        except IOError as e:
            print(f"⚠️ Failed to save config to {config_path}: {e}")
    
    def update_config(self, **kwargs):
        """Update configuration with new values."""
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"⚠️ Unknown configuration key: {key}")
    
    def get_device(self):
        """Get the appropriate device for model loading."""
        
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    def print_config(self):
        """Print current configuration."""
        
        print("\n🐦 Current Emobird Configuration:")
        print("=" * 40)
        
        config_dict = self.to_dict()
        for key, value in sorted(config_dict.items()):
            print(f"{key:25}: {value}")
        
        print("=" * 40)


# Global configuration instance
_global_config = None

def get_config(config_path: Optional[str] = None) -> EmobirdConfig:
    """Get the global configuration instance."""
    
    global _global_config
    if _global_config is None:
        _global_config = EmobirdConfig(config_path)
    return _global_config

def set_config(config: EmobirdConfig):
    """Set the global configuration instance."""
    
    global _global_config
    _global_config = config
