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
VLLM_MAX_MODEL_LEN=8192
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

        # Hardening controls
        # Global stop sequences to terminate trailing chatter or markdown
        # Do NOT include 'END_OF_FACTORS' here; it's a parse-only sentinel to avoid empty outputs.
        # Avoid using "\nEND" as a global stop to prevent collisions with the 'END_OF_FACTORS' line.
        self.stop_seqs = ["\n```", "```", "\n--", "\nNote:", "###"]
        # Dedicated end sentinels for multi-step pipeline
        self.end_draft_sentinel = "<<END_DRAFT>>"
        self.end_json_sentinel = "<<END_JSON>>"
        # Strict JSON controls
        self.strict_json_only = True
        self.allow_format_only_retry = 1  # number of LLM reformat-only retries
        # Dedicated JSON generation controls
        self.json_max_tokens = 256
        self.json_temperature = 0.0
        # Draft essay generation budgets
        self.draft_max_tokens = 1200  # target ~1100-1300
        self.draft_temperature = 0.6
        # JSON rating step budget (tighter than general JSON if desired)
        self.json_rating_max_tokens = 200
        # Unified analysis (factors + values + emotions) controls
        self.temp_analysis = 0.3  # slightly non-zero for creative factor/emotion discovery
        self.max_tokens_analysis = 768
        
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
        # Optional custom download/cache directory for vLLM/HF model weights
        self.vllm_download_dir = None
    
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
            'EMOBIRD_TEMP_ANALYSIS': 'temp_analysis',
            'EMOBIRD_MAX_TOKENS_ANALYSIS': 'max_tokens_analysis',
            'EMOBIRD_JSON_MAX_TOKENS': 'json_max_tokens',
            'EMOBIRD_JSON_TEMPERATURE': 'json_temperature',
            'EMOBIRD_JSON_RATING_MAX_TOKENS': 'json_rating_max_tokens',
            'EMOBIRD_DRAFT_MAX_TOKENS': 'draft_max_tokens',
            'EMOBIRD_DRAFT_TEMPERATURE': 'draft_temperature',
            'EMOBIRD_MAX_CPT_ENTRIES': 'max_cpt_entries',
            'EMOBIRD_USE_BAYESIAN': 'use_bayesian_calibration',
            'EMOBIRD_LOG_LEVEL': 'log_level',
            'EMOBIRD_CACHE_DIR': 'cache_dir',
            # vLLM specific overrides
            'EMOBIRD_USE_VLLM': 'use_vllm',
            'EMOBIRD_VLLM_MAX_MODEL_LEN': 'vllm_max_model_len',
            'EMOBIRD_VLLM_TENSOR_PARALLEL_SIZE': 'vllm_tensor_parallel_size',
            'EMOBIRD_VLLM_GPU_MEMORY_UTILIZATION': 'vllm_gpu_memory_utilization',
            'EMOBIRD_VLLM_DOWNLOAD_DIR': 'vllm_download_dir',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if config_key in ['max_new_tokens', 'max_cpt_entries', 'max_tokens_analysis', 'json_max_tokens', 'json_rating_max_tokens', 'draft_max_tokens', 'vllm_max_model_len', 'vllm_tensor_parallel_size']:
                    setattr(self, config_key, int(env_value))
                elif config_key in ['temperature', 'bayesian_smoothing', 'temp_analysis', 'json_temperature', 'draft_temperature', 'vllm_gpu_memory_utilization']:
                    setattr(self, config_key, float(env_value))
                elif config_key in ['use_bayesian_calibration', 'do_sample', 'use_caching', 'strict_json_only', 'use_vllm']:
                    setattr(self, config_key, env_value.lower() in ['true', '1', 'yes'])
                else:
                    setattr(self, config_key, env_value)

        # STOP sequences via env (comma-separated)
        stop_env = os.getenv('EMOBIRD_STOP_SEQS')
        if stop_env:
            self.stop_seqs = [s for s in stop_env.split(',') if s]
        # Sentinel overrides via env
        draft_s = os.getenv('EMOBIRD_END_DRAFT_SENTINEL')
        if draft_s:
            self.end_draft_sentinel = draft_s
        json_s = os.getenv('EMOBIRD_END_JSON_SENTINEL')
        if json_s:
            self.end_json_sentinel = json_s
        # Allow format-only retries override
        fmt_retry_env = os.getenv('EMOBIRD_ALLOW_FORMAT_ONLY_RETRY')
        if fmt_retry_env is not None:
            try:
                self.allow_format_only_retry = int(fmt_retry_env)
            except ValueError:
                pass
    
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
