import os

# MODEL PARAMS
MODEL_NAME="openai/gpt-oss-20b"  # Upgraded from llama-3.1-8b-instruct for better medical reasoning
MODEL_TEMPERATURE = 0.2  # Lowered from 0.6 for more consistent multi-stage reasoning
ABSTRACT_MAX_TOKENS = 384  # Increased from 256 to support 150-word abstracts with medical details
FACTOR_MAX_TOKENS = 768  # Increased from 512 to prevent truncation of medical factors
EMOTION_MAX_TOKENS = 256
LIKERT_MAX_TOKENS = 384  # Increased from 256 to support medical context guidance
OUTPUT_MAX_TOKENS = 1536

# ENV VARS
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CONNECT_TIMEOUT = int(os.getenv("OPENROUTER_CONNECT_TIMEOUT", "20"))
OPENROUTER_READ_TIMEOUT = int(os.getenv("OPENROUTER_READ_TIMEOUT", "180"))

# --- Hardcoded File Paths ---
# Resolve data directory relative to this package, not the process CWD
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOBIRD_ROOT = os.path.abspath(os.path.join(MODULE_DIR, ".."))
PROMPT_DIR = os.path.join(EMOBIRD_ROOT, "prompts")

# LIKERT SCALE
LIKERT_SCALE = {
    'very-unlikely': 0.05,
    'unlikely': 0.25,
    'neutral': 0.50,
    'likely': 0.75,
    'very-likely': 0.95
}