import json
import torch
import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM # Now handled by ModelInitializer
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
from pathlib import Path # Import Path

# --- CONFIGURABLE PARAMETERS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.6
# -----------------------------

# Adjust path to import from lib using pathlib
script_path = Path(__file__).resolve() # Get absolute path of the current script
project_root = script_path.parent.parent # Get 'scripts' directory's parent (project root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root)) # Add project root to the start of sys.path

from scripts_utils import ModelInitializer, print_gpu_info, get_model_name # Using centralized utilities
# Define paths relative to the project root
ITEMS_PATH = project_root / "data" / "seceu" / "seceu_items.json"
STANDARD_PATH = project_root / "data" / "seceu" / "seceu_standard.json"
PREDICTIONS_OUT = project_root / "results" / "seceu" / "seceu_qwen_predictions.json"
RESULTS_OUT = project_root / "results" / "seceu" / "seceu_qwen_results.json"

# ---------------- INITIALIZE MODEL CLIENT ----------------
print("🔄 Initializing ModelInitializer...")
client = ModelInitializer(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    device=DEVICE,
    default_max_new_tokens=MAX_NEW_TOKENS,
    default_temperature=TEMPERATURE,
    default_do_sample=False # Consistent with original model.generate behavior when temperature is set
)
# ModelInitializer now handles model loading and setup.

# Print GPU information
print_gpu_info()

# ---------------- LOAD DATA ----------------
with open(ITEMS_PATH) as f:
    items = json.load(f)

with open(STANDARD_PATH) as f:
    standard_data = json.load(f)

standard_scores = np.array(standard_data["standard_scores"])
human_pattern = np.array(standard_data["human_pattern"])
population_mean = standard_data["population_mean"]
population_std = standard_data["population_std"]

# ---------------- HELPER FUNCTIONS ----------------
# The local query_model function has been removed.
# client.gen_response(prompt)['completion'] will be used instead.

def extract_scores(text, fallback=[2.5, 2.5, 2.5, 2.5]):
    try:
        numbers = [float(x) for x in text.replace(",", " ").split() if x.replace('.', '', 1).isdigit()]
        numbers = np.array(numbers[:4])
        if len(numbers) != 4: return np.array(fallback)
        numbers = np.maximum(numbers, 0)
        return numbers / numbers.sum() * 10
    except:
        return np.array(fallback)

# ---------------- RUN MODEL ----------------
model_preds = []

for i, item in enumerate(tqdm(items, desc="Processing items")):
    story = item["story"]
    options = item["options"]
    options_str = ", ".join(f"({i+1}) {opt}" for i, opt in enumerate(options))
    prompt = f'''You are an empathetic AI assistant. Your task is to carefully read the following story and evaluate the emotional state of the main character.
You will be given four emotion options. For each option, assign a score from 0 to 10 representing how intensely the main character feels that emotion.

**Important Constraints:**
1. Each score must be between 0 and 10 (inclusive).
2. The sum of your scores for the four options MUST be exactly 10.

Story:
{story}

Emotion Options:
{options_str}

Please follow these steps in your reasoning before providing the scores:
1.  Deeply analyze the provided story, focusing on the main character\'s situation, actions, and any explicit or implicit emotional cues.
2.  For each of the four emotion options, critically assess its relevance and intensity concerning the character\'s experience.
3.  Assign an initial numerical score (0-10) to each emotion based on your analysis.
4.  Verify that the sum of your four scores is exactly 10. If not, carefully adjust the scores, maintaining their relative proportions as much as possible, until they sum precisely to 10.
5.  Provide ONLY the four final numerical scores, separated by spaces (e.g., 1.5 3.0 4.5 1.0). Do not add any other text or explanation before or after the scores.

Final Scores:'''
    
    print(f"🧠 Item {i+1}: Prompting model...")
    response_dict = client.gen_response(prompt)
    output = response_dict['completion']
    scores = extract_scores(output)
    model_preds.append(scores.tolist())

# Save model predictions
with open(PREDICTIONS_OUT, "w") as f:
    json.dump({"predictions": model_preds}, f, indent=2)

# ---------------- CALCULATE SCORES ----------------
model_preds_np = np.array(model_preds)
distances = np.linalg.norm(model_preds_np - standard_scores, axis=1)
seceu_score = distances.mean()
eq_score = 15 * ((population_mean - seceu_score) / population_std) + 100
pattern_similarity, _ = pearsonr(distances, human_pattern)

# ---------------- SAVE RESULTS ----------------
results = {
    "seceu_score": round(float(seceu_score), 3),
    "eq_score": round(float(eq_score), 2),
    "pattern_similarity": round(float(pattern_similarity), 3)
}

with open(RESULTS_OUT, "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ DONE")
print(f"SECEU Score: {results['seceu_score']}")
print(f"EQ Score: {results['eq_score']}")
print(f"Pattern Similarity: {results['pattern_similarity']}")
