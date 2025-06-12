import json
from pathlib import Path
from tqdm import tqdm
import torch

# Import your LlamaModelClient (assumes it's in the same project as emobench_llama.py)
from scripts_utils import ModelInitializer, get_model_name, print_gpu_info # Added print_gpu_info for consistency

# Determine project root directory based on script location
# Assumes this script (eqbench_eval.py) is in a subdirectory (e.g., 'scripts') of the project root.
script_file_abs_path = Path(__file__).resolve()
project_root_dir = script_file_abs_path.parent.parent 

DATA_PATH = project_root_dir / 'EQ-Bench' / 'data' / 'eq_bench_v2_questions_171.json'
RESULTS_OUT = project_root_dir / 'results'/ 'eqbench' / 'eqbench_mistral_results.json' # Save results in project root

with open(DATA_PATH, 'r') as f:
    data = json.load(f)

print_gpu_info() # Call print_gpu_info
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
client = ModelInitializer(
    model_name=get_model_name(), 
    device=DEVICE,
    default_max_new_tokens=512,
    default_temperature=0.6,  
    default_do_sample=False
)

results = []

for idx, (qid, item) in enumerate(tqdm(data.items(), desc="EQ-Bench")):
    # Add forced output prefix to help model start in right format
    forced_prefix = "First pass scores:"
    prompt_with_prefix = item["prompt"].rstrip() + "\n" + forced_prefix
    gt = item.get("reference_answer", {})
    print(f"-----\nSample {qid}")
    print(f"Prompt:\n{prompt_with_prefix}")
    # Parameters are now set as defaults in ModelInitializer, but can be overridden here if needed for specific calls.
    response_dict = client.gen_response(prompt_with_prefix)
    model_output = response_dict['completion']
    print(f"Raw Model Output:\n{model_output}")
    print(f"Ground Truth:\n{gt}")
    results.append({
        'qid': qid,
        'prompt': prompt_with_prefix,
        'model_output': model_output,
        'ground_truth': gt
    })

with open(RESULTS_OUT, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Done.")
