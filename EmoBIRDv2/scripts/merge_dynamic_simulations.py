import json
import sys
import re
from pathlib import Path

# Define paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EMOBIRD_FILE = REPO_ROOT / "EmoBIRDv2" / "eval_results" / "simulation_results_emobird.json"
BASELINE_FILE = REPO_ROOT / "EmoBIRDv2" / "eval_results" / "simulation_results_baseline.json"
OUTPUT_FILE = REPO_ROOT / "EmoBIRDv2" / "eval_results" / "simulation_results_merged.json"

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def load_json(filepath):
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}", file=sys.stderr)
        return []

def main():
    print(f"Loading EmoBIRD results from {EMOBIRD_FILE}...")
    emobird_data = load_json(EMOBIRD_FILE)
    
    print(f"Loading Baseline results from {BASELINE_FILE}...")
    baseline_data = load_json(BASELINE_FILE)
    
    # Create a map for baseline data
    baseline_map = {item["dialogue_id"]: item for item in baseline_data}
    
    merged_results = []
    
    for emobird_item in emobird_data:
        dialogue_id = emobird_item["dialogue_id"]
        baseline_item = baseline_map.get(dialogue_id)
        
        if not baseline_item:
            print(f"Warning: No baseline data found for {dialogue_id}", file=sys.stderr)
            continue
            
        merged_item = {
            "dialogue_id": dialogue_id,
            "diagnosis": emobird_item.get("diagnosis"),
            "treatment_plan": emobird_item.get("treatment_plan"),
            "demographics": emobird_item.get("demographics"),
            "baseline": baseline_item.get("transcript", []),
            "RECAP": emobird_item.get("transcript", [])
        }
        
        merged_results.append(merged_item)
        
    # Sort results
    merged_results.sort(key=lambda x: natural_sort_key(x["dialogue_id"]))
    
    # Save merged results
    print(f"Saving merged results to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)
        
    print("Merge complete.")

if __name__ == "__main__":
    main()
