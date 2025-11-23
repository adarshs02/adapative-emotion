import json
import os
import glob
import sys
from pathlib import Path

# Define paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EMOBIRD_DIR = REPO_ROOT / "EmoBIRDv2" / "eval_results" / "emopatient_multiturn"
BASELINE_DIR = REPO_ROOT / "EmoBIRDv2" / "eval_results" / "emopatient_multiturn_basemodel"

def find_most_recent_file(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def clean_text(text):
    if text:
        return text.strip()
    return ""

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "dialogues" in data:
        return data["dialogues"]
    elif isinstance(data, list):
        return data
    else:
        print(f"Error: Unexpected JSON structure in {filepath}")
        return []

def main():
    # Find the most recent EmoBIRD output file
    emobird_pattern = "emopatient_multiturn_emobirdv2_*.json"
    emobird_file = find_most_recent_file(EMOBIRD_DIR, emobird_pattern)
    
    if not emobird_file:
        print(f"No files matching '{emobird_pattern}' found in {EMOBIRD_DIR}")
        sys.exit(1)
        
    print(f"Processing EmoBIRD file: {emobird_file}")
    emobird_dialogues = load_json_data(emobird_file)
    
    # Find the most recent Baseline output file
    baseline_pattern = "emopatient_multiturn_basemodel_*.json"
    baseline_file = find_most_recent_file(BASELINE_DIR, baseline_pattern)
    
    baseline_map = {}
    if baseline_file:
        print(f"Processing Baseline file: {baseline_file}")
        baseline_dialogues = load_json_data(baseline_file)
        # Create a map for easier lookup
        for d in baseline_dialogues:
            baseline_map[d.get("dialogue_id")] = d
    else:
        print(f"Warning: No files matching '{baseline_pattern}' found in {BASELINE_DIR}. Proceeding without baseline data.")

    cleaned_dialogues = []
    
    for emobird_d in emobird_dialogues:
        d_id = emobird_d.get("dialogue_id")
        baseline_d = baseline_map.get(d_id, {})
        
        cleaned_dialogue = {
            "dialogue_id": d_id,
            "diagnosis": emobird_d.get("diagnosis"),
            "treatment_plan": emobird_d.get("treatment_plan"),
            "narrative": emobird_d.get("narrative"),
            "turns": []
        }
        
        emobird_turns = emobird_d.get("turns", [])
        baseline_turns = baseline_d.get("turns", [])
        
        # Assuming turns are in order and match between files
        for i, turn in enumerate(emobird_turns):
            role = turn.get("role")
            text = turn.get("text", "")
            
            if role == "patient":
                cleaned_dialogue["turns"].append({
                    "role": "patient",
                    "text": clean_text(text)
                })
            elif role == "assistant":
                # Get EmoBIRD output (RECAP) from the text field of the simplified output
                recap_text = text
                
                # Get Baseline output from the corresponding turn in baseline file
                basemodel_text = ""
                if i < len(baseline_turns):
                    baseline_turn = baseline_turns[i]
                    if baseline_turn.get("role") == "assistant":
                        basemodel_text = baseline_turn.get("text", "")
                
                # Add assistant turn with both outputs
                assistant_turn = {
                    "role": "assistant",
                    "basemodel": clean_text(basemodel_text),
                    "RECAP": clean_text(recap_text)
                }
                
                cleaned_dialogue["turns"].append(assistant_turn)
            
        cleaned_dialogues.append(cleaned_dialogue)
        
    # Save the cleaned output
    output_filename = "emopatient_multiturn_merged_cleaned.json"
    output_path = EMOBIRD_DIR / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_dialogues, f, indent=2, ensure_ascii=False)
        
    print(f"Cleaned output saved to: {output_path}")

if __name__ == "__main__":
    main()
