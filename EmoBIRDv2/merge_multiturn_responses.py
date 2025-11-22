#!/usr/bin/env python3
"""
Merge multi-turn evaluation responses from basemodel and EmoBIRDv2.

Creates a side-by-side comparison where each assistant turn shows:
- role: assistant
- basemodel: <response from baseline>
- RECAP: <response from EmoBIRDv2>

Usage:
  python EmoBIRDv2/merge_multiturn_responses.py basemodel.json emobirdv2.json output.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def merge_responses(basemodel_data: Dict[str, Any], emobirdv2_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge assistant responses from basemodel and EmoBIRDv2 results.
    
    Args:
        basemodel_data: Cleaned baseline model results
        emobirdv2_data: Cleaned EmoBIRDv2 results
    
    Returns:
        Merged JSON with side-by-side assistant responses
    """
    # Create mapping of dialogue_id -> dialogue for both datasets
    basemodel_dialogues = {d["dialogue_id"]: d for d in basemodel_data.get("dialogues", [])}
    emobirdv2_dialogues = {d["dialogue_id"]: d for d in emobirdv2_data.get("dialogues", [])}
    
    merged_dialogues = []
    
    # Get all dialogue IDs from both datasets
    all_dialogue_ids = set(basemodel_dialogues.keys()) | set(emobirdv2_dialogues.keys())
    
    for dialogue_id in sorted(all_dialogue_ids):
        base_dialogue = basemodel_dialogues.get(dialogue_id)
        emo_dialogue = emobirdv2_dialogues.get(dialogue_id)
        
        if not base_dialogue or not emo_dialogue:
            print(f"Warning: {dialogue_id} missing from one dataset, skipping", flush=True)
            continue
        
        # Use basemodel dialogue as template
        merged_dialogue = {
            "dialogue_id": dialogue_id,
            "diagnosis": base_dialogue.get("diagnosis"),
            "treatment_plan": base_dialogue.get("treatment_plan"),
            "narrative": base_dialogue.get("narrative"),
            "turns": []
        }
        
        # Merge turns
        base_turns = base_dialogue.get("turns", [])
        emo_turns = emo_dialogue.get("turns", [])
        
        # Ensure both have the same number of turns
        if len(base_turns) != len(emo_turns):
            print(f"Warning: {dialogue_id} has different number of turns (base={len(base_turns)}, emo={len(emo_turns)})", flush=True)
        
        for turn_idx in range(min(len(base_turns), len(emo_turns))):
            base_turn = base_turns[turn_idx]
            emo_turn = emo_turns[turn_idx]
            
            role = base_turn.get("role")
            
            if role == "patient":
                # Patient turns should be identical, just keep one
                merged_dialogue["turns"].append({
                    "role": "patient",
                    "text": base_turn.get("text")
                })
            elif role == "assistant":
                # Assistant turns - merge both responses
                merged_dialogue["turns"].append({
                    "role": "assistant",
                    "basemodel": base_turn.get("text"),
                    "RECAP": emo_turn.get("text")
                })
        
        merged_dialogues.append(merged_dialogue)
    
    return {
        "basemodel_run_id": basemodel_data.get("run_id"),
        "emobirdv2_run_id": emobirdv2_data.get("run_id"),
        "model": basemodel_data.get("model"),
        "temperature": basemodel_data.get("temperature"),
        "dialogues": merged_dialogues
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge basemodel and EmoBIRDv2 multi-turn responses"
    )
    parser.add_argument("basemodel_file", type=str, help="Cleaned basemodel JSON file")
    parser.add_argument("emobirdv2_file", type=str, help="Cleaned EmoBIRDv2 JSON file")
    parser.add_argument("output_file", type=str, help="Output merged JSON file")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2)",
    )
    
    args = parser.parse_args()
    
    basemodel_path = Path(args.basemodel_file)
    emobirdv2_path = Path(args.emobirdv2_file)
    output_path = Path(args.output_file)
    
    if not basemodel_path.exists():
        print(f"Error: Basemodel file not found: {basemodel_path}")
        return 1
    
    if not emobirdv2_path.exists():
        print(f"Error: EmoBIRDv2 file not found: {emobirdv2_path}")
        return 1
    
    # Load inputs
    print(f"Loading {basemodel_path}...", flush=True)
    with basemodel_path.open("r", encoding="utf-8") as f:
        basemodel_data = json.load(f)
    
    print(f"Loading {emobirdv2_path}...", flush=True)
    with emobirdv2_path.open("r", encoding="utf-8") as f:
        emobirdv2_data = json.load(f)
    
    # Merge
    print("Merging responses...", flush=True)
    merged_data = merge_responses(basemodel_data, emobirdv2_data)
    
    # Save
    print(f"Saving to {output_path}...", flush=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=args.indent)
    
    # Report
    print(f"Done!")
    print(f"  Dialogues merged: {len(merged_data.get('dialogues', []))}")
    print(f"  Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
