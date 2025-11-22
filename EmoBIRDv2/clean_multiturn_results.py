#!/usr/bin/env python3
"""
Clean multi-turn evaluation results by removing pipeline information.

Keeps only essential fields:
- dialogue_id
- diagnosis
- treatment_plan
- narrative
- turns (conversation history)

Removes:
- turn_results (pipeline details)
- messages (intermediate pipeline data)

Usage:
  python EmoBIRDv2/clean_multiturn_results.py input.json output.json
  python EmoBIRDv2/clean_multiturn_results.py input.json  # overwrites input file
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def clean_dialogue(dialogue: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a single dialogue by keeping only essential fields."""
    cleaned = {
        "dialogue_id": dialogue.get("dialogue_id"),
        "diagnosis": dialogue.get("diagnosis"),
        "treatment_plan": dialogue.get("treatment_plan"),
        "narrative": dialogue.get("narrative"),
        "turns": dialogue.get("turns", []),
    }
    return cleaned


def clean_results(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean evaluation results by removing pipeline information.
    
    Args:
        input_data: The full evaluation results JSON
    
    Returns:
        Cleaned JSON with only essential fields
    """
    dialogues = input_data.get("dialogues", [])
    
    cleaned_dialogues = []
    for dialogue in dialogues:
        cleaned_dialogues.append(clean_dialogue(dialogue))
    
    # Keep top-level metadata but replace dialogues
    cleaned = {
        "run_id": input_data.get("run_id"),
        "model": input_data.get("model"),
        "temperature": input_data.get("temperature"),
        "dialogues": cleaned_dialogues,
    }
    
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Clean multi-turn evaluation results by removing pipeline info"
    )
    parser.add_argument("input_file", type=str, help="Input JSON file")
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        default=None,
        help="Output JSON file (default: overwrites input)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2)",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file) if args.output_file else input_path
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Load input
    print(f"Loading {input_path}...", flush=True)
    with input_path.open("r", encoding="utf-8") as f:
        input_data = json.load(f)
    
    # Clean
    print("Cleaning results...", flush=True)
    cleaned_data = clean_results(input_data)
    
    # Save
    print(f"Saving to {output_path}...", flush=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=args.indent)
    
    # Report
    input_dialogues = len(input_data.get("dialogues", []))
    output_dialogues = len(cleaned_data.get("dialogues", []))
    
    print(f"Done!")
    print(f"  Dialogues: {input_dialogues} → {output_dialogues}")
    print(f"  Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
