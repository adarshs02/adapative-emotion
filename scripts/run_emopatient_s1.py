#!/usr/bin/env python3
"""
Run EmoBIRD on the EmoPatient dataset (S1 only).

- Loads the first scenario from datasets/EmoPatient/scenarios.json
- For each question, feeds the scenario context + question into EmoBIRD
- Prints the model's generated answer (ignores ground-truth)

Usage:
  python -u /mnt/shared/adarsh/scripts/run_emopatient_s1.py

Environment:
  EMOBIRD_MODEL can be set (defaults to Llama via EmobirdConfig)
"""

import json
from pathlib import Path
from typing import Dict, Any

# Import the EmoBIRD engine
from EmoBIRD.emobird_poc import Emobird

DATASET_PATH = Path("/mnt/shared/adarsh/datasets/EmoPatient/scenarios.json")


def load_first_scenario(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError("No scenarios found in dataset")
    return scenarios[0]


def compose_situation_text(scn: Dict[str, Any], question: str) -> str:
    title = scn.get("title", "")
    diagnosis = scn.get("diagnosis", "")
    tplan = scn.get("treatment_plan", "")
    narrative = scn.get("narrative", "")

    situation = (
        f"Context (clinical scenario):\n"
        f"Title: {title}\n"
        f"Diagnosis: {diagnosis}\n"
        f"Treatment plan: {tplan}\n\n"
        f"Patient narrative:\n{narrative}\n\n"
        f"Question:\n{question}\n\n"
        f"Please provide a concise, empathetic, medically grounded answer tailored to the above context."
    )
    return situation


def main():
    print("\n📄 Loading dataset ...")
    scenario = load_first_scenario(DATASET_PATH)
    qa_list = scenario.get("qa", [])
    if not qa_list:
        raise ValueError("First scenario has no QA items")

    # Initialize EmoBIRD
    print("🐦 Initializing EmoBIRD engine ...")
    emo = Emobird()

    print("\n🚦 Running questions for Scenario S1 ...\n")
    for idx, qa in enumerate(qa_list, start=1):
        q = qa.get("q", "").strip()
        if not q:
            continue
        situation = compose_situation_text(scenario, q)

        print("=" * 80)
        print(f"Q{idx}: {q}")
        print("-" * 80)

        try:
            result = emo.analyze_emotion(situation)
        except Exception as e:
            print(f"❌ Error running EmoBIRD: {e}")
            continue

        # Print the model's answer/response
        response = result.get("response") or "(no response generated)"
        print("Answer:\n")
        print(response)

        # Optional: quick emotion summary
        emotions = result.get("emotions") or {}
        if emotions:
            print("\nTop emotions (model):")
            for em, p in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {em}: {p:.3f}")

    print("\n✅ Completed S1 run.")


if __name__ == "__main__":
    main()
