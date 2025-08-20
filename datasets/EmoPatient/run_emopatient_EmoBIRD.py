#!/usr/bin/env python3
"""
Run EmoBIRD on the EmoPatient dataset (all scenarios) from the EmoPatient directory.

- Loads all scenarios from ./scenarios_30.json
- For each question, feeds the scenario context + question into EmoBIRD
- Prints the model's generated answer (ignores ground-truth)

Usage:
  python -u run_emopatient_s1.py

Environment:
  EMOBIRD_MODEL can be set (defaults to Llama via EmobirdConfig)
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Ensure repository root is on sys.path so we can import EmoBIRD
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # /mnt/shared/adarsh
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the EmoBIRD engine
from EmoBIRD.emobird_poc import Emobird
from EmoBIRD.logger import EmobirdLogger, set_logger, get_logger, close_logger

# Visual progress bar (with safe fallback)
try:
    from tqdm.auto import tqdm  # prefer rich display when available
except Exception:
    class _TqdmFallback:
        def __call__(self, iterable, **kwargs):
            return iterable

        @staticmethod
        def write(msg: str):
            print(msg)

    tqdm = _TqdmFallback()

DATASET_PATH = HERE.parent / "scenarios_30.json"
RESULTS_DIR = HERE.parent / "results"
LOGS_DIR = HERE.parent / "logs"


def load_scenarios(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError("No scenarios found in dataset")
    return scenarios


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
    scenarios = load_scenarios(DATASET_PATH)
    if not scenarios:
        raise ValueError("No scenarios found in dataset")

    # Initialize structured logger for this run and make it global
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = EmobirdLogger(log_dir=str(LOGS_DIR))
    set_logger(logger)
    run_id = logger.session_id  # reuse logger session id for output files

    # Initialize EmoBIRD
    print("🐦 Initializing EmoBIRD engine ...")
    emo = Emobird()

    # Prepare results directory and metadata
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model_info = emo.vllm_wrapper.get_model_info()
    except Exception:
        model_info = None

    # Iterate over all scenarios
    for s_idx, scenario in enumerate(tqdm(scenarios, desc="Scenarios", unit="scn"), start=1):
        qa_list = scenario.get("qa", [])
        if not qa_list: 
            tqdm.write(f"⚠️ Scenario {s_idx} has no QA items; skipping.")
            continue

        scn_id = str(scenario.get("id") or f"S{s_idx}").replace(" ", "_")
        title = scenario.get("title", "")
        print(f"\n🚦 Running questions for Scenario {scn_id} — {title}\n")

        results: List[Dict[str, Any]] = []
        for idx, qa in enumerate(tqdm(qa_list, desc="QAs", unit="q"), start=1):
            q = qa.get("q", "").strip()
            if not q:
                continue
            situation = compose_situation_text(scenario, q)

            tqdm.write("=" * 80)
            tqdm.write(f"Q{idx}: {q}")
            tqdm.write("-" * 80)

            try:
                result = emo.analyze_emotion(situation)
            except Exception as e:
                tqdm.write(f"❌ Error running EmoBIRD: {e}")
                continue

            # Print the model's answer/response
            response = result.get("response") or "(no response generated)"
            tqdm.write("Answer:\n")
            tqdm.write(response)

            # Optional: quick emotion summary
            emotions = result.get("emotions") or {}
            if emotions:
                tqdm.write("\nTop emotions (model):")
                for em, p in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]:
                    tqdm.write(f"  - {em}: {p:.3f}")

            # Collect record for results file
            top5 = (
                sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
                if emotions else []
            )
            rec = {
                "index": idx,
                "question": q,
                "answer": response,
                "emotions": emotions,
                "top_emotions": top5,
                "explanation": result.get("explanation"),
                "metadata": result.get("metadata"),
            }
            results.append(rec)

            # Log the full analysis result to the session logs
            try:
                get_logger().log_analysis_result(situation, result)
            except Exception as e:
                # Non-fatal; continue
                tqdm.write(f"⚠️ Failed to log analysis result: {e}")

        # Write results for this scenario
        try:
            log_info = get_logger().get_session_info()
        except Exception:
            log_info = None

        payload = {
            "scenario_id": scenario.get("id"),
            "title": scenario.get("title"),
            "num_questions": len(qa_list),
            "run_id": run_id,
            "model_info": model_info,
            "log_info": log_info,
            "items": results,
        }
        out_path = RESULTS_DIR / f"{scn_id}_results_{run_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Saved results to: {out_path}")

    # Close the logging session
    try:
        close_logger()
    except Exception as e:
        print(f"⚠️ Failed to close logger: {e}")

    print("\n✅ Completed run across all scenarios.")


if __name__ == "__main__":
    main()
