"""
SECEU evaluation script integrated with EmoBIRD's vLLM wrapper and logger.

Features
- Loads SECEU items and standard scores.
- Generates model outputs using VLLMWrapper (text-only response; no JSON enforced).
- Parses four scores per item (sum normalized to 10), robust to extra text.
- Computes SECEU distance score, EQ conversion, and pattern similarity.
- Logs interactions and saves predictions and results into eval_results/.

Usage (example)
python -m EmoBIRD.evaluate_seceu \
  --items /mnt/shared/adarsh/data/seceu/seceu_items.json \
  --standard /mnt/shared/adarsh/data/seceu/seceu_standard.json \
  --max-items 50 \
  --iterations 1 \
  --temperature 0.0 \
  --output-results ./eval_results/seceu_results.json

Notes
- Respects user preference: no terminal commands are run automatically here.
- The prompt instructs the model to output ONLY four numbers separated by spaces, but we parse
defensively with normalization to ensure the sum equals 10.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

# Local imports from EmoBIRD (support both module and script execution)
try:
    # When run as a module: python -m EmoBIRD.evaluate_seceu
    from .config import EmobirdConfig
    from .vllm_wrapper import VLLMWrapper
    from .logger import get_logger
except ImportError:
    # When run as a script: python evaluate_seceu.py from the EmoBIRD/ directory
    from config import EmobirdConfig
    from vllm_wrapper import VLLMWrapper
    from logger import get_logger


DATA_ROOT = Path("/mnt/shared/adarsh/data/seceu")


def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_standard(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores_with_info(text: str, fallback: List[float] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract four numeric scores from model text robustly and normalize to sum 10.
    Strategy (improved):
    - Prefer the first 4 numbers immediately after the FIRST occurrence of 'Final Scores:' (case-insensitive).
    - Collect additional candidates: after each subsequent 'Final Scores:' marker, the FIRST 4 numbers; also the first 4 anywhere and the last 4 anywhere.
    - Avoid all-zero groups when a non-zero candidate exists.
    - If multiple candidates remain, choose the one with sum > 0 that is closest to 10; if none > 0, choose the one with the largest (non-negative) sum; else fallback.
    - Finally, clip negatives to 0 and rescale to make the sum exactly 10.
    Returns (scores_array, info_dict) where info includes method and raw numbers.
    """
    if fallback is None:
        fallback = [2.5, 2.5, 2.5, 2.5]
    info: Dict[str, Any] = {}
    try:
        float_pattern = re.compile(r"[-+]?\d*\.?\d+")

        # Gather raw numbers across the whole text
        raw_numbers_all = [float(x) for x in float_pattern.findall(text)]

        candidates: List[Tuple[str, List[float]]] = []

        # Candidates from 'Final Scores:' markers (take the FIRST 4 after each marker)
        markers = list(re.finditer(r"final\s*scores\s*:\s*", text, flags=re.IGNORECASE))
        for i, m in enumerate(markers):
            tail = text[m.end():]
            nums_after = [float(x) for x in float_pattern.findall(tail)]
            if len(nums_after) >= 4:
                label = "after_final_scores_first" if i == 0 else f"after_final_scores_{i+1}"
                candidates.append((label, nums_after[:4]))

        # First four anywhere
        if len(raw_numbers_all) >= 4:
            candidates.append(("first_four_anywhere", raw_numbers_all[:4]))
            candidates.append(("last_four_anywhere", raw_numbers_all[-4:]))

        # If no candidates, fallback immediately
        if not candidates:
            arr = np.array(fallback, dtype=float)
            info.update({
                "method": "no_candidates_fallback",
                "raw_numbers": [],
                "chosen": fallback,
                "before_sum": float(arr.sum()),
                "normalized_sum": 10.0,
            })
            return arr, info

        # Remove all-zero candidates if any non-zero exists
        sums = [sum(vals) for _lab, vals in candidates]
        any_non_zero = any(s > 0 for s in sums)
        filtered = []
        for (lab, vals), s in zip(candidates, sums):
            if any_non_zero and s == 0:
                continue
            filtered.append((lab, vals))
        if filtered:
            candidates = filtered

        # Selection policy
        chosen_label = None
        chosen_numbers: List[float] = []

        # 1) Direct preference: first marker group if present and non-zero
        for lab, vals in candidates:
            if lab == "after_final_scores_first" and sum(vals) > 0:
                chosen_label, chosen_numbers = lab, vals
                break

        # 2) If not chosen, pick sum>0 closest to 10
        if not chosen_numbers:
            best = None
            best_lab = None
            for lab, vals in candidates:
                s = sum(vals)
                if s > 0:
                    diff = abs(s - 10.0)
                    if (best is None) or (diff < best[0]):
                        best = (diff, vals)
                        best_lab = lab
            if best is not None:
                chosen_label, chosen_numbers = best_lab, best[1]

        # 3) If still not chosen, pick the largest-sum candidate (could be zero)
        if not chosen_numbers:
            best = None
            best_lab = None
            for lab, vals in candidates:
                s = sum(vals)
                if (best is None) or (s > best[0]):
                    best = (s, vals)
                    best_lab = lab
            if best is not None:
                chosen_label, chosen_numbers = best_lab, best[1]

        # Final safety: if all failed for some reason
        if not chosen_numbers:
            arr = np.array(fallback, dtype=float)
            info.update({
                "method": "selection_fallback",
                "raw_numbers": raw_numbers_all,
                "chosen": fallback,
                "before_sum": float(arr.sum()),
                "normalized_sum": 10.0,
            })
            return arr, info

        # Clip negatives to 0, then rescale to sum 10
        arr = np.maximum(np.array(chosen_numbers, dtype=float), 0.0)
        s = float(arr.sum())
        if s <= 0:
            before_sum = 0.0
            arr = np.array(fallback, dtype=float)
            method = f"{chosen_label}_zero_sum_fallback"
        else:
            before_sum = s
            arr = arr / s * 10.0
            method = chosen_label

        info.update({
            "method": method,
            "raw_numbers": raw_numbers_all,
            "chosen": chosen_numbers,
            "before_sum": before_sum,
            "normalized_sum": float(arr.sum()),
        })
        return arr, info
    except Exception:
        arr = np.array(fallback, dtype=float)
        info.update({
            "method": "exception_fallback",
            "raw_numbers": [],
            "chosen": fallback,
            "before_sum": float(arr.sum()),
            "normalized_sum": 10.0,
        })
        return arr, info


def extract_scores(text: str, fallback: List[float] | None = None) -> np.ndarray:
    """Backward-compatible wrapper returning only the normalized scores."""
    scores, _info = extract_scores_with_info(text, fallback=fallback)
    return scores


def build_prompt(story: str, options: List[str]) -> str:
    opts = ", ".join(f"({i+1}) {opt}" for i, opt in enumerate(options))
    return f'''You are an empathetic AI assistant. Your task is to carefully read the following story and evaluate the emotional state of the main character.
You will be given four emotion options. For each option, assign a score from 0 to 10 representing how intensely the main character feels that emotion.

**Important Constraints:**
1. Each score must be between 0 and 10 (inclusive).
2. The sum of your scores for the four options MUST be exactly 10.

Story:
{story}

Emotion Options:
{opts}

Please follow these steps in your reasoning before providing the scores:
1.  Deeply analyze the provided story, focusing on the main character's situation, actions, and any explicit or implicit emotional cues.
2.  For each of the four emotion options, critically assess its relevance and intensity concerning the character's experience.
3.  Assign an initial numerical score (0-10) to each emotion based on your analysis.
4.  Verify that the sum of your four scores is exactly 10. If not, carefully adjust the scores, maintaining their relative proportions as much as possible, until they sum precisely to 10.
5.  Provide ONLY the four final numerical scores, separated by spaces (e.g., 1.5 3.0 4.5 1.0). Do not add any other text or explanation before or after the scores.

Output exactly one line with four numbers and nothing else.

Final Scores:'''


def main():
    parser = argparse.ArgumentParser(description="Evaluate SECEU with EmoBIRD VLLMWrapper")
    parser.add_argument("--items", type=str, default="")
    parser.add_argument("--standard", type=str, default="")
    parser.add_argument("--max-items", type=int, default=0, help="0 = all")
    parser.add_argument("--iterations", type=int, default=1, help="repeat the full set and average")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--output-preds", type=str, default="")
    parser.add_argument("--output-results", type=str, default="")
    args = parser.parse_args()

    # Resolve defaults (prefer project-root data path if present, else fallback to DATA_ROOT)
    project_root = Path(__file__).resolve().parent.parent
    default_items = project_root / "data" / "seceu" / "seceu_items.json"
    default_std = project_root / "data" / "seceu" / "seceu_standard.json"
    items_path = Path(args.items) if args.items else (default_items if default_items.exists() else (DATA_ROOT / "seceu_items.json"))
    std_path = Path(args.standard) if args.standard else (default_std if default_std.exists() else (DATA_ROOT / "seceu_standard.json"))
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not std_path.exists():
        raise FileNotFoundError(f"Standard file not found: {std_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    # Keep output directory same as earlier: EmoBIRD/eval_results/
    preds_out = Path(args.output_preds) if args.output_preds else (Path(__file__).parent / "eval_results" / f"seceu_predictions_{ts}.json")
    results_out = Path(args.output_results) if args.output_results else (Path(__file__).parent / "eval_results" / f"seceu_results_{ts}.json")
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.parent.mkdir(parents=True, exist_ok=True)

    # Config and model
    config = EmobirdConfig()
    if args.temperature is not None:
        config.update_config(temperature=float(args.temperature))
    wrapper = VLLMWrapper(config)

    # Enforce deterministic, short, single-line numeric output for SECEU
    # - temperature: 0.0 (deterministic)
    # - top_p: 1.0 (disable nucleus bias)
    # - max_tokens: 24 (enough for "a b c d")
    # - stop: newline to stop after first line of numbers
    try:
        wrapper.update_sampling_params(temperature=0.0, top_p=1.0, max_tokens=24, stop=["\n"])
    except Exception:
        # Best-effort; continue with defaults if update not supported
        pass

    # Load data
    items = load_items(items_path)
    standard = load_standard(std_path)
    standard_scores = np.array(standard["standard_scores"], dtype=float)
    human_pattern = np.array(standard["human_pattern"], dtype=float)
    population_mean = float(standard["population_mean"])
    population_std = float(standard["population_std"])

    # Optionally truncate items
    item_list = list(items)
    if args.max_items and args.max_items > 0:
        item_list = item_list[: args.max_items]

    logger = get_logger()

    # Run for N iterations and average distances pattern across iterations
    all_iter_distances: List[np.ndarray] = []
    all_iter_preds: List[List[List[float]]] = []

    for it in range(1, args.iterations + 1):
        iter_preds: List[List[float]] = []
        distances_iter: List[float] = []
        for idx, item in enumerate(tqdm(item_list, desc=f"Processing items (iter {it})")):
            story = item["story"]
            options = item["options"]
            prompt = build_prompt(story, options)

            # Generate model output
            response = wrapper.generate(prompt, component="seceu", interaction_type="seceu_prompt")

            # Log raw interaction
            logger.log_interaction(
                component="seceu",
                interaction_type="generation",
                prompt=prompt,
                response=response,
                metadata={"item_index": idx, "iteration": it},
            )

            # Parse to four scores, normalize to sum 10
            scores, parse_info = extract_scores_with_info(response)
            iter_preds.append(scores.tolist())

            # Log parsed results for traceability
            logger.log_interaction(
                component="seceu",
                interaction_type="parsed_scores",
                prompt="<omitted>",
                response=" ".join(f"{x:.6f}" for x in scores.tolist()),
                metadata={
                    "item_index": idx,
                    "iteration": it,
                    "method": parse_info.get("method"),
                    "chosen_before_normalization": parse_info.get("chosen"),
                    "sum_before_normalization": parse_info.get("before_sum"),
                    "normalized_sum": parse_info.get("normalized_sum"),
                },
            )

            # Distance to standard for this item
            distances_iter.append(float(np.linalg.norm(scores - np.array(standard_scores[idx], dtype=float), ord=2)))

        all_iter_preds.append(iter_preds)
        all_iter_distances.append(np.array(distances_iter, dtype=float))

    # Aggregate across iterations by averaging distances per item, then mean
    distances = np.mean(np.vstack(all_iter_distances), axis=0)
    seceu_score = float(np.mean(distances))
    eq_score = float(15.0 * ((population_mean - seceu_score) / population_std) + 100.0)
    pattern_similarity = float(pearsonr(distances, human_pattern)[0])

    # Save predictions (keep only final iteration if single iteration, else save all)
    preds_payload: Dict[str, Any] = {"predictions": all_iter_preds[-1] if args.iterations == 1 else all_iter_preds}
    with preds_out.open("w", encoding="utf-8") as f:
        json.dump(preds_payload, f, indent=2)

    # Save results
    results = {
        "seceu_score": round(seceu_score, 3),
        "eq_score": round(eq_score, 2),
        "pattern_similarity": round(pattern_similarity, 3),
        "meta": {
            "items_path": str(items_path),
            "standard_path": str(std_path),
            "iterations": args.iterations,
            "model_name": config.llm_model_name,
            "sampling_temperature": config.temperature,
        },
    }
    with results_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ SECEU evaluation complete. SECEU Score: {results['seceu_score']}")
    print(f"📄 Predictions saved to: {preds_out}")
    print(f"📄 Results saved to: {results_out}")


if __name__ == "__main__":
    main()
