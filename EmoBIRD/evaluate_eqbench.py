"""
EQ-Bench evaluation script integrated with EmoBIRD's vLLM wrapper and logger.

Features
- Loads EQ-Bench v1/v2 question files (EN/DE).
- Generates model outputs using VLLMWrapper.
- Parses First pass / Revised scores with official parsers.
- Scores with official formulas (v1 normalized, v2 fullscale).
- Aggregates iteration-level and overall scores.
- Logs interactions and saves structured results in eval_results/.

Usage (example)
python -m EmoBIRD.evaluate_eqbench \
  --dataset v2 \
  --lang en \
  --questions /mnt/shared/adarsh/datasets/EQ-Bench/data/eq_bench_v2_questions_171.json \
  --iterations 1 \
  --max-items 50 \
  --output ./eval_results/eqbench_v2_en.json

Notes
- Respects user preference: no terminal commands are run automatically here.
- This evaluator calls the model with the raw EQ-Bench prompt and expects the specific text format required by EQ-Bench.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Local imports from EmoBIRD (support both module and script execution)
try:
    # When run as a module: python -m EmoBIRD.evaluate_eqbench
    from .config import EmobirdConfig
    from .vllm_wrapper import VLLMWrapper
    from .logger import get_logger
except ImportError:
    # When run as a script: python evaluate_eqbench.py from the EmoBIRD/ directory
    from config import EmobirdConfig
    from vllm_wrapper import VLLMWrapper
    from logger import get_logger

# Add EQ-Bench repo root to path and import official scoring/parsing
EQ_BENCH_ROOT = Path("/mnt/shared/adarsh/datasets/EQ-Bench")
if EQ_BENCH_ROOT.exists():
    sys.path.insert(0, str(EQ_BENCH_ROOT))
try:
    from lib.scoring import (
        parse_answers,
        parse_answers_de,
        calculate_score,
        calculate_score_fullscale,
    )
except Exception as e:
    raise ImportError(
        f"Failed to import EQ-Bench scoring utilities from {EQ_BENCH_ROOT}. Error: {e}"
    )


@dataclass
class ItemScore:
    first_pass_score: Optional[float]
    revised_score: Optional[float]
    first_pass_parseable: bool
    revised_parseable: bool


def load_questions(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # JSON is a dict keyed by string ids
    return data


def parse_both(text: str, lang: str, revise: bool) -> Tuple[Dict[str, str], Dict[str, str]]:
    if lang.lower() == "de":
        fp, rv = parse_answers_de(text, REVISE=revise)
    else:
        fp, rv = parse_answers(text, REVISE=revise)
    return fp, rv


def score_item(
    reference: Dict[str, Any],
    user_fp: Dict[str, str],
    user_rv: Dict[str, str],
    dataset: str,
) -> ItemScore:
    # Convert user scores to numeric (keep keys as-is; EQ-Bench comparer is case-insensitive for v2)
    def to_numeric(d: Dict[str, str]) -> Dict[str, float]:
        out = {}
        for k, v in d.items():
            try:
                out[k] = float(v)
            except Exception:
                # Ignore non-numeric entries
                pass
        return out

    fp_num = to_numeric(user_fp)
    rv_num = to_numeric(user_rv)

    # Choose reference set and calculator
    if dataset == "v2":
        ref = reference.get("reference_answer_fullscale") or reference.get("reference_answer")
        calc = calculate_score_fullscale
    else:  # v1 (normalized)
        ref = reference.get("reference_answer")
        calc = calculate_score

    fp_score = None
    rv_score = None

    if fp_num:
        try:
            fp_score = calc(ref, fp_num)
        except Exception:
            fp_score = None
    if rv_num:
        try:
            rv_score = calc(ref, rv_num)
        except Exception:
            rv_score = None

    return ItemScore(
        first_pass_score=fp_score,
        revised_score=rv_score,
        first_pass_parseable=fp_score is not None,
        revised_parseable=rv_score is not None,
    )


def choose_iteration_final(
    scores: Dict[str, ItemScore]
) -> Tuple[float, int, float, int, float, int]:
    """
    Mirrors the EQ-Bench iteration aggregation logic:
    - Compute average First pass and Revised scores over parseable items.
    - If Revised >= First and revised_parseable >= 0.95 * first_pass_parseable, choose Revised;
      else choose First.
    Returns:
      (first_avg_0_100, first_cnt, rev_avg_0_100, rev_cnt, final_avg_0_100, final_cnt)
    """
    first_scores = [s.first_pass_score for s in scores.values() if s.first_pass_score is not None]
    rev_scores = [s.revised_score for s in scores.values() if s.revised_score is not None]

    first_cnt = len(first_scores)
    rev_cnt = len(rev_scores)

    first_avg = 100.0 * (sum(first_scores) / first_cnt / 10.0) if first_cnt else 0.0
    rev_avg = 100.0 * (sum(rev_scores) / rev_cnt / 10.0) if rev_cnt else 0.0

    if (rev_avg >= first_avg) and (rev_cnt >= 0.95 * first_cnt if first_cnt else False):
        final_avg = rev_avg
        final_cnt = rev_cnt
    else:
        final_avg = first_avg
        final_cnt = first_cnt

    return first_avg, first_cnt, rev_avg, rev_cnt, final_avg, final_cnt


def main():
    parser = argparse.ArgumentParser(description="Evaluate EQ-Bench with EmoBIRD vLLMWrapper")
    parser.add_argument("--dataset", choices=["v1", "v2"], default="v2")
    parser.add_argument("--lang", choices=["en", "de"], default="en")
    parser.add_argument("--questions", type=str, default="")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=0, help="0 = all")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    # Resolve default questions path if not provided
    if not args.questions:
        if args.dataset == "v2":
            args.questions = (
                EQ_BENCH_ROOT / "data" / ("eq_bench_v2_questions_171_de.json" if args.lang == "de" else "eq_bench_v2_questions_171.json")
            )
        else:
            args.questions = EQ_BENCH_ROOT / "data" / "eq_bench_v1_questions_60.json"
    q_path = Path(args.questions)
    if not q_path.exists():
        raise FileNotFoundError(f"Questions file not found: {q_path}")

    # Output path
    if not args.output:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = Path(__file__).parent / "eval_results" / f"eqbench_{args.dataset}_{args.lang}_{ts}.json"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Config and model
    config = EmobirdConfig()
    if args.temperature is not None:
        config.update_config(temperature=float(args.temperature))
    wrapper = VLLMWrapper(config)

    # Load questions
    questions = load_questions(q_path)
    item_ids = sorted(questions.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    if args.max_items and args.max_items > 0:
        item_ids = item_ids[: args.max_items]

    logger = get_logger()
    revise_required = True  # Both v1 and v2 prompt formats include Revised section

    # Results structure
    results: Dict[str, Any] = {
        "meta": {
            "dataset": args.dataset,
            "lang": args.lang,
            "questions_path": str(q_path),
            "iterations": args.iterations,
            "model_name": config.llm_model_name,
            "sampling_temperature": config.temperature,
        },
        "iterations": {},
    }

    for it in range(1, args.iterations + 1):
        iter_key = str(it)
        per_item_scores: Dict[str, ItemScore] = {}

        for qid in item_ids:
            q = questions[qid]
            prompt = q["prompt"]

            # Generate model output (text format; not JSON)
            response = wrapper.generate(prompt, component="eqbench", interaction_type="eq_prompt")

            # Log the raw interaction
            logger.log_interaction(
                component="eqbench",
                interaction_type="generation",
                prompt=prompt,
                response=response,
                metadata={"qid": qid, "iteration": it},
            )

            # Parse
            fp_dict, rv_dict = parse_both(response, args.lang, revise_required)

            # Score
            per_item_scores[qid] = score_item(q, fp_dict, rv_dict, dataset=args.dataset)

        # Aggregate iteration
        first_avg, first_cnt, rev_avg, rev_cnt, final_avg, final_cnt = choose_iteration_final(per_item_scores)

        # Persist iteration results minimally
        results["iterations"][iter_key] = {
            "first_pass_avg": round(first_avg, 2),
            "first_pass_parseable": first_cnt,
            "revised_avg": round(rev_avg, 2),
            "revised_parseable": rev_cnt,
            "final_avg": round(final_avg, 2),
            "final_parseable": final_cnt,
            "items": {
                qid: {
                    "first_pass_score": s.first_pass_score,
                    "revised_score": s.revised_score,
                }
                for qid, s in per_item_scores.items()
            },
        }

    # Overall average of final_avg across iterations
    finals = [itres["final_avg"] for itres in results["iterations"].values()]
    overall = sum(finals) / len(finals) if finals else 0.0
    results["overall_final_avg"] = round(overall, 2)

    # Save
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ EQ-Bench evaluation complete. Overall (0-100): {results['overall_final_avg']}")
    print(f"📄 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
