#!/usr/bin/env python3
"""
SEC-EU evaluation using the EmoBIRDv2 pipeline via OpenRouter.

Flow per item (story):
1) Abstract from situation
2) Factors from abstract
3) Likert ratings for the 4 SEC-EU options (treated as the emotion list)
4) Map Likert ratings to weights, renormalize to produce 4 scores summing to 10

Outputs mirror scripts/seceu_eval.py:
- results/seceu/seceu_emobirdv2_predictions.json
- results/seceu/seceu_emobirdv2_results.json

Requires OPENROUTER_API_KEY in env.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# EmoBIRDv2 imports
from EmoBIRDv2.utils.constants import (
    OPENROUTER_API_KEY,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    ABSTRACT_MAX_TOKENS,
    FACTOR_MAX_TOKENS,
    LIKERT_MAX_TOKENS,
    LIKERT_SCALE,
)
from EmoBIRDv2.scripts.abstract_generator import (
    load_prompt as load_abs_prompt,
    build_user_prompt as build_abs_user,
    call_openrouter,
)
from EmoBIRDv2.scripts.factor_generator import (
    load_prompt as load_fac_prompt,
    build_user_prompt as build_fac_user,
    parse_factor_block,
)
from EmoBIRDv2.scripts.likert_matcher import (
    load_prompt as load_lik_prompt,
    build_user_prompt as build_lik_user,
    parse_likert_lines,
)
from EmoBIRDv2.utils.utils import robust_json_loads
from pathlib import Path
import re


ITEMS_PATH = REPO_ROOT / "data" / "seceu" / "seceu_items.json"
STANDARD_PATH = REPO_ROOT / "data" / "seceu" / "seceu_standard.json"
PREDICTIONS_OUT = REPO_ROOT / "results" / "seceu" / "seceu_emobirdv2_predictions.json"
RESULTS_OUT = REPO_ROOT / "results" / "seceu" / "seceu_emobirdv2_results.json"


def _ensure_api_key():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")


def _run_with_retries(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    attempts: int = 5,
    log_prefix: str = None,
    log_raw: bool = False,
) -> str:
    last = ""
    for i in range(1, attempts + 1):
        try:
            raw = call_openrouter(
                prompt=prompt,
                api_key=OPENROUTER_API_KEY,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if log_prefix and log_raw:
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                print(f"{log_prefix} Attempt {i}/{attempts} raw: {trunc}")
        except Exception as e:
            print(f"{log_prefix or ''} Attempt {i}/{attempts} failed: {e}")
            raw = ""
        if raw:
            last = raw
            break
    return last


def _scores_from_likert(options: List[str], likert_items: List[Dict[str, Any]]) -> List[float]:
    # Build map from emotion to rating
    rating_map: Dict[str, str] = {}
    for it in likert_items:
        em = str(it.get("emotion", "")).strip().lower()
        rt = str(it.get("rating", "")).strip().lower()
        if em: rating_map[em] = rt

    weights = []
    for opt in options:
        key = str(opt).strip().lower()
        rating = rating_map.get(key, "neutral")
        w = LIKERT_SCALE.get(rating, LIKERT_SCALE["neutral"])  # default neutral
        weights.append(w)
    weights = np.array(weights, dtype=float)

    s = float(weights.sum())
    if s <= 0:
        return [2.5, 2.5, 2.5, 2.5]
    scores = (weights / s) * 10.0
    return [float(x) for x in scores]


def _build_seceu_prompt(story: str, options: List[str]) -> str:
    """
    Mirror of the SEC-EU baseline scoring prompt from `scripts/seceu_eval.py`.
    Used only for logging/reproducibility, not for inference in this pipeline.
    """
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
1.  Deeply analyze the provided story, focusing on the main character's situation, actions, and any explicit or implicit emotional cues.
2.  For each of the four emotion options, critically assess its relevance and intensity concerning the character's experience.
3.  Assign an initial numerical score (0-10) to each emotion based on your analysis.
4.  Verify that the sum of your four scores is exactly 10. If not, carefully adjust the scores, maintaining their relative proportions as much as possible, until they sum precisely to 10.
5.  Provide ONLY the four final numerical scores, separated by spaces (e.g., 1.5 3.0 4.5 1.0). Do not add any other text or explanation before or after the scores.

Final Scores:'''
    return prompt


def main():
    parser = argparse.ArgumentParser(description="SEC-EU evaluation using EmoBIRDv2 pipeline (OpenRouter)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N items")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    parser.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--abs-max-tokens", type=int, default=ABSTRACT_MAX_TOKENS, help="Max new tokens for abstract step")
    parser.add_argument("--fac-max-tokens", type=int, default=FACTOR_MAX_TOKENS, help="Max new tokens for factors step")
    parser.add_argument("--likert-max-tokens", type=int, default=LIKERT_MAX_TOKENS, help="Max new tokens for likert step")
    parser.add_argument("--seceu-max-tokens", type=int, default=256, help="Max new tokens for SECEU numeric scoring generation")
    parser.add_argument("--log-raw", action="store_true", help="Print truncated raw model outputs")
    args = parser.parse_args()

    _ensure_api_key()

    # Load SEC-EU data
    with open(ITEMS_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    with open(STANDARD_PATH, "r", encoding="utf-8") as f:
        standard_data = json.load(f)

    standard_scores = np.array(standard_data["standard_scores"])  # shape (N,4)
    human_pattern = np.array(standard_data["human_pattern"])      # shape (N,)

    # Prepare prompts once
    tmpl_abs = load_abs_prompt()
    tmpl_fac = load_fac_prompt()
    tmpl_lik = load_lik_prompt()

    # Load eval output preamble prompt (prepended before SECEU scoring)
    def _load_eval_output_prompt() -> str:
        p = REPO_ROOT / "EmoBIRDv2" / "prompts" / "eval_output_prompt.txt"
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            # Minimal fallback if missing
            return (
                "SITUATION SHARED:\n{user_input}\n\n"
                "EMOTIONAL INSIGHTS:\nBased on psychological analysis: {emotion_insights}\n{context_info}\n"
            )
    tmpl_eval = _load_eval_output_prompt()

    # Slice items per args
    start = max(0, int(args.start))
    end = len(items) if args.limit is None else min(len(items), start + int(args.limit))
    run_items = list(enumerate(items[start:end], start=start))

    predictions: List[List[float]] = []

    for idx, item in tqdm(run_items, desc="SEC-EU via EmoBIRDv2"):
        story = item["story"]
        options = item["options"]  # 4 strings

        # Log the original SEC-EU prompt for this item (for reference) if requested
        if args.log_raw:
            try:
                seceu_prompt = _build_seceu_prompt(story, options)
                print(f"[seceu_prompt][{idx+1}]\n{seceu_prompt}")
            except Exception as e:
                print(f"[seceu_prompt][{idx+1}] failed to build: {e}")

        # 1) Abstract
        abs_user = build_abs_user(tmpl_abs, story)
        abs_raw = _run_with_retries(
            prompt=abs_user,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args["abs_max_tokens"] if isinstance(args, dict) else args.abs_max_tokens,
            log_prefix=f"[abstract][{idx+1}]",
            log_raw=args.log_raw,
        )
        abstract_text = None
        if abs_raw:
            try:
                obj = robust_json_loads(abs_raw)
                if isinstance(obj, dict):
                    abstract_text = obj.get("abstract")
            except Exception as e:
                print(f"[abstract][{idx+1}] parse failed: {e}")
        if not abstract_text:
            # Fallback to using the story directly
            abstract_text = story

        # 2) Factors from abstract
        fac_user = build_fac_user(tmpl_fac, abstract_text)
        fac_raw = _run_with_retries(
            prompt=fac_user,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args["fac_max_tokens"] if isinstance(args, dict) else args.fac_max_tokens,
            log_prefix=f"[factors][{idx+1}]",
            log_raw=args.log_raw,
        )
        factors = parse_factor_block(fac_raw) if fac_raw else []
        if not factors:
            # Minimal fallback factors if parsing failed
            factors = [
                {"name": "importance", "description": "Importance of the outcome", "possible_values": ["low", "high"]},
                {"name": "control", "description": "Perceived control over the situation", "possible_values": ["low", "high"]},
                {"name": "consequences", "description": "Severity of potential outcomes", "possible_values": ["mild", "severe"]},
            ]

        # 3) Likert for the 4 options (as the emotion list)
        lik_user = build_lik_user(tmpl_lik, story, factors, options)
        lik_raw = _run_with_retries(
            prompt=lik_user,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args["likert_max_tokens"] if isinstance(args, dict) else args.likert_max_tokens,
            log_prefix=f"[likert][{idx+1}]",
            log_raw=args.log_raw,
        )
        likert_items = parse_likert_lines(lik_raw) if lik_raw else []

        # 4) Build eval preamble with emotion insights + optional context, then append SECEU scoring prompt
        def _fmt_emotion_insights(items: List[Dict[str, Any]]) -> str:
            outs = []
            for it in items or []:
                em = str(it.get("emotion", "")).strip()
                rt = str(it.get("rating", "")).strip()
                sc = it.get("score")
                if em:
                    if sc is not None:
                        outs.append(f"{em}: {rt} ({sc})")
                    else:
                        outs.append(f"{em}: {rt}")
            return ", ".join(outs) if outs else "(no clear signals)"

        emotion_insights = _fmt_emotion_insights(likert_items)
        context_info = f"Context summary: {abstract_text}"

        preamble = tmpl_eval.format(
            user_input=story,
            emotion_insights=emotion_insights,
            context_info=context_info,
        )
        seceu_prompt = _build_seceu_prompt(story, options)
        combined_prompt = f"{preamble}\n\n{seceu_prompt}"

        # 5) Call OpenRouter to get numeric scores from the combined prompt
        def _extract_scores_with_info(text: str, fallback: List[float] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
            if fallback is None:
                fallback = [2.5, 2.5, 2.5, 2.5]
            info: Dict[str, Any] = {}
            try:
                float_pattern = re.compile(r"[-+]?\d*\.?\d+")
                raw_numbers_all = [float(x) for x in float_pattern.findall(text)]
                candidates: List[Tuple[str, List[float]]] = []
                markers = list(re.finditer(r"final\s*scores\s*:\s*", text, flags=re.IGNORECASE))
                for i, m in enumerate(markers):
                    tail = text[m.end():]
                    nums_after = [float(x) for x in float_pattern.findall(tail)]
                    if len(nums_after) >= 4:
                        label = "after_final_scores_first" if i == 0 else f"after_final_scores_{i+1}"
                        candidates.append((label, nums_after[:4]))
                if len(raw_numbers_all) >= 4:
                    candidates.append(("first_four_anywhere", raw_numbers_all[:4]))
                    candidates.append(("last_four_anywhere", raw_numbers_all[-4:]))
                if not candidates:
                    arr = np.array(fallback, dtype=float)
                    info.update({"method": "no_candidates_fallback", "raw_numbers": [], "chosen": fallback, "before_sum": float(arr.sum()), "normalized_sum": 10.0})
                    return arr, info
                # prefer first marker with non-zero sum
                chosen_label = None
                chosen_numbers: List[float] = []
                for lab, vals in candidates:
                    if lab == "after_final_scores_first" and sum(vals) > 0:
                        chosen_label, chosen_numbers = lab, vals
                        break
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
                if not chosen_numbers:
                    arr = np.array(fallback, dtype=float)
                    info.update({"method": "selection_fallback", "raw_numbers": raw_numbers_all, "chosen": fallback, "before_sum": float(arr.sum()), "normalized_sum": 10.0})
                    return arr, info
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
                info.update({"method": "exception_fallback", "raw_numbers": [], "chosen": fallback, "before_sum": float(arr.sum()), "normalized_sum": 10.0})
                return arr, info

        combined_raw = _run_with_retries(
            prompt=combined_prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args["seceu_max_tokens"] if isinstance(args, dict) else args.seceu_max_tokens,
            log_prefix=f"[seceu][{idx+1}]",
            log_raw=args.log_raw,
        )
        scores_arr, _ = _extract_scores_with_info(combined_raw)
        predictions.append([float(x) for x in scores_arr.tolist()])

    # Save predictions
    PREDICTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_OUT, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions}, f, indent=2)

    # Compute results (slice standard arrays to the processed subset)
    preds_np = np.array(predictions)
    num_processed = preds_np.shape[0]
    std_slice = standard_scores[start:start + num_processed]
    hp_slice = human_pattern[start:start + num_processed]
    distances = np.linalg.norm(preds_np - std_slice, axis=1)
    seceu_score = distances.mean()

    # Same formulas as scripts/seceu_eval.py
    population_mean = standard_data["population_mean"]
    population_std = standard_data["population_std"]
    eq_score = 15 * ((population_mean - seceu_score) / population_std) + 100
    pattern_similarity, _ = pearsonr(distances, hp_slice)

    results = {
        "seceu_score": round(float(seceu_score), 3),
        "eq_score": round(float(eq_score), 2),
        "pattern_similarity": round(float(pattern_similarity), 3),
        "model": args.model,
    }

    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✅ DONE (EmoBIRDv2 SEC-EU)")
    print(f"SECEU Score: {results['seceu_score']}")
    print(f"EQ Score: {results['eq_score']}")
    print(f"Pattern Similarity: {results['pattern_similarity']}")
    print(f"Predictions saved to: {PREDICTIONS_OUT}")
    print(f"Results saved to: {RESULTS_OUT}")


if __name__ == "__main__":
    main()
