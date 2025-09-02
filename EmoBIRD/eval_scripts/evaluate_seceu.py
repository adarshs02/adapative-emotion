"""
SECEU evaluation script integrated with EmoBIRD's vLLM or OpenRouter backend and logger.

Features
- Loads SECEU items and standard scores.
- Generates model outputs using VLLMWrapper or OpenRouterWrapper (text-only response; no JSON enforced).
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
import os
import re
import sys

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

# Local imports from EmoBIRD (support module, repo-root script, or EmoBIRD-dir script execution)
try:
    # Package execution: python -m EmoBIRD.eval_scripts.evaluate_seceu
    from .config import EmobirdConfig
    from .vllm_wrapper import VLLMWrapper
    from .openrouter_wrapper import OpenRouterWrapper
    from .emobird_poc import Emobird
    from .logger import get_logger
except Exception:
    # Running as a script from the repository root:
    #   python EmoBIRD/eval_scripts/evaluate_seceu.py
    try:
        from EmoBIRD.config import EmobirdConfig
        from EmoBIRD.vllm_wrapper import VLLMWrapper
        from EmoBIRD.openrouter_wrapper import OpenRouterWrapper
        from EmoBIRD.emobird_poc import Emobird
        from EmoBIRD.logger import get_logger
    except Exception:
        # Running as a script from inside the EmoBIRD directory:
        #   python eval_scripts/evaluate_seceu.py
        # Ensure EmoBIRD dir is importable then try plain imports.
        CURRENT_DIR = Path(__file__).resolve().parent
        EMOBIRD_DIR = CURRENT_DIR.parent
        if str(EMOBIRD_DIR) not in sys.path:
            sys.path.insert(0, str(EMOBIRD_DIR))
        from config import EmobirdConfig
        from vllm_wrapper import VLLMWrapper
        from openrouter_wrapper import OpenRouterWrapper
        from emobird_poc import Emobird
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
    return f'''Read the story and score each of the four emotion options.

Rules:
- Each score is between 0 and 10 (inclusive).
- The four scores must sum to exactly 10.

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
    parser.add_argument("--env-file", type=str, default="", help="Path to a bash/.env file with KEY=VALUE or 'export KEY=VALUE' lines to pre-load (e.g., OPENROUTER_API_KEY)")
    parser.add_argument("--model", type=str, default="", help="Override model name (updates config.llm_model_name) e.g., meta-llama/Llama-3.1-8B-Instruct or an OpenRouter model ID")
    # OpenRouter-specific overrides
    parser.add_argument("--api-key", type=str, default="", help="OpenRouter API key (overrides env)")
    parser.add_argument("--base-url", type=str, default="", help="OpenRouter base URL (overrides env)")
    parser.add_argument("--provider", type=str, default="", help="Optional provider routing hint (overrides env OPENROUTER_PROVIDER)")
    parser.add_argument("--openrouter-timeout", type=int, default=None, help="Request timeout in seconds for OpenRouter (overrides env)")
    parser.add_argument("--openrouter-max-retries", type=int, default=None, help="Max retries for OpenRouter requests (overrides env)")
    parser.add_argument("--output-preds", type=str, default="")
    parser.add_argument("--output-results", type=str, default="")
    parser.add_argument("--backend", type=str, choices=["vllm", "openrouter"], default="", help="Override backend; else use config.llm_backend/env")
    parser.add_argument("--seceu-max-tokens", type=int, default=256, help="Max tokens for SECEU per-call generation (previously 24)")
    parser.add_argument(
        "--seceu-stop-policy",
        type=str,
        choices=["none", "newline", "auto"],
        default="none",
        help=(
            "Stop policy for SECEU generation: 'none' sends no stop tokens; 'newline' uses \\n; "
            "'auto' uses none for OpenRouter and newline for vLLM. Default: none."
        ),
    )
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

    # Optionally load env vars from a bash/.env file BEFORE creating config
    if getattr(args, "env_file", None):
        env_path = Path(args.env_file).expanduser()
        if env_path.exists():
            try:
                for raw in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # Strip surrounding single or double quotes
                    if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ('"', "'")):
                        v = v[1:-1]
                    # Only set if not already present in process env
                    if k and (k not in os.environ):
                        os.environ[k] = v
            except Exception as e:
                print(f"⚠️ Failed to load env file {env_path}: {e}")
        else:
            print(f"⚠️ Env file not found: {env_path}")

    # Config and model
    config = EmobirdConfig()
    if args.temperature is not None:
        config.update_config(temperature=float(args.temperature))
    # CLI model override (applies to both vLLM and OpenRouter backends)
    if getattr(args, "model", None):
        model_name = (args.model or "").strip()
        if model_name:
            try:
                config.update_config(llm_model_name=model_name)
            except Exception:
                config.llm_model_name = model_name
    # OpenRouter CLI overrides
    if getattr(args, "api_key", None):
        key = (args.api_key or "").strip()
        if key:
            try:
                config.update_config(openrouter_api_key=key)
            except Exception:
                config.openrouter_api_key = key
    if getattr(args, "base_url", None):
        bu = (args.base_url or "").strip()
        if bu:
            try:
                config.update_config(openrouter_base_url=bu)
            except Exception:
                config.openrouter_base_url = bu
    if getattr(args, "provider", None):
        prov = (args.provider or "").strip()
        if prov:
            try:
                config.update_config(openrouter_provider=prov)
            except Exception:
                config.openrouter_provider = prov
    if getattr(args, "openrouter_timeout", None) is not None:
        try:
            config.update_config(openrouter_timeout=int(args.openrouter_timeout))
        except Exception:
            config.openrouter_timeout = int(args.openrouter_timeout)
    if getattr(args, "openrouter_max_retries", None) is not None:
        try:
            config.update_config(openrouter_max_retries=int(args.openrouter_max_retries))
        except Exception:
            config.openrouter_max_retries = int(args.openrouter_max_retries)
    backend = (args.backend or "").strip().lower() or config.llm_backend
    # Persist chosen backend in config for downstream logging
    try:
        config.update_config(llm_backend=backend)
    except Exception:
        config.llm_backend = backend
    # Fail fast if OpenRouter selected but API key is missing
    if backend == "openrouter" and not getattr(config, "openrouter_api_key", None):
        raise RuntimeError(
            "OpenRouter backend selected but API key is missing. Set OPENROUTER_API_KEY in the environment, pass --api-key, or load with --env-file."
        )

    # Mirror critical overrides into environment so Emobird() picks them up in its own EmobirdConfig
    os.environ.setdefault("EMOBIRD_MODEL", str(config.llm_model_name))
    os.environ["EMOBIRD_LLM_BACKEND"] = backend
    if getattr(config, "openrouter_api_key", None):
        os.environ["OPENROUTER_API_KEY"] = str(config.openrouter_api_key)
    if getattr(config, "openrouter_base_url", None):
        os.environ["OPENROUTER_BASE_URL"] = str(config.openrouter_base_url)
    if getattr(config, "openrouter_provider", None):
        os.environ["OPENROUTER_PROVIDER"] = str(config.openrouter_provider)
    if getattr(config, "openrouter_timeout", None) is not None:
        os.environ["OPENROUTER_TIMEOUT"] = str(int(config.openrouter_timeout))
    if getattr(config, "openrouter_max_retries", None) is not None:
        os.environ["OPENROUTER_MAX_RETRIES"] = str(int(config.openrouter_max_retries))

    # Instantiate EmoBIRD pipeline and reuse its wrapper for SECEU generation
    eb = Emobird()
    wrapper = eb.vllm_wrapper

    # Important: do NOT globally override wrapper sampling here, as it affects all
    # EmoBIRD pipeline stages (e.g., factor generation). We enforce deterministic
    # numeric output only in the SECEU call below via per-call overrides.

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

    # Align standards and human pattern to the processed subset length
    subset_n = len(item_list)
    try:
        if subset_n != len(standard_scores):
            standard_scores = np.array(standard_scores[:subset_n], dtype=float)
    except Exception:
        standard_scores = np.array(standard_scores[:subset_n], dtype=float)
    try:
        if subset_n != len(human_pattern):
            human_pattern = np.array(human_pattern[:subset_n], dtype=float)
    except Exception:
        human_pattern = np.array(human_pattern[:subset_n], dtype=float)

    logger = get_logger()

    # Run for N iterations and average distances pattern across iterations
    all_iter_distances: List[np.ndarray] = []
    all_iter_preds: List[List[List[float]]] = []

    for it in range(1, args.iterations + 1):
        iter_preds: List[List[float]] = []
        distances_iter: List[float] = []
        # Stop policy: control per-call stop behavior for numeric output stability.
        policy = (getattr(args, "seceu_stop_policy", "none") or "none").lower()
        if policy == "auto":
            eff_stop = [] if backend == "openrouter" else ["\n"]
            stop_policy_meta = "none" if backend == "openrouter" else "\\n"
        elif policy == "newline":
            eff_stop = ["\n"]
            stop_policy_meta = "\\n"
        else:  # "none"
            eff_stop = []
            stop_policy_meta = "none"
        seceu_max_tokens = int(getattr(args, "seceu_max_tokens", 128) or 128)

        for idx, item in enumerate(tqdm(item_list, desc=f"Processing items (iter {it})")):
            story = item["story"]
            options = item["options"]
            prompt = build_prompt(story, options)

            # Run full EmoBIRD pipeline for logging per stage
            try:
                result = eb.analyze_emotion(story)
                # Persist a structured log of the pipeline output for this story
                try:
                    from EmoBIRD.logger import get_logger as _pkg_get_logger  # ensure same global
                    _pkg_get_logger().log_analysis_result(story, result)
                except Exception:
                    logger.log_analysis_result(story, result)
            except Exception as e:
                logger.log_error(
                    component="seceu",
                    error_type="emobird_pipeline_failure",
                    error_message=str(e),
                    context={"item_index": idx, "iteration": it},
                )

            # Generate model output (deterministic, single-line numeric) using EmoBIRD's wrapper
            response = wrapper.generate(
                prompt,
                component="seceu",
                interaction_type="seceu_prompt",
                stop=eff_stop,
                max_tokens_override=seceu_max_tokens,
                temperature_override=0.0,
            )

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
    # Ensure equal length and enough points for correlation
    if len(distances) != len(human_pattern):
        human_pattern = np.array(human_pattern[: len(distances)], dtype=float)
    pattern_similarity = float(pearsonr(distances, human_pattern)[0]) if len(distances) >= 2 else float("nan")

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
            "backend": getattr(getattr(eb, "config", config), "llm_backend", config.llm_backend),
            "model_name": getattr(getattr(eb, "config", config), "llm_model_name", config.llm_model_name),
            # Actual overrides used for SECEU generation
            "sampling_temperature": 0.0,
            "sampling_top_p": (float(getattr(wrapper, "_top_p", 1.0)) if backend == "openrouter" else None),
            "seceu_max_tokens": seceu_max_tokens,
            "stop_policy": stop_policy_meta,
        },
    }
    with results_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ SECEU evaluation complete. SECEU Score: {results['seceu_score']}  |  EQ Score: {results['eq_score']}")
    print(f"📄 Predictions saved to: {preds_out}")
    print(f"📄 Results saved to: {results_out}")


if __name__ == "__main__":
    main()
