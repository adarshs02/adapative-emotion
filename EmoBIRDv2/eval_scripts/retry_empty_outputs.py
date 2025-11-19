#!/usr/bin/env python3
"""
Retry script: Fix empty outputs in EmoPatient x EmoBIRDv2 results

Reads a consolidated results JSON, identifies QA pairs with empty 'gt' fields,
re-runs the pipeline for those items, and updates the file with new results.

Usage:
  export OPENROUTER_API_KEY=...
  python EmoBIRDv2/eval_scripts/retry_empty_outputs.py \
    --input EmoBIRDv2/eval_results/emopatient/emopatient_emobirdv2_20251118_212610.json \
    --workers 10 \
    --qa-retries 3
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List
from tqdm import tqdm

# Ensure repo root is importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# EmoBIRDv2 imports
from EmoBIRDv2.utils.constants import (
    OPENROUTER_API_KEY,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    ABSTRACT_MAX_TOKENS,
    FACTOR_MAX_TOKENS,
    EMOTION_MAX_TOKENS,
    LIKERT_MAX_TOKENS,
    OUTPUT_MAX_TOKENS,
)
from EmoBIRDv2.utils.utils import robust_json_loads
import EmoBIRDv2.scripts.abstract_generator as AG
import EmoBIRDv2.scripts.factor_generator as FG
import EmoBIRDv2.scripts.factor_value_selector as FVS
import EmoBIRDv2.scripts.emotion_generator as EG
import EmoBIRDv2.scripts.likert_matcher as LM
import EmoBIRDv2.scripts.final_output_generator as FOG


def _ensure_api_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")


def _apply_timeout_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "openrouter_connect_timeout", None) is not None:
        try:
            AG.OPENROUTER_CONNECT_TIMEOUT = int(args.openrouter_connect_timeout)
        except Exception:
            pass
    if getattr(args, "openrouter_read_timeout", None) is not None:
        try:
            AG.OPENROUTER_READ_TIMEOUT = int(args.openrouter_read_timeout)
        except Exception:
            pass


def make_situation_text(narrative: str, question: str) -> str:
    base = narrative.strip()
    q = question.strip()
    return f"{base}\n\nQuestion: {q}" if q else base


def step_abstract(*, situation: str, model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Any:
    tpl = AG.load_prompt()
    prompt = AG.build_user_prompt(tpl, situation)
    raw = ""
    for i in range(1, attempts + 1):
        try:
            raw = AG.call_openrouter(prompt, OPENROUTER_API_KEY, model, temperature, max_tokens)
            if log_raw and raw:
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                print(f"[abstract] raw: {trunc}", file=sys.stderr)
        except Exception as e:
            print(f"[abstract] attempt {i} failed: {e}", file=sys.stderr)
            raw = ""
        if not raw:
            continue
        try:
            obj = robust_json_loads(raw)
            if isinstance(obj, dict) and obj.get("abstract"):
                return obj
        except Exception as e:
            print(f"[abstract] JSON parse failed: {e}", file=sys.stderr)
    return None


def step_factors(*, abstract_text: str, model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Any:
    tpl = FG.load_prompt()
    prompt = FG.build_user_prompt(tpl, abstract_text)
    raw = ""
    for i in range(1, attempts + 1):
        try:
            raw = AG.call_openrouter(prompt, OPENROUTER_API_KEY, model, temperature, max_tokens)
            if log_raw and raw:
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                print(f"[factors] raw: {trunc}", file=sys.stderr)
        except Exception as e:
            print(f"[factors] attempt {i} failed: {e}", file=sys.stderr)
            raw = ""
        if not raw:
            continue
        parsed = FG.parse_factor_block(raw)
        if parsed:
            return parsed
    return None


def step_select(*, situation: str, factors: List[Dict[str, Any]], model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Any:
    tpl = FVS.load_prompt()
    prompt = FVS.build_user_prompt(tpl, situation, factors)
    raw = ""
    for i in range(1, attempts + 1):
        try:
            raw = AG.call_openrouter(prompt, OPENROUTER_API_KEY, model, temperature, max_tokens)
            if log_raw and raw:
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                print(f"[select] raw: {trunc}", file=sys.stderr)
        except Exception as e:
            print(f"[select] attempt {i} failed: {e}", file=sys.stderr)
            raw = ""
        if not raw:
            continue
        parsed = FVS.parse_selection_block(raw)
        if parsed:
            return parsed
    return None


def step_emotions(*, situation: str, model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Any:
    tpl = EG.load_prompt()
    prompt = EG.build_user_prompt(tpl, situation)
    raw = ""
    for i in range(1, attempts + 1):
        try:
            raw = AG.call_openrouter(prompt, OPENROUTER_API_KEY, model, temperature, max_tokens)
            if log_raw and raw:
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                print(f"[emotions] raw: {trunc}", file=sys.stderr)
        except Exception as e:
            print(f"[emotions] attempt {i} failed: {e}", file=sys.stderr)
            raw = ""
        if not raw:
            continue
        parsed = EG.parse_emotion_lines(raw)
        if parsed and len(parsed) >= 3:
            return parsed
    return None


def step_likert(*, situation: str, factors: List[Dict[str, Any]], emotions: List[str], model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Any:
    tpl = LM.load_prompt()
    prompt = LM.build_user_prompt(tpl, situation, factors, emotions)
    raw = ""
    for i in range(1, attempts + 1):
        try:
            raw = AG.call_openrouter(prompt, OPENROUTER_API_KEY, model, temperature, max_tokens)
            if log_raw and raw:
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                print(f"[likert] raw: {trunc}", file=sys.stderr)
        except Exception as e:
            print(f"[likert] attempt {i} failed: {e}", file=sys.stderr)
            raw = ""
        if not raw:
            continue
        parsed = LM.parse_likert_lines(raw)
        if parsed:
            return parsed
    return None


def step_final_output(*, situation: str, abstract: Any, selections: List[Dict[str, str]], likert_items: List[Dict[str, Any]], model: str, temperature: float, max_tokens: int) -> Any:
    try:
        return FOG.generate_final_output(
            situation=situation,
            abstract=abstract,
            selections=selections,
            likert_items=likert_items,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"[final_output] failed: {e}", file=sys.stderr)
        return None


def run_pipeline_for_text(
    *,
    situation: str,
    model: str,
    temperature: float,
    abs_max_tokens: int,
    fac_max_tokens: int,
    sel_max_tokens: int,
    emo_max_tokens: int,
    likert_max_tokens: int,
    out_max_tokens: int,
    attempts: int,
    with_emotions: bool,
    log_raw: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "abstract": None,
        "factors": None,
        "selections": None,
        "emotions": None,
        "likert": None,
        "final_output": None,
        "status": "ok",
        "failed_at": None,
    }

    abs_obj = step_abstract(
        situation=situation,
        model=model,
        temperature=temperature,
        max_tokens=abs_max_tokens,
        attempts=attempts,
        log_raw=log_raw,
    )

    abstract = None
    if not abs_obj:
        abstract = situation
        result["abstract_fallback"] = True
    else:
        abstract = str(abs_obj.get("abstract", "")).strip()
    result["abstract"] = abstract

    factors = step_factors(
        abstract_text=abstract,
        model=model,
        temperature=temperature,
        max_tokens=fac_max_tokens,
        attempts=attempts,
        log_raw=log_raw,
    )
    if not factors:
        result["status"] = "error"
        result["failed_at"] = "factors"
        return result
    result["factors"] = factors

    selections = step_select(
        situation=situation,
        factors=factors,
        model=model,
        temperature=temperature,
        max_tokens=sel_max_tokens,
        attempts=attempts,
        log_raw=log_raw,
    )
    if not selections:
        result["status"] = "error"
        result["failed_at"] = "select"
        return result
    result["selections"] = selections

    emotions = None
    if with_emotions:
        emotions = step_emotions(
            situation=situation,
            model=model,
            temperature=temperature,
            max_tokens=emo_max_tokens,
            attempts=attempts,
            log_raw=log_raw,
        )
        if emotions:
            result["emotions"] = emotions

    likert = None
    if emotions:
        likert = step_likert(
            situation=situation,
            factors=factors,
            emotions=emotions,
            model=model,
            temperature=temperature,
            max_tokens=likert_max_tokens,
            attempts=attempts,
            log_raw=log_raw,
        )
        if likert:
            result["likert"] = likert

    final_text = step_final_output(
        situation=situation,
        abstract=abstract,
        selections=selections,
        likert_items=likert or [],
        model=model,
        temperature=temperature,
        max_tokens=out_max_tokens,
    )
    if final_text:
        result["final_output"] = final_text

    return result


def process_single_empty_qa(task: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single empty QA pair."""
    scn_idx = task["scn_idx"]
    qa_idx = task["qa_idx"]
    situation = task["situation"]
    args = task["args"]
    sid = task["sid"]
    
    max_qa_retries = max(0, int(args.qa_retries))
    resp_text = ""
    
    for rtry in range(max_qa_retries + 1):
        if rtry == 0:
            print(f"[{sid}] QA {qa_idx+1}: retrying empty output...", file=sys.stderr)
        else:
            print(f"[{sid}] QA {qa_idx+1}: still empty, retry {rtry}/{max_qa_retries}...", file=sys.stderr)
        
        p = run_pipeline_for_text(
            situation=situation,
            model=args.model,
            temperature=args.temperature,
            abs_max_tokens=args.abs_max_tokens,
            fac_max_tokens=args.fac_max_tokens,
            sel_max_tokens=args.sel_max_tokens,
            emo_max_tokens=args.emo_max_tokens,
            likert_max_tokens=args.likert_max_tokens,
            out_max_tokens=args.out_max_tokens,
            attempts=args.attempts,
            with_emotions=args.with_emotions,
            log_raw=args.log_raw,
        )
        resp_text = (p.get("final_output") or "").strip()
        if resp_text:
            break
        time.sleep(0.3)
    
    return {
        "scn_idx": scn_idx,
        "qa_idx": qa_idx,
        "new_gt": resp_text,
        "success": bool(resp_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry empty outputs in EmoPatient x EmoBIRDv2 results")
    parser.add_argument("--input", type=str, required=True, help="Path to consolidated results JSON")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: overwrites input)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    parser.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--abs-max-tokens", type=int, default=ABSTRACT_MAX_TOKENS)
    parser.add_argument("--fac-max-tokens", type=int, default=FACTOR_MAX_TOKENS)
    parser.add_argument("--sel-max-tokens", type=int, default=FACTOR_MAX_TOKENS)
    parser.add_argument("--emo-max-tokens", type=int, default=EMOTION_MAX_TOKENS)
    parser.add_argument("--likert-max-tokens", type=int, default=LIKERT_MAX_TOKENS)
    parser.add_argument("--out-max-tokens", type=int, default=OUTPUT_MAX_TOKENS)
    parser.add_argument("--attempts", type=int, default=10, help="Retries per step (default: 10)")
    parser.add_argument("--qa-retries", type=int, default=2, help="Full pipeline retries for empty outputs (default: 2)")
    parser.add_argument("--with-emotions", dest="with_emotions", action="store_true", default=True)
    parser.add_argument("--no-emotions", dest="with_emotions", action="store_false")
    parser.add_argument("--openrouter-connect-timeout", type=int, default=None)
    parser.add_argument("--openrouter-read-timeout", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--log-raw", action="store_true", help="Print raw model outputs")
    
    args = parser.parse_args()
    
    _ensure_api_key()
    _apply_timeout_overrides(args)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else input_path
    
    # Load data
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    scenarios = data.get("scenarios", [])
    
    # Find all empty QA pairs
    empty_tasks = []
    for scn_idx, scn in enumerate(scenarios):
        sid = scn.get("id", f"S{scn_idx+1}")
        narrative = scn.get("narrative", "")
        qa_list = scn.get("qa", [])
        
        for qa_idx, qa in enumerate(qa_list):
            gt = str(qa.get("gt", "")).strip()
            if not gt:
                question = qa.get("q", "")
                situation = make_situation_text(narrative, question)
                empty_tasks.append({
                    "scn_idx": scn_idx,
                    "qa_idx": qa_idx,
                    "situation": situation,
                    "sid": sid,
                    "args": args,
                })
    
    if not empty_tasks:
        print("No empty outputs found. Nothing to retry.", file=sys.stderr)
        return
    
    print(f"Found {len(empty_tasks)} empty QA outputs to retry.", file=sys.stderr)
    
    # Process empty QAs
    workers = max(1, int(args.workers))
    results_dict = {}
    
    if workers == 1:
        # Sequential
        for task in tqdm(empty_tasks, desc="Retrying", unit="qa"):
            result = process_single_empty_qa(task)
            key = (result["scn_idx"], result["qa_idx"])
            results_dict[key] = result
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(process_single_empty_qa, task): task for task in empty_tasks}
            
            with tqdm(total=len(empty_tasks), desc="Retrying", unit="qa") as pbar:
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        key = (result["scn_idx"], result["qa_idx"])
                        results_dict[key] = result
                    except Exception as exc:
                        task = future_to_task[future]
                        print(f"Task {task['sid']} QA {task['qa_idx']+1} failed: {exc}", file=sys.stderr)
                    finally:
                        pbar.update(1)
    
    # Update data with new results
    updated_count = 0
    for (scn_idx, qa_idx), result in results_dict.items():
        if result["success"]:
            scenarios[scn_idx]["qa"][qa_idx]["gt"] = result["new_gt"]
            updated_count += 1
    
    # Save updated data
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[done] Updated {updated_count}/{len(empty_tasks)} empty outputs.", file=sys.stderr)
    print(f"[done] Saved to: {output_path}", file=sys.stderr)
    
    still_empty = len(empty_tasks) - updated_count
    if still_empty > 0:
        print(f"[warning] {still_empty} outputs still empty after retries.", file=sys.stderr)


if __name__ == "__main__":
    main()
