#!/usr/bin/env python3
"""
Batch runner: EmoPatient × EmoBIRDv2 pipeline (QA-only)

- Reads scenarios from the EmoPatient dataset JSON (default: datasets/EmoPatient/scenarios.json)
- For each scenario, iterates its QA items and runs the EmoBIRDv2 multi-step pipeline on a
  single-turn input: situation = narrative + "\n\nQuestion: {q}"
- Saves per-scenario JSON results under EmoBIRDv2/eval_results/emopatient
- Also writes a consolidated dataset-style JSON (like scenarios_30.json) where each QA's
  gt is replaced by the EmoBIRD response.

Usage examples:
  export OPENROUTER_API_KEY=...  # required
  python /mnt/shared/adarsh/EmoBIRDv2/eval_scripts/run_emopatient_emobirdv2.py --limit 2 --q-limit 2 --qa-retries 2 --log-raw
  python /mnt/shared/adarsh/EmoBIRDv2/eval_scripts/run_emopatient_emobirdv2.py --q-start 3 --q-limit 2

Notes:
- QA-only: each question is treated as a single-turn conversation: Context (narrative) + Question
- Scenarios without any QA entries are skipped (no fallback to narrative-only)
- If the abstract JSON fails, we fall back to using the raw situation as the abstract
- If a QA's final output is empty, we re-run the full pipeline up to --qa-retries times
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

# Ensure repo root is importable (so `import EmoBIRDv2` works when running this file directly)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # parent of EmoBIRDv2
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
    # Rebind module-level timeout variables in abstract_generator (used by call_openrouter)
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


def load_emopatient(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expected top-level key: "scenarios"
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(scenarios, list):
        raise ValueError(f"Unexpected EmoPatient format in {path}")
    return scenarios


def make_situation_text(scn: Dict[str, Any], question: str) -> str:
    base = str(scn.get("narrative", "")).strip()
    q = (question or "").strip()
    return f"{base}\n\nQuestion: {q}" if q else base


def step_abstract(*, situation: str, model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Optional[Dict[str, Any]]:
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


def step_factors(*, abstract_text: str, model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Optional[List[Dict[str, Any]]]:
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


def step_select(*, situation: str, factors: List[Dict[str, Any]], model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Optional[List[Dict[str, str]]]:
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


def step_emotions(*, situation: str, model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Optional[List[str]]:
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


def step_likert(*, situation: str, factors: List[Dict[str, Any]], emotions: List[str], model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool) -> Optional[List[Dict[str, Any]]]:
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


def step_final_output(*, situation: str, abstract: Optional[str], selections: List[Dict[str, str]], likert_items: List[Dict[str, Any]], model: str, temperature: float, max_tokens: int) -> Optional[str]:
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

    # Abstract fallback to ensure downstream steps can proceed
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

    # Always attempt final output generation, even if likert is missing (falls back to neutral insights)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EmoBIRDv2 pipeline over EmoPatient dataset (QA-only)")
    parser.add_argument("--data", type=str, default=str(REPO_ROOT / "datasets" / "EmoPatient" / "scenarios_30.json"), help="Path to EmoPatient scenarios.json")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based) for scenarios")
    parser.add_argument("--limit", type=int, default=None, help="Process only N scenarios")
    parser.add_argument("--q-start", type=int, default=0, help="Start index within QA list")
    parser.add_argument("--q-limit", type=int, default=None, help="Process only N questions per scenario")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    parser.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--abs-max-tokens", type=int, default=ABSTRACT_MAX_TOKENS)
    parser.add_argument("--fac-max-tokens", type=int, default=FACTOR_MAX_TOKENS)
    parser.add_argument("--sel-max-tokens", type=int, default=FACTOR_MAX_TOKENS)
    parser.add_argument("--emo-max-tokens", type=int, default=EMOTION_MAX_TOKENS)
    parser.add_argument("--likert-max-tokens", type=int, default=LIKERT_MAX_TOKENS)
    parser.add_argument("--out-max-tokens", type=int, default=OUTPUT_MAX_TOKENS)
    parser.add_argument("--attempts", type=int, default=5, help="Retries per step")
    parser.add_argument("--qa-retries", type=int, default=1, help="Re-run full pipeline for a QA item if final_output is empty (additional attempts)")
    parser.add_argument("--with-emotions", dest="with_emotions", action="store_true", default=True, help="Enable emotions+likert steps (default on)")
    parser.add_argument("--no-emotions", dest="with_emotions", action="store_false", help="Disable emotions+likert steps")
    parser.add_argument("--openrouter-connect-timeout", type=int, default=None, help="Connect timeout (s) to OpenRouter")
    parser.add_argument("--openrouter-read-timeout", type=int, default=None, help="Read timeout (s) to OpenRouter")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "EmoBIRDv2" / "eval_results" / "emopatient"), help="Output directory for per-scenario JSON")
    parser.add_argument("--dataset-out", type=str, default=None, help="Path to write a consolidated dataset-style JSON (like scenarios_30.json) with gt replaced by EmoBIRD responses. If not set, a default file under output-dir will be used.")
    parser.add_argument("--resume", action="store_true", help="Skip scenarios that already have an output file")
    parser.add_argument("--log-raw", action="store_true", help="Print truncated raw model outputs to stderr")

    args = parser.parse_args()

    _ensure_api_key()
    _apply_timeout_overrides(args)

    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_emopatient(data_path)
    start = max(0, int(args.start))
    end = len(scenarios) if args.limit is None else min(len(scenarios), start + int(args.limit))
    run_scenarios = list(enumerate(scenarios[start:end], start=start))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Running EmoPatient x EmoBIRDv2 (QA-only): {len(run_scenarios)} scenarios | run_id={run_id}", file=sys.stderr)

    # Prepare consolidated dataset-style accumulator
    consolidated: Dict[str, Any] = {"scenarios": []}

    for idx, scn in tqdm(run_scenarios, total=len(run_scenarios), desc="Scenarios", unit="scn"):
        sid = scn.get("id") or f"S{idx+1}"
        title = scn.get("title", "").strip()
        qa_list = scn.get("qa") if isinstance(scn.get("qa"), list) else []

        if not qa_list:
            print(f"[skip] {sid} has no QA entries; skipping.", file=sys.stderr)
            continue

        out_path = out_dir / f"{sid}_emobirdv2_{run_id}.json"
        if args.resume:
            # If any file for this scenario already exists (regardless of run_id), skip
            existing = list(out_dir.glob(f"{sid}_emobirdv2_*.json"))
            if existing:
                print(f"[skip] {sid} already has results (resume)", file=sys.stderr)
                continue

        record: Dict[str, Any] = {
            "scenario_id": sid,
            "title": title,
            "run_id": run_id,
            "model": args.model,
            "temperature": args.temperature,
            "attempts": args.attempts,
            "items": [],
        }

        q_start = max(0, int(getattr(args, "q_start", 0)))
        q_end = len(qa_list) if args.q_limit is None else min(len(qa_list), q_start + int(args.q_limit))
        sub_qa = list(enumerate(qa_list[q_start:q_end], start=q_start))

        # Build consolidated scenario record (copy top-level fields except qa for now)
        scn_out: Dict[str, Any] = {}
        for k, v in scn.items():
            if k != "qa":
                scn_out[k] = v
        scn_out["qa"] = []

        for _pos, (j, qa) in enumerate(tqdm(sub_qa, total=len(sub_qa), desc=f"{sid} QA", unit="qa", leave=False), start=1):
            q_text = str(qa.get("q", "")).strip()
            situation = make_situation_text(scn, q_text)

            max_qa_retries = max(0, int(getattr(args, "qa_retries", 1)))
            last_p: Dict[str, Any] | None = None
            resp_text: str = ""

            for rtry in range(max_qa_retries + 1):
                label = f"Q{_pos}/{len(sub_qa)}"
                if rtry == 0:
                    print(f"[{sid}] {label}: running...", file=sys.stderr)
                else:
                    print(f"[{sid}] {label}: empty final output, retry {rtry}/{max_qa_retries}...", file=sys.stderr)

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
                last_p = p
                resp_text = (p.get("final_output") or "").strip()
                if resp_text:
                    break
                # Gentle backoff to avoid hammering the API
                time.sleep(0.3)

            # Record per-item (store retries used)
            item = {
                "index": j + 1,
                "question": q_text,
                "situation": situation,
                "pipeline": last_p or {},
                "qa_retries_used": rtry,
            }
            record["items"].append(item)

            # Consolidated QA entry: copy all fields from input QA, but replace gt with EmoBIRD response
            qa_out = dict(qa)
            qa_out["gt"] = resp_text
            scn_out["qa"].append(qa_out)

        # Append consolidated scenario
        consolidated["scenarios"].append(scn_out)

        # Write per-scenario output
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"[done] {sid} -> {out_path}", file=sys.stderr)

        # Gentle pacing to avoid rate limits across scenarios
        time.sleep(0.5)

    # Write consolidated dataset-style output last
    ds_out_path = Path(args.dataset_out) if args.dataset_out else (out_dir / f"emopatient_emobirdv2_{run_id}.json")
    with ds_out_path.open("w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
    print(f"[done] consolidated -> {ds_out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
