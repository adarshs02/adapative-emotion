#!/usr/bin/env python3
"""
EmoBench baseline (EA and EU) via OpenRouter API only (no EmoBIRD pipeline).

- Loads `datasets/EmoBench/data/EA.jsonl` and `EU.jsonl`
- Prompts the model directly with strict JSON instructions
- Parses outputs, coerces to valid choices when needed
- Computes accuracy metrics
- Saves to `results/emobench/baseline_ea_results.json` and `baseline_eu_results.json`

Logging:
- Raw outputs are printed to stderr when `--log-raw`/`--print-raw` are set.
- Additionally, use `--log-file` to append raw outputs to a persistent file.
- If `--log-file` is not specified, a default file will be used: `results/emobench/baseline_<run_id>.log`.
  - Per-attempt raw outputs are written when `--log-raw` is used (truncated unless `--raw-full`).
  - Final per-item raw predictions are written when `--print-raw` is used.

Requirements:
- OPENROUTER_API_KEY in env
- Defaults to Meta-Llama 3.1 8B Instruct (override with --model or OPENROUTER_MODEL/REMOTE_MODEL env)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import time
import random

# Ensure repo root is importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse OpenRouter utilities from EmoBIRDv2
from EmoBIRDv2.utils.constants import OPENROUTER_API_KEY, MODEL_TEMPERATURE
import EmoBIRDv2.scripts.abstract_generator as AG  # for call_openrouter
from EmoBIRDv2.utils.utils import robust_json_loads

# Data paths
EMOBENCH_DIR = REPO_ROOT / "datasets" / "EmoBench" / "data"
EA_PATH = EMOBENCH_DIR / "EA.jsonl"
EU_PATH = EMOBENCH_DIR / "EU.jsonl"

OUT_DIR = REPO_ROOT / "results" / "emobench"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EA_OUT = OUT_DIR / "baseline_ea_results.json"
EU_OUT = OUT_DIR / "baseline_eu_results.json"


def _append_log(log_file: Optional[Path], text: str) -> None:
    """Append a single line (or block) to the given log file, creating parents as needed."""
    if not log_file:
        return
    try:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except Exception:
        # Fail silently to avoid interrupting eval
        pass


def _ensure_api_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _run_with_retries(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    attempts: int = 5,
    log_prefix: Optional[str] = None,
    log_raw: bool = False,
    log_full: bool = False,
    log_file: Optional[Path] = None,
    *,
    initial_wait: float = 1.0,
    backoff: float = 1.5,
    jitter: float = 0.5,
) -> str:
    last = ""
    wait_s = max(0.0, float(initial_wait))
    for i in range(1, attempts + 1):
        try:
            raw = AG.call_openrouter(
                prompt=prompt,
                api_key=OPENROUTER_API_KEY,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                system=(
                    "You are a strict JSON responder. Output exactly one JSON object and nothing else."
                ),
                extra_payload={"stream": False},
            )
            if log_prefix and log_raw and raw is not None:
                if log_full:
                    msg = f"{log_prefix} Attempt {i}/{attempts} raw: {raw}"
                else:
                    trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                    msg = f"{log_prefix} Attempt {i}/{attempts} raw: {trunc}"
                print(msg, file=sys.stderr)
                _append_log(log_file, msg)
        except Exception as e:
            msg = f"{log_prefix or ''} Attempt {i}/{attempts} failed: {e}"
            print(msg, file=sys.stderr)
            _append_log(log_file, msg)
            raw = ""
        if raw:
            last = raw
            break
        # If no output and more attempts remain, wait with backoff and jitter
        if i < attempts:
            sleep_extra = random.uniform(0.0, jitter) if jitter and jitter > 0 else 0.0
            sleep_s = max(0.0, wait_s + sleep_extra)
            wait_msg = f"{log_prefix or ''} Waiting {sleep_s:.2f}s before retry {i+1}/{attempts}"
            print(wait_msg, file=sys.stderr)
            _append_log(log_file, wait_msg)
            try:
                time.sleep(sleep_s)
            except Exception:
                pass
            try:
                if backoff and backoff > 0:
                    wait_s *= float(backoff)
            except Exception:
                pass
    return last


# ---------- Prompt builders (baseline, no pipeline context) ----------

def build_ea_prompt_baseline(*, scenario: str, subject: str, choices: List[str]) -> str:
    choice_lines = [f"{chr(97+i)}) {c}" for i, c in enumerate(choices)]
    system = (
        "Select exactly one option out of the given choices."
    )
    user = f"""
Subject: {subject}
Scenario: {scenario}

Choices (select exactly one):
{os.linesep.join(choice_lines)}

Return STRICT JSON only with keys:
{{
  "choice_letter": "a|b|c|d",
  "choice_text": "<exact text copied from one of the choices>"
}}
Rules:
- The choice_text must be exactly one of the provided choices (verbatim match).
- Do not include any text before or after the JSON object.
- If unsure, pick the safest, most empathetic option.
"""
    return f"{system}\n\n{user}"


def build_eu_prompt_baseline(*, scenario: str, subject: str, emotion_choices: List[str], cause_choices: List[str]) -> str:
    emo_block = "- " + "\n- ".join(str(c) for c in emotion_choices)
    cause_block = "- " + "\n- ".join(str(c) for c in cause_choices)
    user = f"""
You are EmoBIRD. Read the scenario and choose exactly ONE emotion and ONE cause from the provided choices.
Return STRICT JSON only (no markdown, no code fences) with keys 'emo_label' and 'cause_label'.
The values MUST be verbatim copies of one option from the respective choices. Do not invent new options.

Subject: {subject}
Scenario: {scenario}

Emotion choices:
{emo_block}

Cause choices:
{cause_block}

Output format strictly: {{"emo_label": "<one of emotion choices>", "cause_label": "<one of cause choices>"}}
"""
    return user


# ---------- Helpers ----------

def _coerce_choice_to_valid(letter: str, text: str, choices: List[str]) -> str:
    letter = (letter or "").strip().lower()
    if letter in ["a", "b", "c", "d"]:
        idx = ord(letter) - ord("a")
        if 0 <= idx < len(choices):
            return choices[idx]
    txt = (text or "").strip()
    for c in choices:
        if txt.lower() == c.strip().lower():
            return c
    for c in choices:
        cl = c.strip().lower()
        if cl in txt.lower() or txt.lower() in cl:
            return c
    return choices[0] if choices else ""


def _coerce_to_choice(val: str, choices: List[str]) -> str:
    v = (val or "").strip()
    if v in choices:
        return v
    lower_map = {c.lower(): c for c in choices}
    if v.lower() in lower_map:
        return lower_map[v.lower()]
    try:
        import difflib
        m = difflib.get_close_matches(v, choices, n=1, cutoff=0.0)
        return m[0] if m else (choices[0] if choices else "")
    except Exception:
        return choices[0] if choices else ""


# ---------- Evaluations ----------

def evaluate_ea(
    *,
    data: List[Dict[str, Any]],
    model: str,
    temperature: float,
    pred_max_tokens: int,
    attempts: int,
    log_raw: bool,
    log_full: bool,
    print_raw: bool,
    log_file: Optional[Path],
    retry_initial_wait: float,
    retry_backoff: float,
    retry_jitter: float,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    correct = 0

    for idx, item in enumerate(tqdm(data, desc="EA (baseline)", unit="item")):
        scenario = str(item.get("scenario", "")).strip()
        subject = str(item.get("subject", "user")).strip()
        choices = list(item.get("choices", []))

        prompt = build_ea_prompt_baseline(scenario=scenario, subject=subject, choices=choices)
        raw = _run_with_retries(
            prompt,
            model,
            temperature=0.0,
            max_tokens=pred_max_tokens,
            attempts=attempts,
            log_prefix=f"[ea][{idx+1}]",
            log_raw=log_raw,
            log_full=log_full,
            log_file=log_file,
            initial_wait=retry_initial_wait,
            backoff=retry_backoff,
            jitter=retry_jitter,
        )
        if print_raw and raw:
            print(f"[ea][{idx+1}] raw prediction:\n{raw}\n", file=sys.stderr)
            _append_log(log_file, f"[ea][{idx+1}] raw prediction:\n{raw}\n")

        choice_letter = ""
        choice_text = ""
        try:
            obj = robust_json_loads(raw)
            if isinstance(obj, dict):
                choice_letter = str(obj.get("choice_letter", "")).strip()
                choice_text = str(obj.get("choice_text", "")).strip()
        except Exception:
            pass
        pred_choice = _coerce_choice_to_valid(choice_letter, choice_text, choices)
        gt_choice = str(item.get("label", "")).strip()
        is_correct = pred_choice.strip().lower() == gt_choice.strip().lower()
        if is_correct:
            correct += 1

        res: Dict[str, Any] = {
            "idx": idx,
            "scenario": scenario,
            "subject": subject,
            "choices": choices,
            "raw_model_output": raw,
            "pred_choice_text": pred_choice,
            "gt_choice_text": gt_choice,
            "is_correct": is_correct,
        }
        results.append(res)

    accuracy = (correct / len(data)) if data else 0.0
    return {"items": results, "accuracy": accuracy}


def evaluate_eu(
    *,
    data: List[Dict[str, Any]],
    model: str,
    temperature: float,
    pred_max_tokens: int,
    attempts: int,
    log_raw: bool,
    log_full: bool,
    print_raw: bool,
    log_file: Optional[Path],
    retry_initial_wait: float,
    retry_backoff: float,
    retry_jitter: float,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    correct = 0
    emo_correct = 0

    for idx, item in enumerate(tqdm(data, desc="EU (baseline)", unit="item")):
        scenario = str(item.get("scenario", "")).strip()
        subject = str(item.get("subject", "user")).strip()
        emo_choices = list(item.get("emotion_choices", []))
        cause_choices = list(item.get("cause_choices", []))

        prompt = build_eu_prompt_baseline(scenario=scenario, subject=subject, emotion_choices=emo_choices, cause_choices=cause_choices)
        raw = _run_with_retries(
            prompt,
            model,
            temperature=0.0,
            max_tokens=pred_max_tokens,
            attempts=attempts,
            log_prefix=f"[eu][{idx+1}]",
            log_raw=log_raw,
            log_full=log_full,
            log_file=log_file,
            initial_wait=retry_initial_wait,
            backoff=retry_backoff,
            jitter=retry_jitter,
        )
        if print_raw and raw:
            print(f"[eu][{idx+1}] raw prediction:\n{raw}\n", file=sys.stderr)
            _append_log(log_file, f"[eu][{idx+1}] raw prediction:\n{raw}\n")

        pred_emo = ""
        pred_cause = ""
        try:
            obj = robust_json_loads(raw)
            if isinstance(obj, dict):
                pred_emo = str(obj.get("emo_label") or obj.get("emotion") or "").strip()
                pred_cause = str(obj.get("cause_label") or obj.get("cause") or "").strip()
        except Exception:
            pass
        pred_emo = _coerce_to_choice(pred_emo, emo_choices)
        pred_cause = _coerce_to_choice(pred_cause, cause_choices)

        gt_emo = str(item.get("emotion_label", "")).strip()
        gt_cause = str(item.get("cause_label", "")).strip()

        is_emo_correct = pred_emo.strip().lower() == gt_emo.strip().lower()
        is_correct = is_emo_correct and (pred_cause.strip().lower() == gt_cause.strip().lower())
        if is_correct:
            correct += 1
        if is_emo_correct:
            emo_correct += 1

        res: Dict[str, Any] = {
            "idx": idx,
            "scenario": scenario,
            "subject": subject,
            "emotion_choices": emo_choices,
            "cause_choices": cause_choices,
            "raw_model_output": raw,
            "pred_emo": pred_emo,
            "pred_cause": pred_cause,
            "gt_emo": gt_emo,
            "gt_cause": gt_cause,
            "is_correct": is_correct,
            "is_emo_correct": is_emo_correct,
        }
        results.append(res)

    n = len(data) if data else 0
    accuracy = (correct / n) if n else 0.0
    emo_accuracy = (emo_correct / n) if n else 0.0
    return {"items": results, "accuracy": accuracy, "emo_accuracy": emo_accuracy}


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="EmoBench baseline (EA & EU) via OpenRouter API only")
    # Default model preference: Meta-Llama 3.1 8B Instruct; allow env overrides
    default_model = os.environ.get("OPENROUTER_MODEL") or os.environ.get("REMOTE_MODEL") or "meta-llama/llama-3.1-8b-instruct"

    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items per split")
    parser.add_argument("--model", type=str, default=default_model, help="OpenRouter model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for prediction steps")
    parser.add_argument("--pred-max-tokens", type=int, default=256, help="Max new tokens for strict JSON prediction")
    parser.add_argument("--attempts", type=int, default=5, help="Retries per item")
    parser.add_argument("--retry-initial-wait", type=float, default=1.0, help="Initial wait (seconds) before retrying an empty output")
    parser.add_argument("--retry-backoff", type=float, default=1.5, help="Multiplicative backoff factor for each subsequent retry")
    parser.add_argument("--retry-jitter", type=float, default=0.5, help="Add up to this many seconds of random jitter to each wait")
    parser.add_argument("--log-raw", action="store_true", help="Print raw model outputs per attempt to stderr (truncated)")
    parser.add_argument("--raw-full", action="store_true", help="When logging raw outputs, print full text without truncation")
    parser.add_argument("--print-raw", action="store_true", help="Print final raw prediction per item to stderr")
    parser.add_argument("--log-file", type=str, default=None, help="Path to append raw outputs when using --log-raw/--print-raw")
    parser.add_argument("--only-ea", action="store_true", help="Run only EA split")
    parser.add_argument("--only-eu", action="store_true", help="Run only EU split")
    args = parser.parse_args()

    _ensure_api_key()

    # Determine which splits to run
    run_ea = True
    run_eu = True
    if args.only_ea or args.only_eu:
        run_ea = bool(args.only_ea)
        run_eu = bool(args.only_eu)

    # Load datasets
    ea_data = _load_jsonl(EA_PATH) if run_ea else []
    eu_data = _load_jsonl(EU_PATH) if run_eu else []

    # Slice
    s = max(0, int(args.start))
    ea_run: List[Dict[str, Any]] = []
    eu_run: List[Dict[str, Any]] = []
    if run_ea:
        if args.limit is None:
            ea_run = ea_data[s:]
        else:
            e = min(len(ea_data), s + int(args.limit))
            ea_run = ea_data[s:e]
    if run_eu:
        if args.limit is None:
            eu_run = eu_data[s:]
        else:
            e2 = min(len(eu_data), s + int(args.limit))
            eu_run = eu_data[s:e2]

    print(f"Running EmoBench baseline via OpenRouter: EA={len(ea_run) if run_ea else 0}, EU={len(eu_run) if run_eu else 0}", file=sys.stderr)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path: Optional[Path] = (
        Path(args.log_file)
        if getattr(args, "log_file", None)
        else (OUT_DIR / f"baseline_{run_id}.log")
    )

    meta = {
        "run_id": run_id,
        "model": args.model,
        "temperature": args.temperature,
        "attempts": args.attempts,
        "method": "openrouter_baseline",
    }

    # Evaluate EA
    if run_ea:
        ea_res = evaluate_ea(
            data=ea_run,
            model=args.model,
            temperature=args.temperature,
            pred_max_tokens=args.pred_max_tokens,
            attempts=args.attempts,
            log_raw=args.log_raw,
            log_full=args.raw_full,
            print_raw=args.print_raw,
            log_file=log_path,
            retry_initial_wait=args.retry_initial_wait,
            retry_backoff=args.retry_backoff,
            retry_jitter=args.retry_jitter,
        )
        with EA_OUT.open("w", encoding="utf-8") as f:
            json.dump({"meta": meta, **ea_res}, f, ensure_ascii=False, indent=2)
        print(f"EA baseline results saved -> {EA_OUT}", file=sys.stderr)
        # Print final EA accuracy summary
        try:
            print(
                f"EA accuracy: {ea_res.get('accuracy', 0.0):.4f} on {len(ea_run)} items",
                file=sys.stderr,
            )
        except Exception:
            pass

    # Evaluate EU
    if run_eu:
        eu_res = evaluate_eu(
            data=eu_run,
            model=args.model,
            temperature=args.temperature,
            pred_max_tokens=args.pred_max_tokens,
            attempts=args.attempts,
            log_raw=args.log_raw,
            log_full=args.raw_full,
            print_raw=args.print_raw,
            log_file=log_path,
            retry_initial_wait=args.retry_initial_wait,
            retry_backoff=args.retry_backoff,
            retry_jitter=args.retry_jitter,
        )
        with EU_OUT.open("w", encoding="utf-8") as f:
            json.dump({"meta": meta, **eu_res}, f, ensure_ascii=False, indent=2)
        print(f"EU baseline results saved -> {EU_OUT}", file=sys.stderr)
        # Print final EU accuracy summary
        try:
            print(
                f"EU accuracy: {eu_res.get('accuracy', 0.0):.4f} | emo_accuracy: {eu_res.get('emo_accuracy', 0.0):.4f} on {len(eu_run)} items",
                file=sys.stderr,
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
