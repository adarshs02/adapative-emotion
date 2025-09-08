#!/usr/bin/env python3
"""
EmoBench EA baseline via OpenRouter API only (no EmoBIRD pipeline).

What this fixes vs typical gotchas:
- Handles choices that are dicts ({"A": "...", ...}) OR lists (["...", ...]).
- Prompts show canonical letters with the real choice TEXT (A) TextA, (B) TextB, ...
- Strict vs lenient metrics:
    * Strict: JSON parses; letter valid; text exactly matches that letter’s text.
    * Lenient: allows inferring the letter from text when JSON is imperfect.
- No silent "default to first choice" coercion.
- Audit: parse-success rate + majority-letter baseline.

Output:
- results/emobench/baseline_ea_results.json

Requirements:
- OPENROUTER_API_KEY set in env
- Uses EmoBIRDv2 helpers:
    - EmoBIRDv2.utils.utils.robust_json_loads
    - EmoBIRDv2.scripts.abstract_generator.call_openrouter
Default model: meta-llama/llama-3.1-8b-instruct (override with --model or env OPENROUTER_MODEL/REMOTE_MODEL)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from collections import OrderedDict
from tqdm import tqdm
import time
import random

# ---------- Repo paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------- OpenRouter + utils ----------
from EmoBIRDv2.utils.constants import (
    OPENROUTER_API_KEY,
    OPENROUTER_CONNECT_TIMEOUT,
    OPENROUTER_READ_TIMEOUT,
)
import EmoBIRDv2.scripts.abstract_generator as AG  # call_openrouter
from EmoBIRDv2.utils.utils import robust_json_loads

# ---------- Data & output paths ----------
EMOBENCH_DIR = REPO_ROOT / "datasets" / "EmoBench" / "data"
EA_PATH = EMOBENCH_DIR / "EA.jsonl"

OUT_DIR = REPO_ROOT / "results" / "emobench"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EA_OUT = OUT_DIR / "baseline_ea_results.json"

# ---------- Small utils ----------
LETTER_ALPHABET = [chr(ord("A") + i) for i in range(26)]

def _append_log(log_file: Optional[Path], text: str) -> None:
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
        pass  # never interrupt eval for logging issues

def _ensure_api_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def _run_with_retries(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    attempts: int = 1,
    log_prefix: Optional[str] = None,
    log_raw: bool = False,
    log_full: bool = False,
    log_file: Optional[Path] = None,
    initial_wait: float = 1.0,
    backoff: float = 1.5,
    jitter: float = 0.5,
) -> str:
    """Use attempts=1 for deterministic head-to-heads. Increase only if you must."""
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
            # increase wait using backoff
            try:
                if backoff and backoff > 0:
                    wait_s *= float(backoff)
            except Exception:
                pass
    return last

# ---------- Choice normalization & mapping ----------
def _normalize_choices(choices_obj: Any) -> Tuple[OrderedDict[str, str], List[str]]:
    """
    Accepts:
      - dict: {"A": "textA", "B": "textB", ...}
      - list: ["textA", "textB", "textC", "textD"]
    Returns:
      choices_map: OrderedDict("A" -> textA, "B" -> textB, ...)
      letters: list of letters in order
    """
    if isinstance(choices_obj, dict):
        letters = [L for L in LETTER_ALPHABET if L in choices_obj]
        choices_map = OrderedDict((L, str(choices_obj[L])) for L in letters)
    elif isinstance(choices_obj, list):
        n = len(choices_obj)
        letters = LETTER_ALPHABET[:n]
        choices_map = OrderedDict((LETTER_ALPHABET[i], str(choices_obj[i])) for i in range(n))
    else:
        raise ValueError(f"Unsupported choices type: {type(choices_obj)}")

    if len(choices_map) < 2:
        raise ValueError(f"Expected at least 2 choices, got {len(choices_map)}")
    return choices_map, letters

def _infer_letter_from_text(txt: str, choices_map: OrderedDict[str, str], *, fuzzy: bool) -> Optional[str]:
    if not txt:
        return None
    txt = txt.strip()
    # exact
    for L, c in choices_map.items():
        if txt == c:
            return L
    # case-insensitive
    for L, c in choices_map.items():
        if txt.lower() == c.lower():
            return L
    # fuzzy
    if fuzzy:
        try:
            import difflib
            best = difflib.get_close_matches(txt, list(choices_map.values()), n=1, cutoff=0.0)
            if best:
                inv = {v: k for k, v in choices_map.items()}
                return inv.get(best[0])
        except Exception:
            pass
    return None

def _letter_from_any(label: str, choices_map: OrderedDict[str, str]) -> Optional[str]:
    """Map GT label (letter OR text) to canonical letter."""
    if not label:
        return None
    lab = str(label).strip()
    up = lab.upper()
    if up in choices_map:
        return up
    return _infer_letter_from_text(lab, choices_map, fuzzy=False)

# ---------- Prompt ----------
def build_ea_prompt_baseline(*, scenario: str, subject: str, choices_obj: Any) -> str:
    choices_map, letters = _normalize_choices(choices_obj)
    choice_lines = [f"{L}) {choices_map[L]}" for L in letters]
    system = "Select exactly one option out of the given choices. Respond with strict JSON only."
    user = f"""
Subject: {subject}
Scenario: {scenario}

Choices (select exactly one):
{os.linesep.join(choice_lines)}

Return STRICT JSON ONLY (no markdown/code fences) with keys:
{{
  "choice_letter": "A|B|C|D|...",
  "choice_text": "<verbatim copy of the chosen option>"
}}
Rules:
- choice_text must be an exact verbatim match of the chosen option.
- No text before or after the JSON object.
"""
    return f"{system}\n\n{user}"

# ---------- Evaluation ----------
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
    items: List[Dict[str, Any]] = []
    strict_correct = 0
    lenient_correct = 0
    parse_success = 0

    gt_letter_counts: Dict[str, int] = {}

    for idx, item in enumerate(tqdm(data, desc="EA (baseline)", unit="item")):
        scenario = str(item.get("scenario", "")).strip()
        subject = str(item.get("subject", "user")).strip()
        choices_obj = item.get("choices")

        # Normalize & ground truth
        choices_map, letters = _normalize_choices(choices_obj)
        gt_label = str(item.get("label", "")).strip()
        gt_letter = _letter_from_any(gt_label, choices_map)
        if gt_letter:
            gt_letter_counts[gt_letter] = gt_letter_counts.get(gt_letter, 0) + 1

        # Prompt + call
        prompt = build_ea_prompt_baseline(scenario=scenario, subject=subject, choices_obj=choices_obj)
        raw = _run_with_retries(
            prompt,
            model,
            temperature=temperature,
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
            msg = f"[ea][{idx+1}] raw prediction:\n{raw}\n"
            print(msg, file=sys.stderr)
            _append_log(log_file, msg)

        # Parse strict JSON
        pred_letter = ""
        pred_text = ""
        strict_valid = False
        parsed_ok = False
        try:
            obj = robust_json_loads(raw)
            parsed_ok = isinstance(obj, dict)
            if parsed_ok:
                pred_letter = str(obj.get("choice_letter", "")).strip().upper()
                pred_text = str(obj.get("choice_text", "")).strip()
                if pred_letter in choices_map and pred_text == choices_map[pred_letter]:
                    strict_valid = True
        except Exception:
            parsed_ok = False

        if parsed_ok:
            parse_success += 1

        # If letter missing, infer from text (lenient only)
        if not pred_letter:
            inferred = _infer_letter_from_text(pred_text, choices_map, fuzzy=True)
            if inferred:
                pred_letter = inferred

        # Scoring
        is_correct_strict = bool(strict_valid and gt_letter and (pred_letter == gt_letter))
        is_correct_lenient = bool(gt_letter and (pred_letter == gt_letter))

        if is_correct_strict:
            strict_correct += 1
        if is_correct_lenient:
            lenient_correct += 1

        items.append({
            "idx": idx,
            "scenario": scenario,
            "subject": subject,
            "choices": dict(choices_map),
            "raw_model_output": raw,
            "pred_letter": pred_letter,
            "pred_text": pred_text,
            "gt_letter": gt_letter,
            "gt_text": choices_map.get(gt_letter) if gt_letter else None,
            "parse_success": parsed_ok,
            "is_correct_strict": is_correct_strict,
            "is_correct_lenient": is_correct_lenient,
        })

    n = len(data) if data else 0
    strict_acc = (strict_correct / n) if n else 0.0
    lenient_acc = (lenient_correct / n) if n else 0.0
    parse_rate = (parse_success / n) if n else 0.0

    # Majority-letter baseline
    if gt_letter_counts:
        maj_letter = max(gt_letter_counts.items(), key=lambda kv: kv[1])[0]
        maj_acc = (gt_letter_counts[maj_letter] / sum(gt_letter_counts.values()))
    else:
        maj_letter, maj_acc = None, 0.0

    return {
        "items": items,
        "accuracy_strict": strict_acc,
        "accuracy_lenient": lenient_acc,
        "parse_success_rate": parse_rate,
        "majority_letter": maj_letter,
        "majority_baseline_acc": maj_acc,
    }

# ---------- Main ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="EmoBench EA baseline via OpenRouter API only")
    default_model = os.environ.get("OPENROUTER_MODEL") or os.environ.get("REMOTE_MODEL") or "meta-llama/llama-3.1-8b-instruct"

    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    parser.add_argument("--model", type=str, default=default_model, help="OpenRouter model name")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--pred-max-tokens", type=int, default=1536, help="Max new tokens for strict JSON prediction")
    parser.add_argument("--attempts", type=int, default=1, help="Retries per item (use 1 for deterministic evals)")
    parser.add_argument("--retry-initial-wait", type=float, default=1.0, help="Initial wait (seconds) before retrying an empty output")
    parser.add_argument("--retry-backoff", type=float, default=0.1, help="Multiplicative backoff factor for each subsequent retry")
    parser.add_argument("--retry-jitter", type=float, default=0.1, help="Add up to this many seconds of random jitter to each wait")
    parser.add_argument("--connect-timeout", type=float, default=float(OPENROUTER_CONNECT_TIMEOUT), help="HTTP connect timeout (seconds) for OpenRouter requests")
    parser.add_argument("--read-timeout", type=float, default=float(OPENROUTER_READ_TIMEOUT), help="HTTP read/response timeout (seconds) for OpenRouter requests")
    parser.add_argument("--log-raw", action="store_true", help="Print raw model outputs per attempt to stderr (truncated)")
    parser.add_argument("--raw-full", action="store_true", help="When logging raw outputs, print full text without truncation")
    parser.add_argument("--print-raw", action="store_true", help="Print final raw prediction per item to stderr")
    parser.add_argument("--log-file", type=str, default=None, help="Path to append raw outputs when using --log-raw/--print-raw")
    args = parser.parse_args()

    _ensure_api_key()

    # Configure OpenRouter timeouts on the AG module (no changes to abstract_generator)
    try:
        if args.connect_timeout is not None:
            AG.OPENROUTER_CONNECT_TIMEOUT = float(args.connect_timeout)
        if args.read_timeout is not None:
            AG.OPENROUTER_READ_TIMEOUT = float(args.read_timeout)
    except Exception:
        pass

    # Load & slice
    ea_data = _load_jsonl(EA_PATH)
    s = max(0, int(args.start))
    if args.limit is None:
        run_data = ea_data[s:]
    else:
        e = min(len(ea_data), s + int(args.limit))
        run_data = ea_data[s:e]

    print(f"Running EmoBench EA via OpenRouter: n={len(run_data)}", file=sys.stderr)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path: Optional[Path] = Path(args.log_file) if getattr(args, "log_file", None) else (OUT_DIR / f"baseline_ea_{run_id}.log")

    meta = {
        "run_id": run_id,
        "model": args.model,
        "temperature": args.temperature,
        "attempts": args.attempts,
        "pred_max_tokens": args.pred_max_tokens,
        "connect_timeout": args.connect_timeout,
        "read_timeout": args.read_timeout,
        "method": "openrouter_ea_baseline_fixed",
    }

    # Evaluate EA
    ea_res = evaluate_ea(
        data=run_data,
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

    # Summary line (strict first; lenient for debugging)
    try:
        print(
            f"EA strict_acc={ea_res.get('accuracy_strict', 0.0):.4f} | "
            f"lenient_acc={ea_res.get('accuracy_lenient', 0.0):.4f} | "
            f"parse={ea_res.get('parse_success_rate', 0.0):.3f} | "
            f"maj={ea_res.get('majority_baseline_acc', 0.0):.3f} ({ea_res.get('majority_letter')})",
            file=sys.stderr,
        )
    except Exception:
        pass

if __name__ == "__main__":
    main()
