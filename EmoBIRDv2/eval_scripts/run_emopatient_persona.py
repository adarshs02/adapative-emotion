#!/usr/bin/env python3
"""
Baseline runner: EmoPatient × Base Model via OpenRouter (QA-only, no EmoBIRD pipeline)

- Reads scenarios from the EmoPatient dataset JSON (default: datasets/EmoPatient/scenarios.json)
- For each scenario's QA item, sends a single-turn prompt to a chosen OpenRouter model
- Writes a consolidated dataset-style JSON (like scenarios_30.json) with gt replaced by the model's response

Usage examples:
  export OPENROUTER_API_KEY=...  # required
  python EmoBIRDv2/eval_scripts/run_emopatient_basemodel.py --limit 1 --q-limit 2 --model openai/gpt-4o-mini --log-raw
  python EmoBIRDv2/eval_scripts/run_emopatient_basemodel.py --dataset-out /tmp/emopatient_basemodel.json --qa-retries 2

Notes:
- QA-only: each question is treated as a single-turn conversation: Context (narrative + optional metadata) + Question
- Change --model to switch base models easily (uses OpenRouter chat completions)
- If a QA's response is empty, the script will retry up to --qa-retries times
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

# Repo paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
import sys
# Ensure package imports like 'from EmoBIRDv2.utils...' resolve when running directly
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# EmoBIRDv2 constants (for defaults and timeouts)
from EmoBIRDv2.utils.constants import (
    OPENROUTER_API_KEY,
    MODEL_NAME as DEFAULT_MODEL,
    MODEL_TEMPERATURE as DEFAULT_TEMPERATURE,
    OPENROUTER_CONNECT_TIMEOUT,
    OPENROUTER_READ_TIMEOUT,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def ensure_api_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")


def load_emopatient(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(scenarios, list):
        raise ValueError(f"Unexpected EmoPatient format in {path}")
    return scenarios


def format_user_prompt(scn: Dict[str, Any], question: str, *, include_meta: bool = True) -> str:
    title = str(scn.get("title", "")).strip()
    diagnosis = str(scn.get("diagnosis", "")).strip()
    treatment = str(scn.get("treatment_plan", "")).strip()
    narrative = str(scn.get("narrative", "")).strip()

    parts: List[str] = []
    if include_meta:
        if title:
            parts.append(f"Title: {title}")
        if diagnosis:
            parts.append(f"Diagnosis: {diagnosis}")
        if treatment:
            parts.append(f"Treatment plan: {treatment}")
    if narrative:
        parts.append(f"Narrative: {narrative}")

    parts.append("Question:")
    parts.append(question.strip())

    parts.append(
        "\nInstructions:\n"
        "You are an AI Chatbot."
        "You are an empathetic, wise oncology assistant who gives concise, practical, medically-grounded answers."
        "Given the context, Answer the question concisely (Total response length: 100-150 words).\n"
    )

    return "\n\n".join(parts).strip()


def call_openrouter_chat(*, api_key: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str], user_prompt: str, connect_timeout: int, read_timeout: int) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    resp = requests.post(
        OPENROUTER_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=(connect_timeout, read_timeout),
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def run_for_question(
    *,
    scn: Dict[str, Any],
    question: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: Optional[str],
    attempts: int,
    log_raw: bool,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    user_prompt = format_user_prompt(scn, question, include_meta=True)
    last: str = ""
    for i in range(1, attempts + 1):
        try:
            raw = call_openrouter_chat(
                api_key=OPENROUTER_API_KEY or "",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            if log_raw and raw:
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                print(f"[raw] {trunc}", file=sys.stderr)
        except Exception as e:
            print(f"[error] attempt {i}/{attempts} failed: {e}", file=sys.stderr)
            raw = ""
        if raw:
            last = raw
            break
    return last


def process_single_qa(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single QA pair through the base model.
    Returns the QA output with gt replaced by model response.
    """
    j = task["qa_idx"]
    qa = task["qa"]
    scn = task["scn"]
    args = task["args"]
    sid = task["sid"]
    
    q_text = str(qa.get("q", "")).strip()
    if not q_text:
        return {"qa_idx": j, "qa_out": dict(qa)}
    
    answer = ""
    max_qa_retries = max(0, int(args.qa_retries))
    
    for rtry in range(max_qa_retries + 1):
        if rtry > 0:
            print(f"[{sid} Q{j+1}] empty, retry {rtry}/{max_qa_retries}...", file=sys.stderr)
        else:
            print(f"[{sid} Q{j+1}] running...", file=sys.stderr)
        
        answer = run_for_question(
            scn=scn,
            question=q_text,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=args.system_prompt,
            attempts=args.attempts,
            log_raw=args.log_raw,
            connect_timeout=int(args.openrouter_connect_timeout),
            read_timeout=int(args.openrouter_read_timeout),
        )
        if answer.strip():
            break
        time.sleep(0.3)
    
    qa_out = dict(qa)
    qa_out["gt"] = answer.strip()
    
    return {"qa_idx": j, "qa_out": qa_out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run base model over EmoPatient dataset (QA-only, OpenRouter)")
    parser.add_argument("--data", type=str, default=str(REPO_ROOT / "datasets" / "EmoPatient" / "scenarios_30.json"), help="Path to EmoPatient scenarios.json")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based) for scenarios")
    parser.add_argument("--limit", type=int, default=None, help="Process only N scenarios")
    parser.add_argument("--q-start", type=int, default=0, help="Start within QA list")
    parser.add_argument("--q-limit", type=int, default=None, help="Process only N questions per scenario")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter model name (easily switch)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1536, help="Max new tokens for the answer")
    parser.add_argument("--system-prompt", type=str, default="You are an empathetic, wise oncology assistant who gives concise, practical, medically-grounded answers.", help="Optional system prompt")

    parser.add_argument("--attempts", type=int, default=5, help="Retries per QA call")
    parser.add_argument("--qa-retries", type=int, default=1, help="Additional re-runs if response is empty")
    parser.add_argument("--openrouter-connect-timeout", type=int, default=OPENROUTER_CONNECT_TIMEOUT, help="Connect timeout (s)")
    parser.add_argument("--openrouter-read-timeout", type=int, default=OPENROUTER_READ_TIMEOUT, help="Read timeout (s)")

    parser.add_argument("--dataset-out", type=str, default=None, help="Output consolidated dataset JSON path (gt replaced by base model response)")
    parser.add_argument("--log-raw", action="store_true", help="Print truncated raw model outputs to stderr")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (threads) for processing QA pairs. Default: 1 (sequential). Recommended: 4-10 for faster processing.")

    args = parser.parse_args()

    ensure_api_key()

    data_path = Path(args.data)
    scenarios = load_emopatient(data_path)

    start = max(0, int(args.start))
    end = len(scenarios) if args.limit is None else min(len(scenarios), start + int(args.limit))
    run_scenarios = list(enumerate(scenarios[start:end], start=start))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Running EmoPatient x BaseModel: {len(run_scenarios)} scenarios | model={args.model} | run_id={run_id}", file=sys.stderr)

    consolidated: Dict[str, Any] = {"scenarios": []}
    consolidated_lock = Lock()

    for idx, scn in tqdm(run_scenarios, total=len(run_scenarios), desc="Scenarios", unit="scn"):
        sid = scn.get("id") or f"S{idx+1}"
        qa_list = scn.get("qa") if isinstance(scn.get("qa"), list) else []
        if not qa_list:
            print(f"[skip] {sid} has no QA entries; skipping.", file=sys.stderr)
            continue

        q_start = max(0, int(getattr(args, "q_start", 0)))
        q_end = len(qa_list) if args.q_limit is None else min(len(qa_list), q_start + int(args.q_limit))
        sub_qa = list(enumerate(qa_list[q_start:q_end], start=q_start))

        # Build output scenario (copy all fields except qa, then rebuild qa)
        scn_out: Dict[str, Any] = {}
        for k, v in scn.items():
            if k != "qa":
                scn_out[k] = v
        scn_out["qa"] = [None] * len(sub_qa)  # Pre-allocate to preserve order

        # Prepare QA processing tasks
        qa_tasks = []
        for j, qa in sub_qa:
            qa_tasks.append({
                "qa_idx": j,
                "qa": qa,
                "scn": scn,
                "args": args,
                "sid": sid,
            })

        # Process QA pairs in parallel or sequential
        workers = max(1, int(args.workers))
        if workers == 1:
            # Sequential processing (original behavior)
            for i, task in enumerate(tqdm(qa_tasks, desc=f"{sid} QA", unit="qa", leave=False)):
                result = process_single_qa(task)
                scn_out["qa"][i] = result["qa_out"]
        else:
            # Parallel processing with ThreadPoolExecutor
            results_dict = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_idx = {executor.submit(process_single_qa, task): i for i, task in enumerate(qa_tasks)}
                
                # Collect results as they complete
                with tqdm(total=len(qa_tasks), desc=f"{sid} QA", unit="qa", leave=False) as pbar:
                    for future in as_completed(future_to_idx):
                        task_idx = future_to_idx[future]
                        try:
                            result = future.result()
                            results_dict[task_idx] = result
                        except Exception as exc:
                            print(f"[{sid}] QA task {task_idx+1} generated exception: {exc}", file=sys.stderr)
                            # Store empty result
                            results_dict[task_idx] = {
                                "qa_idx": qa_tasks[task_idx]["qa_idx"],
                                "qa_out": dict(qa_tasks[task_idx]["qa"]),
                            }
                        finally:
                            pbar.update(1)
            
            # Reassemble results in original order
            for i in range(len(qa_tasks)):
                if i in results_dict:
                    scn_out["qa"][i] = results_dict[i]["qa_out"]

        # Append consolidated scenario (thread-safe)
        with consolidated_lock:
            consolidated["scenarios"].append(scn_out)

    # Pick default dataset path if not provided
    out_path = Path(args.dataset_out) if args.dataset_out else (REPO_ROOT / "EmoBIRDv2" / "eval_results" / "emopatient" / f"emopatient_basemodel_{run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
    print(f"[done] consolidated -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()