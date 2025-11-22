#!/usr/bin/env python3
"""
Multi-turn evaluation: EmoPatientMulti × EmoBIRDv2 pipeline

- Reads multi-turn scenarios from datasets/EmoPatientMulti/scenarios.json
- For each dialogue, processes turns sequentially, building conversation history
- Each assistant turn runs through the EmoBIRDv2 pipeline
- Uses the generated content as context for subsequent turns
- Saves per-dialogue JSON results under EmoBIRDv2/eval_results/emopatient_multiturn

Usage examples:
  export OPENROUTER_API_KEY=...  # required
  python EmoBIRDv2/eval_scripts/run_emopatient_multiturn_emobirdv2.py --limit 2
  python EmoBIRDv2/eval_scripts/run_emopatient_multiturn_emobirdv2.py --limit 10 --workers 5
  python EmoBIRDv2/eval_scripts/run_emopatient_multiturn_emobirdv2.py --start 5 --limit 1 --log-raw
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def load_multiturn_scenarios(path: Path) -> List[Dict[str, Any]]:
    """Load multi-turn scenarios from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of scenarios in {path}, got {type(data)}")
    return data


def build_system_message(dialogue: Dict[str, Any]) -> str:
    """Build system message with clinical context."""
    parts = []
    
    if dialogue.get("diagnosis"):
        parts.append(f"Diagnosis: {dialogue['diagnosis']}")
    if dialogue.get("treatment_plan"):
        parts.append(f"Treatment Plan: {dialogue['treatment_plan']}")
    if dialogue.get("narrative"):
        parts.append(f"Patient Background: {dialogue['narrative']}")
    
    return "\n".join(parts)


def build_messages_for_pipeline(
    system_context: str,
    conversation_history: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Build message array for pipeline input, similar to vLLM chat API.
    Returns array of {role, content} messages.
    """
    messages = []
    
    # Add system message with clinical context
    if system_context:
        messages.append({"role": "system", "content": system_context})
    
    # Add conversation history (user/assistant pairs)
    for msg in conversation_history:
        role = "user" if msg["role"] == "patient" else "assistant"
        messages.append({"role": role, "content": msg["text"]})
    
    return messages


def build_situation_from_messages(messages: List[Dict[str, str]]) -> str:
    """Convert messages back to situation text for existing pipeline steps."""
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            parts.append(msg["content"])
        elif msg["role"] == "user":
            parts.append(f"Patient: {msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"Assistant: {msg['content']}")
    return "\n\n".join(parts)


def call_openrouter_with_messages(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call OpenRouter with structured message history."""
    import requests
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=(AG.OPENROUTER_CONNECT_TIMEOUT, AG.OPENROUTER_READ_TIMEOUT),
    )
    resp.raise_for_status()
    data = resp.json()
    
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"OpenRouter error: {data.get('error')}")
    
    content = (data["choices"][0]["message"].get("content") or "").strip()
    return content


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


def run_pipeline_for_turn(
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
    """Run the full EmoBIRDv2 pipeline for a single turn."""
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

    # Step 1: Abstract
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

    # Step 2: Factors
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

    # Step 3: Select
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

    # Step 4: Emotions (optional)
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

    # Step 5: Likert (requires emotions)
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

    # Step 6: Final output
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


def process_dialogue_task(task_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for processing a single dialogue with file I/O.
    Used for parallel processing.
    """
    dialogue = task_info["dialogue"]
    args = task_info["args"]
    idx = task_info["idx"]
    out_dir = task_info["out_dir"]
    run_id = task_info["run_id"]
    
    dialogue_id = dialogue.get("dialogue_id", f"D{idx+1}")
    
    # Check for existing results if resume is enabled
    if args.resume:
        existing = list(out_dir.glob(f"{dialogue_id}_emobirdv2_*.json"))
        if existing:
            print(f"[skip] {dialogue_id} already has results (resume)", file=sys.stderr)
            return None
    
    # Process the dialogue
    result = process_dialogue(dialogue, args, idx)
    
    # Write individual dialogue result
    out_path = out_dir / f"{dialogue_id}_emobirdv2_{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[done] {dialogue_id} -> {out_path}", file=sys.stderr)
    
    return result


def process_dialogue(
    dialogue: Dict[str, Any],
    args: argparse.Namespace,
    dialogue_idx: int,
) -> Dict[str, Any]:
    """
    Process a single multi-turn dialogue through the pipeline.
    Each assistant turn is generated using the EmoBIRDv2 pipeline with full conversation context.
    Uses message-based context similar to vLLM chat API.
    """
    dialogue_id = dialogue.get("dialogue_id", f"D{dialogue_idx+1}")
    turns = dialogue.get("turns", [])
    
    print(f"\n[{dialogue_id}] Processing {len(turns)} turns...", file=sys.stderr)
    
    # Build system message with clinical context (used for all turns)
    system_context = build_system_message(dialogue)
    
    # Track conversation history as messages (patient/assistant pairs)
    conversation_history: List[Dict[str, str]] = []
    
    # Results for this dialogue
    turn_results = []
    completed_turns = []
    
    for turn_idx, turn in enumerate(turns):
        role = turn.get("role")
        text = turn.get("text", "").strip()
        
        if role == "patient":
            # Patient turn - add to history for next assistant turn
            conversation_history.append({"role": "patient", "text": text})
            completed_turns.append({"role": "patient", "text": text})
            print(f"  Turn {turn_idx+1}: Patient message added to context", file=sys.stderr)
            
        elif role == "assistant":
            # Assistant turn - run through EmoBIRDv2 pipeline
            
            # Build full message array with system + conversation history
            messages = build_messages_for_pipeline(system_context, conversation_history)
            
            # Convert messages to situation text for pipeline
            situation = build_situation_from_messages(messages)
            
            print(f"  Turn {turn_idx+1}: Generating assistant response...", file=sys.stderr)
            
            # Run pipeline with retries
            max_retries = max(0, int(getattr(args, "turn_retries", 1)))
            response_text = ""
            pipeline_result = None
            
            for retry in range(max_retries + 1):
                if retry > 0:
                    print(f"    Retry {retry}/{max_retries} (empty output)...", file=sys.stderr)
                
                pipeline_result = run_pipeline_for_turn(
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
                
                response_text = (pipeline_result.get("final_output") or "").strip()
                if response_text:
                    break
                time.sleep(0.3)
            
            # Add assistant response to history for next turns
            conversation_history.append({"role": "assistant", "text": response_text})
            completed_turns.append({"role": "assistant", "text": response_text})
            
            # Record pipeline result for this turn
            turn_results.append({
                "turn_index": turn_idx + 1,
                "role": "assistant",
                "messages": messages,  # Store the message array used
                "situation": situation,
                "pipeline": pipeline_result,
                "retries_used": retry,
            })
            
            print(f"    Generated: {response_text[:100]}{'...' if len(response_text) > 100 else ''}", file=sys.stderr)
    
    return {
        "dialogue_id": dialogue_id,
        "diagnosis": dialogue.get("diagnosis"),
        "treatment_plan": dialogue.get("treatment_plan"),
        "narrative": dialogue.get("narrative"),
        "turns": completed_turns,
        "turn_results": turn_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EmoBIRDv2 pipeline on multi-turn EmoPatient dialogues")
    parser.add_argument("--data", type=str, default=str(REPO_ROOT / "datasets" / "EmoPatientMulti" / "scenarios.json"), help="Path to multi-turn scenarios JSON")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based) for dialogues")
    parser.add_argument("--limit", type=int, default=None, help="Process only N dialogues")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    parser.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--abs-max-tokens", type=int, default=ABSTRACT_MAX_TOKENS)
    parser.add_argument("--fac-max-tokens", type=int, default=FACTOR_MAX_TOKENS)
    parser.add_argument("--sel-max-tokens", type=int, default=FACTOR_MAX_TOKENS)
    parser.add_argument("--emo-max-tokens", type=int, default=EMOTION_MAX_TOKENS)
    parser.add_argument("--likert-max-tokens", type=int, default=LIKERT_MAX_TOKENS)
    parser.add_argument("--out-max-tokens", type=int, default=2048)  # Increased from OUTPUT_MAX_TOKENS (1536) to prevent truncation
    parser.add_argument("--attempts", type=int, default=5, help="Retries per pipeline step")
    parser.add_argument("--turn-retries", "--qa-retries", type=int, default=1, dest="turn_retries", help="Re-run pipeline for a turn if output is empty")
    parser.add_argument("--with-emotions", dest="with_emotions", action="store_true", default=True, help="Enable emotions+likert steps")
    parser.add_argument("--no-emotions", dest="with_emotions", action="store_false", help="Disable emotions+likert steps")
    parser.add_argument("--openrouter-connect-timeout", type=int, default=None)
    parser.add_argument("--openrouter-read-timeout", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "EmoBIRDv2" / "eval_results" / "emopatient_multiturn"), help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Skip dialogues that already have output files")
    parser.add_argument("--log-raw", action="store_true", help="Print truncated raw model outputs to stderr")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for processing dialogues. Default: 1 (sequential)")

    args = parser.parse_args()

    _ensure_api_key()
    _apply_timeout_overrides(args)

    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dialogues = load_multiturn_scenarios(data_path)
    start = max(0, int(args.start))
    end = len(dialogues) if args.limit is None else min(len(dialogues), start + int(args.limit))
    run_dialogues = list(enumerate(dialogues[start:end], start=start))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Running EmoPatientMulti × EmoBIRDv2: {len(run_dialogues)} dialogues | run_id={run_id}", file=sys.stderr)

    all_results = []
    workers = max(1, int(args.workers))

    # Prepare tasks for all dialogues
    tasks = []
    for idx, dialogue in run_dialogues:
        tasks.append({
            "dialogue": dialogue,
            "args": args,
            "idx": idx,
            "out_dir": out_dir,
            "run_id": run_id,
        })

    if workers == 1:
        # Sequential processing (original behavior)
        print(f"Processing {len(tasks)} dialogues sequentially...", file=sys.stderr)
        for task in tasks:
            result = process_dialogue_task(task)
            if result is not None:
                all_results.append(result)
            time.sleep(0.5)  # Gentle pacing
    else:
        # Parallel processing with ThreadPoolExecutor
        print(f"Processing {len(tasks)} dialogues with {workers} workers...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_dialogue_task, task): i for i, task in enumerate(tasks)}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                task_idx = future_to_task[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                except Exception as exc:
                    dialogue_id = tasks[task_idx]["dialogue"].get("dialogue_id", f"D{task_idx+1}")
                    print(f"[error] {dialogue_id} generated exception: {exc}", file=sys.stderr)
                finally:
                    completed += 1
                    print(f"Progress: {completed}/{len(tasks)} dialogues completed", file=sys.stderr)

    # Write consolidated results
    consolidated_path = out_dir / f"emopatient_multiturn_emobirdv2_{run_id}.json"
    consolidated = {
        "run_id": run_id,
        "model": args.model,
        "temperature": args.temperature,
        "dialogues": all_results,
    }
    with consolidated_path.open("w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
    print(f"\n[done] Consolidated results -> {consolidated_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
