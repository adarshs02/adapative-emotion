#!/usr/bin/env python3
"""
EmoBench (EA and EU) evaluation using the EmoBIRDv2 pipeline via OpenRouter.

Pipeline per item:
1) Abstract from scenario
2) Factors from abstract
3) Factor value selection from full scenario using discovered factors
4) Strict JSON prediction:
   - EA: select exactly one choice from choices
   - EU: select exactly one emotion and one cause from given choices
5) Optional policy-based closeness scoring via OpenRouter

Outputs:
- results/emobench/emobirdv2_ea_results.json
- results/emobench/emobirdv2_eu_results.json

Requirements:
- OPENROUTER_API_KEY in env
- Defaults to Meta-Llama 3.1 8B Instruct (can override with --model)

Logging:
- Raw model outputs are printed to stderr when `--log-raw`/`--print-raw` are provided.
- Additionally, use `--log-file` to append these raw outputs to a persistent file.
- If `--log-file` is not specified, a default file will be used: `results/emobench/emobirdv2_<run_id>.log`.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

# Ensure repo root is importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
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
)
import EmoBIRDv2.scripts.abstract_generator as AG
import EmoBIRDv2.scripts.factor_generator as FG
import EmoBIRDv2.scripts.factor_value_selector as FVS
from EmoBIRDv2.utils.utils import robust_json_loads

# Data paths
EMOBENCH_DIR = REPO_ROOT / "datasets" / "EmoBench" / "data"
EA_PATH = EMOBENCH_DIR / "EA.jsonl"
EU_PATH = EMOBENCH_DIR / "EU.jsonl"

OUT_DIR = REPO_ROOT / "results" / "emobench"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EA_OUT = OUT_DIR / "emobirdv2_ea_results.json"
EU_OUT = OUT_DIR / "emobirdv2_eu_results.json"


def _append_log(log_file: Optional[Path], text: str) -> None:
    """Append a line/block to a log file; ignore errors to not break evaluation."""
    if not log_file:
        return
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    except Exception:
        pass


def _ensure_api_key() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")


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
) -> str:
    last = ""
    for i in range(1, attempts + 1):
        try:
            raw = AG.call_openrouter(
                prompt=prompt,
                api_key=OPENROUTER_API_KEY,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if log_prefix and log_raw and raw:
                if log_full:
                    print(f"{log_prefix} Attempt {i}/{attempts} raw: {raw}", file=sys.stderr)
                    _append_log(log_file, f"{log_prefix} Attempt {i}/{attempts} raw: {raw}")
                else:
                    trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                    print(f"{log_prefix} Attempt {i}/{attempts} raw: {trunc}", file=sys.stderr)
                    _append_log(log_file, f"{log_prefix} Attempt {i}/{attempts} raw: {trunc}")
        except Exception as e:
            print(f"{log_prefix or ''} Attempt {i}/{attempts} failed: {e}", file=sys.stderr)
            raw = ""
        if raw:
            last = raw
            break
    return last


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# -------- EmoBIRDv2 pipeline steps --------

def step_abstract(*, situation: str, model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool, log_full: bool, log_file: Optional[Path]) -> Optional[str]:
    tpl = AG.load_prompt()
    prompt = AG.build_user_prompt(tpl, situation)
    raw = _run_with_retries(prompt, model, temperature, max_tokens, attempts, log_prefix="[abstract]", log_raw=log_raw, log_full=log_full, log_file=log_file)
    if not raw:
        return situation  # fallback to original
    try:
        obj = robust_json_loads(raw)
        if isinstance(obj, dict) and obj.get("abstract"):
            return str(obj["abstract"]).strip()
    except Exception as e:
        print(f"[abstract] JSON parse failed: {e}", file=sys.stderr)
    return situation


def step_factors(*, abstract_text: str, model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool, log_full: bool, log_file: Optional[Path]) -> List[Dict[str, Any]]:
    tpl = FG.load_prompt()
    prompt = FG.build_user_prompt(tpl, abstract_text)
    raw = _run_with_retries(prompt, model, temperature, max_tokens, attempts, log_prefix="[factors]", log_raw=log_raw, log_full=log_full, log_file=log_file)
    if raw:
        parsed = FG.parse_factor_block(raw)
        if parsed:
            return parsed
    # minimal fallback
    return [
        {"name": "importance", "description": "Importance of outcome", "possible_values": ["low", "high"]},
        {"name": "control", "description": "Perceived control", "possible_values": ["low", "high"]},
        {"name": "consequences", "description": "Severity of outcomes", "possible_values": ["mild", "severe"]},
    ]


def step_select(*, situation: str, factors: List[Dict[str, Any]], model: str, temperature: float, max_tokens: int, attempts: int, log_raw: bool, log_full: bool, log_file: Optional[Path]) -> List[Dict[str, str]]:
    tpl = FVS.load_prompt()
    prompt = FVS.build_user_prompt(tpl, situation, factors)
    raw = _run_with_retries(prompt, model, temperature, max_tokens, attempts, log_prefix="[select]", log_raw=log_raw, log_full=log_full, log_file=log_file)
    if raw:
        parsed = FVS.parse_selection_block(raw)
        if parsed:
            return parsed
    return []


# -------- Strict JSON prompts for EA/EU --------

def build_ea_json_prompt(
    *,
    scenario: str,
    subject: str,
    choices: List[str],
    abstract: Optional[str],
    selections: List[Dict[str, str]],
) -> str:
    choice_lines = [f"{chr(97+i)}) {c}" for i, c in enumerate(choices)]
    selected_pairs = []
    for it in selections or []:
        n = str(it.get("name", "")).strip()
        v = str(it.get("value", "")).strip()
        if n and v:
            selected_pairs.append(f"{n}={v}")
    selected_summary = ", ".join(selected_pairs) if selected_pairs else "(none)"

    system = (
        "You are an empathetic decision-maker. Select exactly one option that is most empathetic, supportive, and safe. "
        "Consider emotional wellbeing, de-escalation, and respect."
    )
    user = f"""
Scenario: {scenario}
Subject: {subject}

Choices (select exactly one):
{os.linesep.join(choice_lines)}

Context:
- Abstract summary: {abstract or '(missing)'}
- Selected factors: {selected_summary}

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


def build_eu_json_prompt(
    *,
    scenario: str,
    subject: str,
    emotion_choices: List[str],
    cause_choices: List[str],
    abstract: Optional[str],
    selections: List[Dict[str, str]],
) -> str:
    emo_block = "- " + "\n- ".join(str(c) for c in emotion_choices)
    cause_block = "- " + "\n- ".join(str(c) for c in cause_choices)
    selected_pairs = []
    for it in selections or []:
        n = str(it.get("name", "")).strip()
        v = str(it.get("value", "")).strip()
        if n and v:
            selected_pairs.append(f"{n}={v}")
    selected_summary = ", ".join(selected_pairs) if selected_pairs else "(none)"

    prompt = f"""
You are EmoBIRD. Read the scenario and choose exactly ONE emotion and ONE cause from the provided choices.
Return STRICT JSON only (no markdown, no code fences) with keys 'emo_label' and 'cause_label'.
The values MUST be verbatim copies of one option from the respective choices. Do not invent new options.

Subject: {subject}
Scenario: {scenario}

Context:
- Abstract summary: {abstract or '(missing)'}
- Selected factors: {selected_summary}

Emotion choices:
{emo_block}

Cause choices:
{cause_block}

Output format strictly: {{"emo_label": "<one of emotion choices>", "cause_label": "<one of cause choices>"}}
"""
    return prompt


# -------- Policy scoring via OpenRouter (optional) --------

def score_ea_policy_openrouter(
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    attempts: int,
    item: Dict[str, Any],
    pred_choice_text: str,
    gt_choice_text: str,
    raw_output: str,
    log_raw: bool,
    log_full: bool,
    log_file: Optional[Path],
) -> Tuple[float, str]:
    system_prompt = (
        "You are a strict but fair evaluator for empathetic multiple-choice decisions. "
        "Assess how close the model's chosen option is to the ground-truth option, considering empathy, appropriateness, and safety."
    )
    choices = item.get("choices", [])
    choice_lines = [f"{chr(97+i)}) {c}" for i, c in enumerate(choices)]
    user_prompt = f"""
Scenario: {item.get('scenario','')}
Subject: {item.get('subject','')}
Choices:\n{os.linesep.join(choice_lines)}

Ground truth choice text: {gt_choice_text}
Model predicted choice text: {pred_choice_text}
Model raw output: {raw_output}

Scoring rubric:
- 1.0 if prediction is effectively equivalent in meaning and quality to ground-truth.
- 0.6-0.9 if generally appropriate and empathetic but misses some key aspect.
- 0.3-0.5 if partially appropriate but with notable issues.
- 0.1-0.2 if weakly related or slightly inappropriate.
- 0.0 if unrelated, clearly inappropriate, or harmful.

Return only JSON: {{"score": <float 0..1>, "justification": "<brief>"}}
"""
    prompt = f"{system_prompt}\n\n{user_prompt}"
    raw = _run_with_retries(prompt, model, temperature, max_tokens, attempts, log_prefix="[policy_ea]", log_raw=log_raw, log_full=log_full, log_file=log_file)
    try:
        obj = robust_json_loads(raw)
        score = float(obj.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        just = str(obj.get("justification", ""))
        return score, just
    except Exception as e:
        return 0.0, f"Policy scoring parse failed: {e}"


def score_eu_policy_openrouter(
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    attempts: int,
    item: Dict[str, Any],
    pred_emo: str,
    pred_cause: str,
    gt_emo: str,
    gt_cause: str,
    raw_output: str,
    log_raw: bool,
    log_full: bool,
    log_file: Optional[Path],
) -> Tuple[float, str]:
    system_prompt = (
        "You are an evaluator for emotion understanding. Score the closeness between the predicted emotion/cause "
        "and the ground truth. Consider synonyms for emotions and reasonable paraphrases for causes."
    )
    user_prompt = f"""
Scenario: {item.get('scenario','')}
Subject: {item.get('subject','')}

Ground truth emotion: {gt_emo}
Predicted emotion: {pred_emo}

Ground truth cause: {gt_cause}
Predicted cause: {pred_cause}

Model raw output: {raw_output}

Scoring rubric (produce a single overall score 0..1):
- Start from emotion similarity (semantic equivalence and category closeness) as 60% of the score.
- Cause similarity (semantic overlap and relevance) as 40% of the score.
- 1.0 if both are essentially equivalent; 0.0 if both are unrelated.

Return only JSON: {{"score": <float 0..1>, "justification": "<brief>"}}
"""
    prompt = f"{system_prompt}\n\n{user_prompt}"
    raw = _run_with_retries(prompt, model, temperature, max_tokens, attempts, log_prefix="[policy_eu]", log_raw=log_raw, log_full=log_full, log_file=log_file)
    try:
        obj = robust_json_loads(raw)
        score = float(obj.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        just = str(obj.get("justification", ""))
        return score, just
    except Exception as e:
        return 0.0, f"Policy scoring parse failed: {e}"


# -------- Evaluation loops --------

def _coerce_choice_to_valid(letter: str, text: str, choices: List[str]) -> str:
    # Prefer exact letter mapping
    letter = (letter or "").strip().lower()
    if letter in ["a", "b", "c", "d"]:
        idx = ord(letter) - ord("a")
        if 0 <= idx < len(choices):
            return choices[idx]
    txt = (text or "").strip()
    # Exact case-insensitive match
    for c in choices:
        if txt.lower() == c.strip().lower():
            return c
    # Substring containment
    for c in choices:
        cl = c.strip().lower()
        if cl in txt.lower() or txt.lower() in cl:
            return c
    # Fallback to first choice
    return choices[0] if choices else ""


def evaluate_ea(
    *,
    data: List[Dict[str, Any]],
    model: str,
    temperature: float,
    abs_max_tokens: int,
    fac_max_tokens: int,
    sel_max_tokens: int,
    pred_max_tokens: int,
    attempts: int,
    do_policy: bool,
    policy_max_tokens: int,
    log_raw: bool,
    log_full: bool,
    print_raw: bool,
    log_file: Optional[Path],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    correct = 0

    for idx, item in enumerate(tqdm(data, desc="EA", unit="item")):
        scenario = str(item.get("scenario", "")).strip()
        subject = str(item.get("subject", "user")).strip()
        choices = list(item.get("choices", []))

        abstract_text = step_abstract(
            situation=scenario,
            model=model,
            temperature=temperature,
            max_tokens=abs_max_tokens,
            attempts=attempts,
            log_raw=log_raw,
            log_full=log_full,
            log_file=log_file,
        )
        factors = step_factors(
            abstract_text=abstract_text,
            model=model,
            temperature=temperature,
            max_tokens=fac_max_tokens,
            attempts=attempts,
            log_raw=log_raw,
            log_full=log_full,
            log_file=log_file,
        )
        selections = step_select(
            situation=scenario,
            factors=factors,
            model=model,
            temperature=temperature,
            max_tokens=sel_max_tokens,
            attempts=attempts,
            log_raw=log_raw,
            log_full=log_full,
            log_file=log_file,
        )

        p_prompt = build_ea_json_prompt(
            scenario=scenario,
            subject=subject,
            choices=choices,
            abstract=abstract_text,
            selections=selections,
        )
        p_raw = _run_with_retries(
            p_prompt, model, temperature=0.0, max_tokens=pred_max_tokens, attempts=attempts, log_prefix=f"[ea][{idx+1}]", log_raw=log_raw, log_full=log_full, log_file=log_file
        )
        if print_raw and p_raw:
            print(f"[ea][{idx+1}] raw prediction:\n{p_raw}\n", file=sys.stderr)
            _append_log(log_file, f"[ea][{idx+1}] raw prediction:\n{p_raw}\n")
        choice_letter = ""
        choice_text = ""
        try:
            obj = robust_json_loads(p_raw)
            if isinstance(obj, dict):
                choice_letter = str(obj.get("choice_letter", "")).strip()
                choice_text = str(obj.get("choice_text", "")).strip()
        except Exception as _e:
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
            "abstract": abstract_text,
            "factors": factors,
            "selections": selections,
            "raw_model_output": p_raw,
            "pred_choice_text": pred_choice,
            "gt_choice_text": gt_choice,
            "is_correct": is_correct,
        }

        if do_policy:
            try:
                if is_correct:
                    res["policy_score"] = 1.0
                    res["policy_justification"] = ""
                    res["final_score"] = 1.0
                else:
                    ps, pj = score_ea_policy_openrouter(
                        model=model,
                        temperature=temperature,
                        max_tokens=policy_max_tokens,
                        attempts=attempts,
                        item=item,
                        pred_choice_text=pred_choice,
                        gt_choice_text=gt_choice,
                        raw_output=p_raw,
                        log_raw=log_raw,
                        log_full=log_full,
                        log_file=log_file,
                    )
                    res["policy_score"] = float(ps)
                    res["policy_justification"] = pj
                    res["final_score"] = float(ps)
            except Exception as e:
                res["policy_score"] = 1.0 if is_correct else 0.0
                res["policy_justification"] = f"Policy scoring failed: {e}"
                res["final_score"] = res["policy_score"]

        results.append(res)

    accuracy = (correct / len(data)) if data else 0.0
    return {"items": results, "accuracy": accuracy}


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


def evaluate_eu(
    *,
    data: List[Dict[str, Any]],
    model: str,
    temperature: float,
    abs_max_tokens: int,
    fac_max_tokens: int,
    sel_max_tokens: int,
    pred_max_tokens: int,
    attempts: int,
    do_policy: bool,
    policy_max_tokens: int,
    log_raw: bool,
    log_full: bool,
    print_raw: bool,
    log_file: Optional[Path],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    correct = 0
    emo_correct = 0

    for idx, item in enumerate(tqdm(data, desc="EU", unit="item")):
        scenario = str(item.get("scenario", "")).strip()
        subject = str(item.get("subject", "user")).strip()
        emo_choices = list(item.get("emotion_choices", []))
        cause_choices = list(item.get("cause_choices", []))

        abstract_text = step_abstract(
            situation=scenario,
            model=model,
            temperature=temperature,
            max_tokens=abs_max_tokens,
            attempts=attempts,
            log_raw=log_raw,
            log_full=log_full,
            log_file=log_file,
        )
        factors = step_factors(
            abstract_text=abstract_text,
            model=model,
            temperature=temperature,
            max_tokens=fac_max_tokens,
            attempts=attempts,
            log_raw=log_raw,
            log_full=log_full,
            log_file=log_file,
        )
        selections = step_select(
            situation=scenario,
            factors=factors,
            model=model,
            temperature=temperature,
            max_tokens=sel_max_tokens,
            attempts=attempts,
            log_raw=log_raw,
            log_full=log_full,
            log_file=log_file,
        )

        p_prompt = build_eu_json_prompt(
            scenario=scenario,
            subject=subject,
            emotion_choices=emo_choices,
            cause_choices=cause_choices,
            abstract=abstract_text,
            selections=selections,
        )
        p_raw = _run_with_retries(
            p_prompt, model, temperature=0.0, max_tokens=pred_max_tokens, attempts=attempts, log_prefix=f"[eu][{idx+1}]", log_raw=log_raw, log_full=log_full, log_file=log_file
        )
        if print_raw and p_raw:
            print(f"[eu][{idx+1}] raw prediction:\n{p_raw}\n", file=sys.stderr)
            _append_log(log_file, f"[eu][{idx+1}] raw prediction:\n{p_raw}\n")
        pred_emo = ""
        pred_cause = ""
        try:
            obj = robust_json_loads(p_raw)
            if isinstance(obj, dict):
                pred_emo = str(obj.get("emo_label") or obj.get("emotion") or "").strip()
                pred_cause = str(obj.get("cause_label") or obj.get("cause") or "").strip()
        except Exception as _e:
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
            "abstract": abstract_text,
            "factors": factors,
            "selections": selections,
            "raw_model_output": p_raw,
            "pred_emo": pred_emo,
            "pred_cause": pred_cause,
            "gt_emo": gt_emo,
            "gt_cause": gt_cause,
            "is_correct": is_correct,
            "is_emo_correct": is_emo_correct,
        }

        if do_policy:
            try:
                if is_correct:
                    res["policy_score"] = 1.0
                    res["policy_justification"] = ""
                    res["final_score"] = 1.0
                else:
                    ps, pj = score_eu_policy_openrouter(
                        model=model,
                        temperature=temperature,
                        max_tokens=policy_max_tokens,
                        attempts=attempts,
                        item=item,
                        pred_emo=pred_emo,
                        pred_cause=pred_cause,
                        gt_emo=gt_emo,
                        gt_cause=gt_cause,
                        raw_output=p_raw,
                        log_raw=log_raw,
                        log_full=log_full,
                        log_file=log_file,
                    )
                    res["policy_score"] = float(ps)
                    res["policy_justification"] = pj
                    res["final_score"] = float(ps)
            except Exception as e:
                res["policy_score"] = 1.0 if is_correct else 0.0
                res["policy_justification"] = f"Policy scoring failed: {e}"
                res["final_score"] = res["policy_score"]

        results.append(res)

    n = len(data) if data else 0
    accuracy = (correct / n) if n else 0.0
    emo_accuracy = (emo_correct / n) if n else 0.0
    return {"items": results, "accuracy": accuracy, "emo_accuracy": emo_accuracy}


# -------- Main --------

def main() -> None:
    parser = argparse.ArgumentParser(description="EmoBench (EA & EU) evaluation using EmoBIRDv2 via OpenRouter")
    # Prefer user-preferred Llama model by default; allow env overrides
    default_model = os.environ.get("OPENROUTER_MODEL") or os.environ.get("REMOTE_MODEL") or "meta-llama/llama-3.1-8b-instruct"
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items per split")
    parser.add_argument("--model", type=str, default=default_model, help="OpenRouter model name")
    parser.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--abs-max-tokens", type=int, default=ABSTRACT_MAX_TOKENS)
    parser.add_argument("--fac-max-tokens", type=int, default=FACTOR_MAX_TOKENS)
    parser.add_argument("--sel-max-tokens", type=int, default=512)
    parser.add_argument("--pred-max-tokens", type=int, default=256, help="Max new tokens for strict JSON prediction")
    parser.add_argument("--policy-max-tokens", type=int, default=256, help="Max new tokens for policy scoring")
    parser.add_argument("--attempts", type=int, default=5, help="Retries per step")
    parser.add_argument("--no-policy", action="store_true", help="Disable policy-based closeness scoring")
    parser.add_argument("--log-raw", action="store_true", help="Print truncated raw model outputs to stderr")
    parser.add_argument("--only-ea", action="store_true", help="Run only EA split")
    parser.add_argument("--only-eu", action="store_true", help="Run only EU split")
    parser.add_argument("--raw-full", action="store_true", help="When logging raw outputs, print full text without truncation")
    parser.add_argument("--print-raw", action="store_true", help="Print raw prediction outputs per item to stderr")
    parser.add_argument("--log-file", type=str, default=None, help="Path to append raw outputs; defaults to results/emobench/emobirdv2_<run_id>.log")
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

    print(f"Running EmoBench via EmoBIRDv2: EA={len(ea_run) if run_ea else 0}, EU={len(eu_run) if run_eu else 0}", file=sys.stderr)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path: Optional[Path] = Path(args.log_file) if args.log_file else (OUT_DIR / f"emobirdv2_{run_id}.log")

    meta = {
        "run_id": run_id,
        "model": args.model,
        "temperature": args.temperature,
        "attempts": args.attempts,
    }

    # Evaluate EA
    if run_ea:
        ea_res = evaluate_ea(
            data=ea_run,
            model=args.model,
            temperature=args.temperature,
            abs_max_tokens=args.abs_max_tokens,
            fac_max_tokens=args.fac_max_tokens,
            sel_max_tokens=args.sel_max_tokens,
            pred_max_tokens=args.pred_max_tokens,
            attempts=args.attempts,
            do_policy=(not args.no_policy),
            policy_max_tokens=args.policy_max_tokens,
            log_raw=args.log_raw,
            log_full=args.raw_full,
            print_raw=args.print_raw,
            log_file=log_path,
        )
        with EA_OUT.open("w", encoding="utf-8") as f:
            json.dump({"meta": meta, **ea_res}, f, ensure_ascii=False, indent=2)
        print(f"EA results saved -> {EA_OUT}", file=sys.stderr)
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
            abs_max_tokens=args.abs_max_tokens,
            fac_max_tokens=args.fac_max_tokens,
            sel_max_tokens=args.sel_max_tokens,
            pred_max_tokens=args.pred_max_tokens,
            attempts=args.attempts,
            do_policy=(not args.no_policy),
            policy_max_tokens=args.policy_max_tokens,
            log_raw=args.log_raw,
            log_full=args.raw_full,
            print_raw=args.print_raw,
            log_file=log_path,
        )
        with EU_OUT.open("w", encoding="utf-8") as f:
            json.dump({"meta": meta, **eu_res}, f, ensure_ascii=False, indent=2)
        print(f"EU results saved -> {EU_OUT}", file=sys.stderr)
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
