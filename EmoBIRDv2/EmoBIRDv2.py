#!/usr/bin/env python3
"""
EmoBIRDv2 main orchestrator.

Invokes subparts from a single entry point.
Currently supports: abstract generation via OpenRouter.

Examples:
  python EmoBIRDv2/EmoBIRDv2.py abstract --situation "I missed a deadline..."
  python EmoBIRDv2/EmoBIRDv2.py abstract --input-file path/to/situation.txt
  echo "I missed a deadline..." | python EmoBIRDv2/EmoBIRDv2.py abstract
"""

import argparse
import json
import os
import sys
import threading
from typing import Optional

# Ensure the parent of the package (repo root) is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
from EmoBIRDv2.scripts.abstract_generator import (
    load_prompt,
    build_user_prompt,
    call_openrouter,
    read_stdin,
)
from EmoBIRDv2.scripts import abstract_generator as ag
from EmoBIRDv2.utils.logger import get_logger
from EmoBIRDv2.utils.utils import robust_json_loads
from EmoBIRDv2.scripts import factor_generator as fg
from EmoBIRDv2.scripts import factor_value_selector as fvs
from EmoBIRDv2.scripts import emotion_generator as eg
from EmoBIRDv2.scripts import likert_matcher as lm
from EmoBIRDv2.scripts import final_output_generator as fog

# Per-run log file
logger = get_logger("EmoBIRDv2", per_run=True)


def _apply_timeout_overrides(args: argparse.Namespace) -> None:
    """Optionally override OpenRouter timeouts for this process."""
    try:
        if hasattr(args, "openrouter_connect_timeout") and args.openrouter_connect_timeout is not None:
            ag.OPENROUTER_CONNECT_TIMEOUT = int(args.openrouter_connect_timeout)
        if hasattr(args, "openrouter_read_timeout") and args.openrouter_read_timeout is not None:
            ag.OPENROUTER_READ_TIMEOUT = int(args.openrouter_read_timeout)
    except Exception:
        logger.exception("Failed to apply timeout overrides; continuing with defaults")


def cmd_abstract(args: argparse.Namespace) -> int:
    logger.info("Starting abstract generation")
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set in environment.")
        return 1

    situation: Optional[str] = None
    if args.situation:
        situation = args.situation
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            situation = f.read()
    else:
        situation = read_stdin()

    if not situation:
        logger.error("No situation provided. Use --situation, --input-file, or pipe via stdin.")
        return 2

    _apply_timeout_overrides(args)
    template = load_prompt()
    user_prompt = build_user_prompt(template, situation)

    attempts = 5  # increased retries to improve robustness
    for i in range(1, attempts + 1):
        try:
            logger.info(
                f"Attempt {i}/{attempts}: Calling OpenRouter model={args.model}, temp={args.temperature}, max_tokens={args.max_tokens}"
            )
            raw_text = call_openrouter(
                prompt=user_prompt,
                api_key=OPENROUTER_API_KEY,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if getattr(args, "log_raw", False):
                trunc = (raw_text[:2000] + "...[truncated]") if len(raw_text) > 2000 else raw_text
                logger.info(f"[abstract][raw] {trunc}")
        except Exception:
            logger.exception("OpenRouter request failed")
            raw_text = ""

        if not raw_text:
            logger.warning("Empty abstract content returned")
            continue

        # Parse JSON object using robust_json_loads
        try:
            obj = robust_json_loads(raw_text)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            obj = None

        if isinstance(obj, dict) and obj.get("abstract"):
            logger.info("Abstract generated successfully (JSON)")
            print(json.dumps(obj, ensure_ascii=False))
            return 0
        else:
            logger.warning("Parsed object missing 'abstract' or invalid; retrying if possible")

    print("abstract generation failed")
    return 4


def cmd_factors(args: argparse.Namespace) -> int:
    logger.info("Starting factor generation")
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set in environment.")
        return 1

    # Resolve abstract text from args or stdin (supports piping JSON from previous step)
    abstract_text: Optional[str] = None
    if getattr(args, "abstract", None):
        abstract_text = args.abstract
    elif getattr(args, "abstract_json", None):
        try:
            with open(args.abstract_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                abstract_text = data.get("abstract")
        except Exception:
            logger.exception("Failed to read --abstract-json file")
            return 2
    else:
        piped = read_stdin()
        if piped:
            # Try to parse JSON, otherwise treat as raw abstract text
            try:
                obj = robust_json_loads(piped)
            except Exception:
                try:
                    obj = json.loads(piped)
                except Exception:
                    obj = None
            if isinstance(obj, dict) and obj.get("abstract"):
                abstract_text = obj["abstract"]
            else:
                abstract_text = piped.strip()

    if not abstract_text:
        logger.error("No abstract provided. Use --abstract, --abstract-json, or pipe previous JSON output.")
        return 3

    _apply_timeout_overrides(args)
    template = fg.load_prompt()
    user_prompt = fg.build_user_prompt(template, abstract_text)

    attempts = 5
    for i in range(1, attempts + 1):
        try:
            logger.info(
                f"Attempt {i}/{attempts}: Calling OpenRouter model={args.model}, temp={args.temperature}, max_tokens={args.max_tokens}"
            )
            raw_text = call_openrouter(
                prompt=user_prompt,
                api_key=OPENROUTER_API_KEY,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if getattr(args, "log_raw", False):
                trunc = (raw_text[:2000] + "...[truncated]") if len(raw_text) > 2000 else raw_text
                logger.info(f"[factors][raw] {trunc}")
        except Exception:
            logger.exception("OpenRouter request failed")
            raw_text = ""

        if not raw_text:
            logger.warning("Empty factor content returned")
            continue

        factors = fg.parse_factor_block(raw_text)
        if factors:
            logger.info("Factors generated successfully")
            print(json.dumps({"factors": factors}, ensure_ascii=False))
            return 0
        else:
            logger.warning("No valid factor lines parsed; retrying if possible")

    print("factor generation failed")
    return 4


def cmd_select(args: argparse.Namespace) -> int:
    logger.info("Starting factor value selection")
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set in environment.")
        return 1

    # Resolve situation text
    situation: Optional[str] = None
    if getattr(args, "situation", None):
        situation = args.situation
    elif getattr(args, "input_file", None):
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                situation = f.read()
        except Exception:
            logger.exception("Failed to read --input-file")
            return 2
    else:
        # Only read situation from stdin if we are NOT consuming stdin for factors
        if not getattr(args, "factors_stdin", False):
            piped = read_stdin()
            if piped:
                situation = piped.strip()

    # Resolve factors JSON
    factors = None
    if getattr(args, "factors_json", None):
        try:
            with open(args.factors_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                factors = data.get("factors")
            elif isinstance(data, list):
                factors = data
        except Exception:
            logger.exception("Failed to read --factors-json file")
            return 3
    elif getattr(args, "factors_stdin", False):
        piped = read_stdin()
        if piped:
            try:
                obj = robust_json_loads(piped)
            except Exception:
                try:
                    obj = json.loads(piped)
                except Exception:
                    obj = None
            if isinstance(obj, dict) and obj.get("factors"):
                factors = obj["factors"]
            elif isinstance(obj, list):
                factors = obj

    if not situation:
        if getattr(args, "factors_stdin", False):
            logger.error("When using --factors-stdin, provide situation via --situation or --input-file.")
        else:
            logger.error("No situation text provided. Use --situation, --input-file, or pipe text/JSON.")
        return 4
    if not factors:
        logger.error("No factors provided. Use --factors-json or pipe JSON with a 'factors' list.")
        return 5

    _apply_timeout_overrides(args)
    template = fvs.load_prompt()
    user_prompt = fvs.build_user_prompt(template, situation, factors)

    attempts = 5
    for i in range(1, attempts + 1):
        try:
            logger.info(
                f"Attempt {i}/{attempts}: Calling OpenRouter model={args.model}, temp={args.temperature}, max_tokens={args.max_tokens}"
            )
            raw_text = call_openrouter(
                prompt=user_prompt,
                api_key=OPENROUTER_API_KEY,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if getattr(args, "log_raw", False):
                trunc = (raw_text[:2000] + "...[truncated]") if len(raw_text) > 2000 else raw_text
                logger.info(f"[select][raw] {trunc}")
        except Exception:
            logger.exception("OpenRouter request failed")
            raw_text = ""

        if not raw_text:
            logger.warning("Empty selection content returned")
            continue

        selections = fvs.parse_selection_block(raw_text)
        if selections:
            logger.info("Factor values selected successfully")
            print(json.dumps({"selections": selections}, ensure_ascii=False))
            return 0
        else:
            logger.warning("No valid selection lines parsed; retrying if possible")

    print("factor value selection failed")
    return 6


def cmd_emotions(args: argparse.Namespace) -> int:
    logger.info("Starting emotion extraction")
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set in environment.")
        return 1

    # Resolve situation text
    situation: Optional[str] = None
    if getattr(args, "situation", None):
        situation = args.situation
    elif getattr(args, "input_file", None):
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                situation = f.read()
        except Exception:
            logger.exception("Failed to read --input-file")
            return 2
    else:
        piped = read_stdin()
        if piped:
            situation = piped.strip()

    if not situation:
        logger.error("No situation text provided. Use --situation, --input-file, or pipe text.")
        return 3

    _apply_timeout_overrides(args)
    template = eg.load_prompt()
    user_prompt = eg.build_user_prompt(template, situation)

    attempts = 5
    for i in range(1, attempts + 1):
        try:
            logger.info(
                f"Attempt {i}/{attempts}: Calling OpenRouter model={args.model}, temp={args.temperature}, max_tokens={args.max_tokens}"
            )
            raw_text = call_openrouter(
                prompt=user_prompt,
                api_key=OPENROUTER_API_KEY,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if getattr(args, "log_raw", False):
                trunc = (raw_text[:2000] + "...[truncated]") if len(raw_text) > 2000 else raw_text
                logger.info(f"[emotions][raw] {trunc}")
        except Exception:
            logger.exception("OpenRouter request failed")
            raw_text = ""

        if not raw_text:
            logger.warning("Empty emotion content returned")
            continue

        emotions = eg.parse_emotion_lines(raw_text)
        if emotions and len(emotions) >= 3:
            logger.info("Emotions extracted successfully")
            print(json.dumps({"emotions": emotions}, ensure_ascii=False))
            return 0
        else:
            logger.warning("No valid emotions parsed; retrying if possible")

    print("emotion extraction failed")
    return 7


def cmd_likert(args: argparse.Namespace) -> int:
    logger.info("Starting likert matching")
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set in environment.")
        return 1

    # Resolve situation
    situation: Optional[str] = None
    if getattr(args, "situation", None):
        situation = args.situation
    elif getattr(args, "input_file", None):
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                situation = f.read()
        except Exception:
            logger.exception("Failed to read --input-file")
            return 2
    else:
        piped = read_stdin()
        if piped:
            situation = piped.strip()

    # Resolve factors JSON
    factors = None
    if getattr(args, "factors_json", None):
        try:
            with open(args.factors_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                factors = data.get("factors")
            elif isinstance(data, list):
                factors = data
        except Exception:
            logger.exception("Failed to read --factors-json file")
            return 3
    elif getattr(args, "factors_stdin", False):
        piped = read_stdin()
        if piped:
            try:
                obj = robust_json_loads(piped)
            except Exception:
                try:
                    obj = json.loads(piped)
                except Exception:
                    obj = None
            if isinstance(obj, dict) and obj.get("factors"):
                factors = obj["factors"]
            elif isinstance(obj, list):
                factors = obj

    # Resolve emotions list
    emotions = None
    if getattr(args, "emotions_json", None):
        try:
            with open(args.emotions_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                emotions = data.get("emotions")
            elif isinstance(data, list):
                emotions = data
        except Exception:
            logger.exception("Failed to read --emotions-json file")
            return 4
    elif getattr(args, "emotions_stdin", False):
        piped = read_stdin()
        if piped:
            try:
                obj = robust_json_loads(piped)
            except Exception:
                try:
                    obj = json.loads(piped)
                except Exception:
                    obj = None
            if isinstance(obj, dict) and obj.get("emotions"):
                emotions = obj["emotions"]
            elif isinstance(obj, list):
                emotions = obj

    if not situation:
        logger.error("No situation text provided. Use --situation, --input-file, or pipe text/JSON.")
        return 5
    if not factors:
        logger.error("No factors provided. Use --factors-json or --factors-stdin.")
        return 6
    if not emotions:
        logger.error("No emotions provided. Use --emotions-json or --emotions-stdin.")
        return 7

    _apply_timeout_overrides(args)
    template = lm.load_prompt()
    user_prompt = lm.build_user_prompt(template, situation, factors, emotions)

    attempts = 5
    for i in range(1, attempts + 1):
        try:
            logger.info(
                f"Attempt {i}/{attempts}: Calling OpenRouter model={args.model}, temp={args.temperature}, max_tokens={args.max_tokens}"
            )
            raw_text = call_openrouter(
                prompt=user_prompt,
                api_key=OPENROUTER_API_KEY,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            if getattr(args, "log_raw", False):
                trunc = (raw_text[:2000] + "...[truncated]") if len(raw_text) > 2000 else raw_text
                logger.info(f"[likert][raw] {trunc}")
        except Exception:
            logger.exception("OpenRouter request failed")
            raw_text = ""

        if not raw_text:
            logger.warning("Empty likert content returned")
            continue

        ratings = lm.parse_likert_lines(raw_text)
        if ratings:
            logger.info("Likert ratings parsed successfully")
            print(json.dumps({"likert": ratings}, ensure_ascii=False))
            return 0
        else:
            logger.warning("No valid likert lines parsed; retrying if possible")

    print("likert matching failed")
    return 8


def cmd_all(args: argparse.Namespace) -> int:
    if getattr(args, "with_emotions", False):
        logger.info("Starting full pipeline: abstract -> factors -> select -> emotions -> likert")
    else:
        logger.info("Starting full pipeline: abstract -> factors -> select")
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set in environment.")
        return 1

    # Resolve situation
    situation: Optional[str] = None
    if getattr(args, "situation", None):
        situation = args.situation
    elif getattr(args, "input_file", None):
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                situation = f.read()
        except Exception:
            logger.exception("Failed to read --input-file")
            return 2
    else:
        piped = read_stdin()
        if piped:
            situation = piped.strip()

    if not situation:
        logger.error("No situation provided. Use --situation, --input-file, or pipe text.")
        return 3

    _apply_timeout_overrides(args)
    # Emotions and likert will be run sequentially to avoid rate limits
    emotions = None
    likert = None

    # Step 1: Abstract
    template_abs = load_prompt()
    user_abs = build_user_prompt(template_abs, situation)
    abs_attempts = 5
    abstract_obj = None
    for i in range(1, abs_attempts + 1):
        try:
            logger.info(
                f"[abstract] Attempt {i}/{abs_attempts}: model={args.model}, temp={args.temperature}, max_tokens={args.abs_max_tokens}"
            )
            raw = call_openrouter(
                prompt=user_abs,
                api_key=OPENROUTER_API_KEY,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.abs_max_tokens,
            )
            if getattr(args, "log_raw", False):
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                logger.info(f"[abstract][raw] {trunc}")
        except Exception:
            logger.exception("OpenRouter request failed (abstract)")
            raw = ""
        if not raw:
            continue
        try:
            obj = robust_json_loads(raw)
        except Exception as e:
            logger.warning(f"[abstract] JSON parse failed: {e}")
            obj = None
        if isinstance(obj, dict) and obj.get("abstract"):
            abstract_obj = obj
            break
    if not abstract_obj:
        print("pipeline failed at abstract")
        return 10

    # Step 2: Factors (from abstract)
    abstract_text = abstract_obj["abstract"]
    tmpl_fac = fg.load_prompt()
    user_fac = fg.build_user_prompt(tmpl_fac, abstract_text)
    fac_attempts = 5
    factors = None
    for i in range(1, fac_attempts + 1):
        try:
            logger.info(
                f"[factors] Attempt {i}/{fac_attempts}: model={args.model}, temp={args.temperature}, max_tokens={args.fac_max_tokens}"
            )
            raw = call_openrouter(
                prompt=user_fac,
                api_key=OPENROUTER_API_KEY,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.fac_max_tokens,
            )
            if getattr(args, "log_raw", False):
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                logger.info(f"[factors][raw] {trunc}")
        except Exception:
            logger.exception("OpenRouter request failed (factors)")
            raw = ""
        if not raw:
            continue
        parsed = fg.parse_factor_block(raw)
        if parsed:
            factors = parsed
            break
    if not factors:
        print("pipeline failed at factors")
        return 11

    # Step 3: Select (with full situation + factors)
    tmpl_sel = fvs.load_prompt()
    user_sel = fvs.build_user_prompt(tmpl_sel, situation, factors)
    sel_attempts = 5
    selections = None
    for i in range(1, sel_attempts + 1):
        try:
            logger.info(
                f"[select] Attempt {i}/{sel_attempts}: model={args.model}, temp={args.temperature}, max_tokens={args.sel_max_tokens}"
            )
            raw = call_openrouter(
                prompt=user_sel,
                api_key=OPENROUTER_API_KEY,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.sel_max_tokens,
            )
            if getattr(args, "log_raw", False):
                trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                logger.info(f"[select][raw] {trunc}")
        except Exception:
            logger.exception("OpenRouter request failed (select)")
            raw = ""
        if not raw:
            continue
        parsed = fvs.parse_selection_block(raw)
        if parsed:
            selections = parsed
            break
    if not selections:
        print("pipeline failed at select")
        return 12

    # Step 4: Emotions (sequential, optional)
    if getattr(args, "with_emotions", False):
        try:
            tmpl_emo = eg.load_prompt()
            usr_emo = eg.build_user_prompt(tmpl_emo, situation)
            attempts = 5
            for j in range(1, attempts + 1):
                try:
                    logger.info(
                        f"[emotions] Attempt {j}/{attempts}: model={args.model}, temp={args.temperature}, max_tokens={args.emo_max_tokens}"
                    )
                    raw = call_openrouter(
                        prompt=usr_emo,
                        api_key=OPENROUTER_API_KEY,
                        model=args.model,
                        temperature=args.temperature,
                        max_tokens=args.emo_max_tokens,
                    )
                    if getattr(args, "log_raw", False):
                        trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                        logger.info(f"[emotions][raw] {trunc}")
                except Exception:
                    logger.exception("OpenRouter request failed (emotions)")
                    raw = ""
                if not raw:
                    continue
                parsed = eg.parse_emotion_lines(raw)
                if parsed and len(parsed) >= 3:
                    emotions = parsed
                    break
        except Exception:
            logger.exception("Emotion extraction failed")

    # Step 5: Likert matching (requires emotions)
    if emotions:
        try:
            tmpl_lik = lm.load_prompt()
            usr_lik = lm.build_user_prompt(tmpl_lik, situation, factors, emotions)
            attempts = 5
            for k in range(1, attempts + 1):
                try:
                    logger.info(
                        f"[likert] Attempt {k}/{attempts}: model={args.model}, temp={args.temperature}, max_tokens={args.likert_max_tokens}"
                    )
                    raw = call_openrouter(
                        prompt=usr_lik,
                        api_key=OPENROUTER_API_KEY,
                        model=args.model,
                        temperature=args.temperature,
                        max_tokens=args.likert_max_tokens,
                    )
                    if getattr(args, "log_raw", False):
                        trunc = (raw[:2000] + "...[truncated]") if len(raw) > 2000 else raw
                        logger.info(f"[likert][raw] {trunc}")
                except Exception:
                    logger.exception("OpenRouter request failed (likert)")
                    raw = ""
                if not raw:
                    continue
                parsed = lm.parse_likert_lines(raw)
                if parsed:
                    likert = parsed
                    break
        except Exception:
            logger.exception("Likert matching failed")

    # Emit combined JSON, and generate final conversational output if we have Likert ratings
    out = {
        "abstract": abstract_obj["abstract"],
        "factors": factors,
        "selections": selections,
    }
    if getattr(args, "with_emotions", False) and emotions:
        out["emotions"] = emotions
    if likert:
        out["likert"] = likert

    # Final output generation (after Likert)
    final_text = None
    try:
        if likert:
            final_text = fog.generate_final_output(
                situation=situation,
                abstract=abstract_obj["abstract"],
                selections=selections,
                likert_items=likert,
                model=args.model,
                temperature=args.temperature,
                max_tokens=OUTPUT_MAX_TOKENS,
            )
    except Exception:
        logger.exception("Final output generation failed")

    if final_text:
        out["final_output"] = final_text

    print(json.dumps(out, ensure_ascii=False))
    return 0

def main():
    parser = argparse.ArgumentParser(description="EmoBIRDv2 Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_abs = subparsers.add_parser("abstract", help="Generate an abstract from a situation")
    p_abs.add_argument("--situation", type=str, default=None, help="Situation text to summarize")
    p_abs.add_argument("--input-file", type=str, default=None, help="Path to file containing situation text")
    p_abs.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    p_abs.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    p_abs.add_argument("--max-tokens", type=int, default=ABSTRACT_MAX_TOKENS, help="Max new tokens for summary")
    p_abs.add_argument("--openrouter-connect-timeout", type=int, default=None, help="Connect timeout (s) to OpenRouter")
    p_abs.add_argument("--openrouter-read-timeout", type=int, default=None, help="Read timeout (s) to OpenRouter")
    p_abs.add_argument("--log-raw", action="store_true", help="Log raw model outputs for debugging")
    p_abs.set_defaults(func=cmd_abstract)

    p_fac = subparsers.add_parser("factors", help="Generate factors from an abstract")
    p_fac.add_argument("--abstract", type=str, default=None, help="Abstract text to analyze")
    p_fac.add_argument("--abstract-json", type=str, default=None, help="Path to JSON file with an 'abstract' field")
    p_fac.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    p_fac.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    p_fac.add_argument("--max-tokens", type=int, default=FACTOR_MAX_TOKENS, help="Max new tokens for factor list")
    p_fac.add_argument("--openrouter-connect-timeout", type=int, default=None, help="Connect timeout (s) to OpenRouter")
    p_fac.add_argument("--openrouter-read-timeout", type=int, default=None, help="Read timeout (s) to OpenRouter")
    p_fac.add_argument("--log-raw", action="store_true", help="Log raw model outputs for debugging")
    p_fac.set_defaults(func=cmd_factors)

    p_sel = subparsers.add_parser("select", help="Select factor values using full situation text")
    p_sel.add_argument("--situation", type=str, default=None, help="Full situation text")
    p_sel.add_argument("--input-file", type=str, default=None, help="Path to file containing situation text")
    p_sel.add_argument("--factors-json", type=str, default=None, help="Path to JSON with a 'factors' list")
    p_sel.add_argument("--factors-stdin", action="store_true", help="Read factors JSON from stdin")
    p_sel.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    p_sel.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    p_sel.add_argument("--max-tokens", type=int, default=FACTOR_MAX_TOKENS, help="Max new tokens for selections")
    p_sel.add_argument("--openrouter-connect-timeout", type=int, default=None, help="Connect timeout (s) to OpenRouter")
    p_sel.add_argument("--openrouter-read-timeout", type=int, default=None, help="Read timeout (s) to OpenRouter")
    p_sel.add_argument("--log-raw", action="store_true", help="Log raw model outputs for debugging")
    p_sel.set_defaults(func=cmd_select)

    p_emo = subparsers.add_parser("emotions", help="Extract 3-5 key emotions from the situation")
    p_emo.add_argument("--situation", type=str, default=None, help="Full situation text")
    p_emo.add_argument("--input-file", type=str, default=None, help="Path to file containing situation text")
    p_emo.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    p_emo.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    p_emo.add_argument("--max-tokens", type=int, default=EMOTION_MAX_TOKENS, help="Max new tokens for emotions step")
    p_emo.add_argument("--openrouter-connect-timeout", type=int, default=None, help="Connect timeout (s) to OpenRouter")
    p_emo.add_argument("--openrouter-read-timeout", type=int, default=None, help="Read timeout (s) to OpenRouter")
    p_emo.add_argument("--log-raw", action="store_true", help="Log raw model outputs for debugging")
    p_emo.set_defaults(func=cmd_emotions)

    p_lik = subparsers.add_parser("likert", help="Rate emotions on Likert scale using situation and factors")
    p_lik.add_argument("--situation", type=str, default=None, help="Full situation text")
    p_lik.add_argument("--input-file", type=str, default=None, help="Path to file containing situation text")
    p_lik.add_argument("--factors-json", type=str, default=None, help="Path to JSON with a 'factors' list")
    p_lik.add_argument("--factors-stdin", action="store_true", help="Read factors JSON from stdin")
    p_lik.add_argument("--emotions-json", type=str, default=None, help="Path to JSON with an 'emotions' list")
    p_lik.add_argument("--emotions-stdin", action="store_true", help="Read emotions list from stdin")
    p_lik.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    p_lik.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    p_lik.add_argument("--max-tokens", type=int, default=LIKERT_MAX_TOKENS, help="Max new tokens for likert step")
    p_lik.add_argument("--openrouter-connect-timeout", type=int, default=None, help="Connect timeout (s) to OpenRouter")
    p_lik.add_argument("--openrouter-read-timeout", type=int, default=None, help="Read timeout (s) to OpenRouter")
    p_lik.add_argument("--log-raw", action="store_true", help="Log raw model outputs for debugging")
    p_lik.set_defaults(func=cmd_likert)

    p_all = subparsers.add_parser("all", help="Run abstract -> factors -> select in one pipeline")
    p_all.add_argument("--situation", type=str, default=None, help="Full situation text")
    p_all.add_argument("--input-file", type=str, default=None, help="Path to file containing situation text")
    p_all.add_argument("--model", type=str, default=MODEL_NAME, help="OpenRouter model name")
    p_all.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE, help="Sampling temperature")
    p_all.add_argument("--abs-max-tokens", type=int, default=ABSTRACT_MAX_TOKENS, help="Max new tokens for abstract step")
    p_all.add_argument("--fac-max-tokens", type=int, default=FACTOR_MAX_TOKENS, help="Max new tokens for factors step")
    p_all.add_argument("--sel-max-tokens", type=int, default=FACTOR_MAX_TOKENS, help="Max new tokens for select step")
    p_all.add_argument("--with-emotions", dest="with_emotions", action="store_true", default=True, help="Also extract emotions in parallel and include in output (default: on)")
    p_all.add_argument("--no-emotions", dest="with_emotions", action="store_false", help="Disable emotion extraction in the all pipeline")
    p_all.add_argument("--emo-max-tokens", type=int, default=EMOTION_MAX_TOKENS, help="Max new tokens for emotions step")
    p_all.add_argument("--likert-max-tokens", type=int, default=LIKERT_MAX_TOKENS, help="Max new tokens for likert step")
    p_all.add_argument("--openrouter-connect-timeout", type=int, default=None, help="Connect timeout (s) to OpenRouter")
    p_all.add_argument("--openrouter-read-timeout", type=int, default=None, help="Read timeout (s) to OpenRouter")
    p_all.add_argument("--log-raw", action="store_true", help="Log raw model outputs for debugging")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
