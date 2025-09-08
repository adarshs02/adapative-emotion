#!/usr/bin/env python3
"""
OpenRouter model interaction utility.

- Flexible, single-call CLI to query any OpenRouter model
- Reads input from --input, --input-file, or stdin
- Exposes sampling controls (model, temperature, max_tokens)
- Robust retry with wait/backoff/jitter
- Optional JSON-only mode and custom system prompt

Environment:
- OPENROUTER_API_KEY (required unless provided via --api-key override)
- OPENROUTER_MODEL or REMOTE_MODEL (optional default model)
- OPENROUTER_CONNECT_TIMEOUT / OPENROUTER_READ_TIMEOUT respected via EmoBIRDv2 constants
- Optional headers via env: OPENROUTER_REFERER/HTTP_REFERER and OPENROUTER_TITLE

Examples:
  echo "Hello" | python scripts/openapi_router.py \
    --model meta-llama/llama-3.1-8b-instruct --temperature 0.2 --max-tokens 256

  python scripts/openapi_router.py \
    --input "Return JSON with keys a and b." --json --attempts 5 --retry-backoff 1.6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root is importable for EmoBIRDv2 utilities
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse OpenRouter utilities and env constants
from EmoBIRDv2.utils.constants import OPENROUTER_API_KEY as ENV_API_KEY  # type: ignore
import EmoBIRDv2.scripts.abstract_generator as AG  # for call_openrouter and stdin helper


def _read_stdin() -> Optional[str]:
    try:
        if sys.stdin and not sys.stdin.isatty():
            data = sys.stdin.read()
            return data.strip() if data else None
    except Exception:
        pass
    return None


def _ensure_api_key(explicit_key: Optional[str]) -> str:
    api_key = (explicit_key or ENV_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required (set env or pass --api-key)")
    return api_key


def _run_with_retries(
    *,
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    attempts: int = 3,
    initial_wait: float = 1.0,
    backoff: float = 1.5,
    jitter: float = 0.5,
    system: Optional[str] = None,
    json_mode: bool = False,
    stream: bool = False,
    log_prefix: Optional[str] = None,
    log_raw: bool = False,
    log_full: bool = False,
    log_file: Optional[Path] = None,
) -> str:
    """Attempt call_openrouter multiple times, sleeping between attempts.
    Returns last non-empty response (or empty string if all attempts fail).
    """
    def _append_log(text: str) -> None:
        if not log_file:
            return
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(log_file).open("a", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass

    headers: Dict[str, str] = {}
    # Optional OpenRouter-recommended headers
    referer = os.environ.get("OPENROUTER_REFERER") or os.environ.get("HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_TITLE")
    if title:
        headers["X-Title"] = title

    response_format: Optional[Dict[str, Any]] = {"type": "json_object"} if json_mode else None

    wait_s = max(0.0, float(initial_wait))
    last: str = ""
    for i in range(1, int(max(1, attempts)) + 1):
        try:
            raw = AG.call_openrouter(
                prompt=prompt,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                system=system,
                extra_headers=headers if headers else None,
                extra_payload={"stream": bool(stream)},
            )
            if log_prefix and log_raw and raw is not None:
                msg = (
                    f"{log_prefix} Attempt {i}/{attempts} raw: {raw}"
                    if log_full
                    else f"{log_prefix} Attempt {i}/{attempts} raw: {(raw[:2000] + '...[truncated]') if len(raw) > 2000 else raw}"
                )
                print(msg, file=sys.stderr)
                _append_log(msg)
        except Exception as e:
            msg = f"{log_prefix or ''} Attempt {i}/{attempts} failed: {e}"
            print(msg, file=sys.stderr)
            _append_log(msg)
            raw = ""

        if raw:
            last = raw
            break

        if i < attempts:
            # sleep with backoff + jitter
            sleep_extra = random.uniform(0.0, float(jitter)) if jitter and float(jitter) > 0 else 0.0
            sleep_s = max(0.0, wait_s + sleep_extra)
            wait_msg = f"{log_prefix or ''} Waiting {sleep_s:.2f}s before retry {i+1}/{attempts}"
            print(wait_msg, file=sys.stderr)
            _append_log(wait_msg)
            try:
                time.sleep(sleep_s)
            except Exception:
                pass
            try:
                if backoff and float(backoff) > 0:
                    wait_s *= float(backoff)
            except Exception:
                pass

    return last


def build_parser() -> argparse.ArgumentParser:
    # Default model per user preference: Meta-Llama 3.1 8B Instruct
    default_model = (
        os.environ.get("OPENROUTER_MODEL")
        or os.environ.get("REMOTE_MODEL")
        or "meta-llama/llama-3.1-8b-instruct"
    )
    p = argparse.ArgumentParser(description="Query an OpenRouter model with retries")
    # Input
    p.add_argument("--input", type=str, default=None, help="Inline input text; if omitted, reads stdin unless --input-file is set")
    p.add_argument("--input-file", type=str, default=None, help="Path to a file containing the input text")
    p.add_argument("--system", type=str, default=None, help="Optional system prompt")

    # Sampling
    p.add_argument("--model", type=str, default=default_model, help="OpenRouter model name")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    p.add_argument("--json", action="store_true", help="Request strict JSON output (response_format=json_object)")
    p.add_argument("--stream", action="store_true", help="Set stream=True in payload (no client-side streaming)")

    # Auth
    p.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (overrides env OPENROUTER_API_KEY)")

    # Retries
    p.add_argument("--attempts", type=int, default=5, help="Number of attempts before giving up")
    p.add_argument("--retry-initial-wait", type=float, default=1.0, help="Initial wait seconds before first retry")
    p.add_argument("--retry-backoff", type=float, default=1.5, help="Multiplicative backoff per retry")
    p.add_argument("--retry-jitter", type=float, default=0.5, help="Add up to this many seconds random jitter each wait")

    # Logging
    p.add_argument("--log-raw", action="store_true", help="Log raw outputs per attempt to stderr (truncated)")
    p.add_argument("--raw-full", action="store_true", help="When logging raw outputs, print full text without truncation")
    p.add_argument("--log-file", type=str, default=None, help="Append logs to this file as well")
    return p


def _resolve_input(args: argparse.Namespace) -> str:
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            txt = f.read()
            txt = txt.strip()
            if not txt:
                raise SystemExit(f"Input file is empty: {args.input_file}")
            return txt
    if args.input is not None:
        s = str(args.input).strip()
        if s:
            return s
    stdin_txt = _read_stdin()
    if stdin_txt:
        return stdin_txt
    raise SystemExit("No input provided. Use --input, --input-file, or pipe via stdin.")


def main() -> None:
    args = build_parser().parse_args()

    api_key = _ensure_api_key(args.api_key)
    prompt = _resolve_input(args)

    log_file = Path(args.log_file) if getattr(args, "log_file", None) else None

    output = _run_with_retries(
        prompt=prompt,
        api_key=api_key,
        model=args.model,
        temperature=max(0.0, float(args.temperature)),
        max_tokens=int(args.max_tokens),
        attempts=int(args.attempts),
        initial_wait=float(args.retry_initial_wait),
        backoff=float(args.retry_backoff),
        jitter=float(args.retry_jitter),
        system=(args.system or None),
        json_mode=bool(args.json),
        stream=bool(args.stream),
        log_prefix="[openapi_router]",
        log_raw=bool(args.log_raw),
        log_full=bool(args.raw_full),
        log_file=log_file,
    )

    # Always print the final output to stdout
    if output is None:
        output = ""
    print(output)


if __name__ == "__main__":
    main()
