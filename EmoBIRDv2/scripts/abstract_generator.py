#!/usr/bin/env python3
"""
Abstract Generator (EmoBIRDv2)

- Uses OpenRouter Chat Completions API
- Loads the abstract prompt from prompts/abstract_generator_prompt.txt
- Fills {situation} and requests a concise summary
"""

import json
import os
import sys
from typing import Optional, Dict, Any

import requests

# Ensure project root is on sys.path so `EmoBIRDv2` is importable when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from EmoBIRDv2.utils.constants import (
    OPENROUTER_API_KEY,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    PROMPT_DIR,
    OPENROUTER_CONNECT_TIMEOUT,
    OPENROUTER_READ_TIMEOUT,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_prompt() -> str:
    prompt_path = os.path.join(PROMPT_DIR, "abstract_generator_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_user_prompt(template: str, situation: str) -> str:
    return template.format(situation=situation.strip())


def call_openrouter(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    *,
    response_format: Optional[Dict[str, Any]] = None,
    system: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update({k: v for k, v in extra_headers.items() if v is not None})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if extra_payload:
        payload.update({k: v for k, v in extra_payload.items() if v is not None})

    resp = requests.post(
        OPENROUTER_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=(OPENROUTER_CONNECT_TIMEOUT, OPENROUTER_READ_TIMEOUT),
    )
    resp.raise_for_status()
    data = resp.json()
    # Surface API-declared errors even with 200 status
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"OpenRouter error: {data.get('error')}")
    try:
        content = (data["choices"][0]["message"].get("content") or "").strip()
        if content:
            return content
        # Fallback: use tool call function arguments if present
        tool_calls = data["choices"][0]["message"].get("tool_calls") or []
        if tool_calls:
            args = tool_calls[0].get("function", {}).get("arguments")
            return (args or "").strip()
        return ""
    except Exception:
        return ""


def read_stdin() -> Optional[str]:
    if sys.stdin and not sys.stdin.isatty():
        text = sys.stdin.read()
        return text.strip() if text else None
    return None
