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
from typing import Optional

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


def call_openrouter(prompt: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        # Do not send stop tokens to avoid premature truncation
    }
    resp = requests.post(
        OPENROUTER_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=(OPENROUTER_CONNECT_TIMEOUT, OPENROUTER_READ_TIMEOUT),
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def read_stdin() -> Optional[str]:
    if sys.stdin and not sys.stdin.isatty():
        text = sys.stdin.read()
        return text.strip() if text else None
    return None
