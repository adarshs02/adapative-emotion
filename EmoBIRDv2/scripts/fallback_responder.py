#!/usr/bin/env python3
"""
Fallback Responder (EmoBIRDv2)

If any structured stage fails, generate a concise, empathetic response directly
from the base model using a dedicated prompt.

Reuses the OpenRouter call helper from abstract_generator.call_openrouter so it
inherits the same timeout behavior and headers.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# Ensure project root on PYTHONPATH when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from EmoBIRDv2.utils.constants import (
    PROMPT_DIR,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    OUTPUT_MAX_TOKENS,
    OPENROUTER_API_KEY,
)
from EmoBIRDv2.scripts.abstract_generator import call_openrouter as _call_openrouter


def load_prompt() -> str:
    path = os.path.join(PROMPT_DIR, "fallback_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_selections(selections: Optional[List[Dict[str, str]]]) -> str:
    if not selections:
        return ""
    lines: List[str] = []
    for sel in selections:
        name = str(sel.get("name", "")).strip()
        value = str(sel.get("value", "")).strip()
        exp = str(sel.get("explanation", "")).strip()
        if not name or not value:
            continue
        if exp:
            lines.append(f"- {name}: {value} ({exp})")
        else:
            lines.append(f"- {name}: {value}")
    return "\n".join(lines)


def _format_emotions(emotions: Optional[List[str]]) -> str:
    if not emotions:
        return ""
    return ", ".join([str(e).strip() for e in emotions if str(e).strip()])


def _format_likert(likert_items: Optional[List[Dict[str, Any]]]) -> str:
    if not likert_items:
        return ""
    parts: List[str] = []
    for it in likert_items:
        emo = str(it.get("emotion", "")).strip()
        rating = str(it.get("rating", "")).strip()
        score = it.get("score")
        if not emo:
            continue
        try:
            score_str = f"{float(score):.2f}" if score is not None else ""
        except Exception:
            score_str = ""
        if rating and score_str:
            parts.append(f"{emo} ({rating}, {score_str})")
        elif rating:
            parts.append(f"{emo} ({rating})")
        else:
            parts.append(emo)
    return ", ".join([p for p in parts if p.strip()])


def build_user_prompt(
    template: str,
    *,
    situation: str,
) -> str:
    # Situation-only fallback formatting
    formatted = template.format(
        situation=situation.strip(),
    )
    return formatted


def generate_fallback(
    *,
    situation: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate a concise, empathetic fallback response.

    Returns the best-effort string. Returns an empty string on error.
    """
    tpl = load_prompt()
    prompt = build_user_prompt(tpl, situation=situation)
    mdl = model or MODEL_NAME
    temp = MODEL_TEMPERATURE if temperature is None else float(temperature)
    mtok = OUTPUT_MAX_TOKENS if max_tokens is None else int(max_tokens)
    key = (api_key or OPENROUTER_API_KEY or "")
    try:
        content = _call_openrouter(prompt, key, mdl, temp, mtok)
        return (content or "").strip()
    except Exception:
        return ""
