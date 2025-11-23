#!/usr/bin/env python3
"""
Final Output Generator (EmoBIRDv2)

- Purpose: Produce a natural, empathetic final narrative that weaves together the situation
  and all intermediate insights. This is the LAST step of EmoBIRD; any downstream evaluation
  (e.g., SEC-EU) is external and not handled here.
- Loads the conversational prompt from prompts/final_output_prompt.txt
- Fills placeholders for {situation} (or {user_input}), {emotion_insights}, {context_info}
- Calls OpenRouter and returns the textual response
"""

import os
import re
import sys
from typing import Any, Dict, List, Tuple, Optional

# Ensure project root is on sys.path so `EmoBIRDv2` is importable when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from EmoBIRDv2.utils.constants import (
    PROMPT_DIR,
    LIKERT_SCALE,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    OUTPUT_MAX_TOKENS,
    OPENROUTER_API_KEY,
)
from EmoBIRDv2.scripts.abstract_generator import call_openrouter


def load_prompt(prompt_path: Optional[str] = None) -> str:
    # Conversational final output prompt (not SEC-EU). Must contain:
    # either {situation} or {user_input}, plus {emotion_insights} and {context_info}
    if prompt_path:
        path = prompt_path
    else:
        path = os.path.join(PROMPT_DIR, "final_output_prompt.txt")
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_selections(selections: List[Dict[str, str]]) -> str:
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


def _format_emotion_insights(likert_items: List[Dict[str, Any]]) -> str:
    """
    Turn Likert items into a compact insight string like:
    "anxiety (very-likely, 0.95), uncertainty (likely, 0.75), sadness (neutral, 0.50)"
    Sorted by descending LIKERT score.
    """
    if not likert_items:
        return "neutral"
    def _score(item: Dict[str, Any]) -> float:
        r = str(item.get("rating", "")).strip().lower().replace(" ", "-")
        return float(LIKERT_SCALE.get(r, LIKERT_SCALE["neutral"]))
    # Deduplicate by emotion (keep best rating)
    best: Dict[str, Dict[str, Any]] = {}
    for it in likert_items:
        em = str(it.get("emotion", "")).strip().lower()
        if not em:
            continue
        cur = best.get(em)
        if cur is None or _score(it) > _score(cur):
            best[em] = it
    ordered = sorted(best.values(), key=_score, reverse=True)
    def _fmt_score(v: Any) -> str:
        try:
            return f"{float(v):.2f}"
        except Exception:
            return "0.50"
    parts = [
        f"{it.get('emotion','').strip()} ({str(it.get('rating','')).strip()}, {_fmt_score(it.get('score'))})"
        for it in ordered
    ]
    return ", ".join([p for p in parts if p.strip()]) or "neutral"


def _build_context_info(*, abstract: Optional[str], selections: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    if abstract and abstract.strip():
        lines.append(abstract.strip())
    if selections:
        lines.append("\nKey factors (selected):")
        lines.append(_format_selections(selections))
    return "\n".join(lines).strip()


def build_user_prompt(template: str, *, situation: str, emotion_insights: str, context_info: str) -> str:
    # Normalize template to accept either {user_input} or {situation}
    # We replace {user_input} with {situation} so standard .format works.
    tpl = template.replace("{user_input}", "{situation}")
    return tpl.format(
        situation=situation.strip(),
        emotion_insights=emotion_insights.strip(),
        context_info=context_info.strip(),
    )


def generate_final_output(
    *,
    situation: str,
    abstract: Optional[str],
    selections: List[Dict[str, str]],
    likert_items: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    prompt_path: Optional[str] = None,
) -> str:
    """Build the final conversational output and call OpenRouter.

    - situation: original user situation text
    - abstract: concise summary (optional)
    - selections: factor selections (list of {name,value,explanation})
    - likert_items: list of {emotion, rating, score}
    - model/temperature/max_tokens: overrides; defaults from constants if None
    - prompt_path: optional path to a custom prompt file
    """
    template = load_prompt(prompt_path)
    emotion_insights = _format_emotion_insights(likert_items)
    context_info = _build_context_info(abstract=abstract, selections=selections)
    user_prompt = build_user_prompt(
        template,
        situation=situation,
        emotion_insights=emotion_insights,
        context_info=context_info,
    )

    api_key = OPENROUTER_API_KEY or ""
    mdl = model or MODEL_NAME
    temp = MODEL_TEMPERATURE if temperature is None else float(temperature)
    mtok = OUTPUT_MAX_TOKENS if max_tokens is None else int(max_tokens)
    return call_openrouter(user_prompt, api_key, mdl, temp, mtok)
