#!/usr/bin/env python3
"""
Likert Matcher (EmoBIRDv2)

- Uses situation, identified factors, and a list of emotions
- Prompt template: prompts/likert_matching_prompt
- Parses lines primarily of the form: `emotion_name: rating`
- Robust to minor format drift (bullets, "emotion - rating", parentheses/comments)
- Maps rating to a probability using LIKERT_SCALE
- Returns a list of {emotion, rating, score}
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so `EmoBIRDv2` is importable when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from EmoBIRDv2.utils.constants import PROMPT_DIR, LIKERT_SCALE
from EmoBIRDv2.scripts.abstract_generator import call_openrouter
from EmoBIRDv2.scripts.factor_value_selector import format_factors_text


def load_prompt() -> str:
    path = os.path.join(PROMPT_DIR, "likert_matching_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_user_prompt(template: str, situation: str, factors: List[Dict[str, Any]], emotions: List[str]) -> str:
    # Render factors using existing formatter for consistency
    factors_text = format_factors_text(factors)
    # Format emotions as a newline-separated list
    emotions_block = "\n".join([e.strip() for e in emotions if str(e).strip()])
    return template.format(
        situation=situation.strip(),
        factors=factors_text,
        emotions_list=emotions_block,
    )


_ALLOWED_RATINGS = {k.lower() for k in LIKERT_SCALE.keys()}


def _normalize_rating(r: str) -> Optional[str]:
    """
    Normalize raw rating text to one of the LIKERT_SCALE keys.
    - Accepts variants like "very likely" -> "very-likely"
    - Trims trailing comments like "likely (0.75)" or after commas/semicolons
    - Case-insensitive
    """
    s = r.strip().lower()
    # Cut off after common separators or comment starters
    s = re.split(r"[|,;\\(]", s)[0].strip()
    # Remove surrounding quotes
    s = s.strip('"\'')
    # Collapse spaces
    s = re.sub(r"\s+", " ", s)
    s_h = s.replace(" ", "-")

    # Direct match first
    if s_h in _ALLOWED_RATINGS:
        return s_h

    # Common synonym/variant mapping
    synonyms = {
        "very likely": "very-likely",
        "very-likely": "very-likely",
        "likely": "likely",
        "somewhat likely": "likely",
        "somewhat-likely": "likely",
        "neutral/uncertain": "neutral",
        "uncertain": "neutral",
        "very unlikely": "very-unlikely",
        "very-unlikely": "very-unlikely",
        "unlikely": "unlikely",
    }
    if s in synonyms:
        return synonyms[s]
    if s_h in synonyms:
        return synonyms[s_h]

    return None


def parse_likert_lines(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse lines: `emotion_name: rating`
    - emotion_name: free text token; we lowercase and strip
    - rating: must be one of LIKERT_SCALE keys (case-insensitive)
    Output: list of {emotion, rating, score}
    """
    out: List[Dict[str, Any]] = []
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        s = line.strip()
        # Skip fenced code or obvious non-content lines
        if s.startswith("```"):
            continue
        # Drop leading bullets or numbering
        s = re.sub(r"^[-*]\s+", "", s)
        s = re.sub(r"^\d+[.)]\s*", "", s)

        # Accept either ':' or '-' as separator between emotion and rating
        parts = re.split(r"\s*:\s*|\s+-\s+", s, maxsplit=1)
        if len(parts) != 2:
            continue
        emotion_raw, rating_raw = parts[0], parts[1]
        emotion = emotion_raw.strip().strip("-*").strip().lower()
        rating_norm = _normalize_rating(rating_raw)
        if not rating_norm or rating_norm not in _ALLOWED_RATINGS:
            continue
        score = LIKERT_SCALE.get(rating_norm)
        out.append({
            "emotion": emotion,
            "rating": rating_norm,
            "score": score,
        })
    return out
