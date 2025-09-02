#!/usr/bin/env python3
"""
Emotion Generator (EmoBIRDv2)

- Uses the full situation to extract 3-5 key emotions
- Prompt template: prompts/emotion_generator_prompt.txt
- Parses 3-5 lines, each a single lowercase emotion token
- Returns a list of emotion strings
"""

import os
import re
import sys
from typing import List

# Ensure project root is on sys.path so `EmoBIRDv2` is importable when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from EmoBIRDv2.utils.constants import PROMPT_DIR
from EmoBIRDv2.scripts.abstract_generator import call_openrouter


def load_prompt() -> str:
    path = os.path.join(PROMPT_DIR, "emotion_generator_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_user_prompt(template: str, situation: str) -> str:
    return template.format(situation=situation.strip())


def parse_emotion_lines(raw_text: str) -> List[str]:
    """
    Extract 3-5 emotion tokens from raw model output.
    Accept only lines that are a single lowercase word (allowing hyphens/spaces minimally),
    and stop after 5. Return at least 3 if available.
    """
    emotions: List[str] = []
    # Allow simple lowercase words with optional internal hyphens or spaces
    pat = re.compile(r"^[a-z][a-z\-\s]{0,30}$")
    for line in raw_text.splitlines():
        token = line.strip()
        if not token:
            continue
        token = token.lower()
        if pat.match(token):
            emotions.append(token)
            if len(emotions) >= 5:
                break
    # De-duplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for e in emotions:
        if e not in seen:
            deduped.append(e)
            seen.add(e)
    # Return 3-5 if possible
    if len(deduped) >= 3:
        return deduped[:5]
    return deduped
