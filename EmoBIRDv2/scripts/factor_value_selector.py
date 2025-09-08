#!/usr/bin/env python3
"""
Factor Value Selector (EmoBIRDv2)

- Uses the full situation and a provided factor list to select a value per factor
- Prompt template: prompts/factor_value_selector_prompt.txt
- Parses lines until END_OF_ANALYSIS sentinel
- Outputs a structured list of selections
"""

import json
import os
import re
import sys
from typing import Any, Dict, List

# Ensure project root is on sys.path so `EmoBIRDv2` is importable when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from EmoBIRDv2.utils.constants import PROMPT_DIR


def load_prompt() -> str:
    path = os.path.join(PROMPT_DIR, "factor_value_selector_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


essential_keys = ("name", "description", "possible_values")

def format_factors_text(factors: List[Dict[str, Any]]) -> str:
    """
    Render factors into a compact text block consumed by the prompt.
    Each line: `1. Name: Description (v1/v2)`
    """
    lines: List[str] = []
    for i, f in enumerate(factors, start=1):
        if not all(k in f for k in essential_keys):
            # skip malformed entries
            continue
        name = str(f["name"]).strip()
        desc = str(f["description"]).strip()
        vals = f.get("possible_values") or []
        vals = [str(v).strip() for v in vals if str(v).strip()]
        if len(vals) < 2:
            # ensure two values for clarity
            if len(vals) == 1:
                vals = [vals[0], vals[0]]
            else:
                vals = ["low", "high"]
        vals_text = "/".join(vals[:2])
        lines.append(f"{i}. {name}: {desc} ({vals_text})")
    return "\n".join(lines)


def build_user_prompt(template: str, situation: str, factors: List[Dict[str, Any]]) -> str:
    return template.format(
        situation=situation.strip(),
        factors_text=format_factors_text(factors),
    )


def parse_selection_block(raw_text: str) -> List[Dict[str, str]]:
    """
    Parse lines of the form: `factor_name: chosen_value - brief explanation`
    Collect lines until END_OF_ANALYSIS.
    Returns a list of {name, value, explanation}.
    """
    lines = raw_text.splitlines()
    out: List[Dict[str, str]] = []
    for line in lines:
        if line.strip().upper() == "END_OF_ANALYSIS":
            break
        if not line.strip():
            continue
        m = re.match(r"^\s*(?P<name>[^:]+?)\s*:\s*(?P<value>[^-]+?)\s*-\s*(?P<exp>.+?)\s*$", line)
        if not m:
            continue
        out.append({
            "name": m.group("name").strip(),
            "value": m.group("value").strip(),
            "explanation": m.group("exp").strip(),
        })
    return out
