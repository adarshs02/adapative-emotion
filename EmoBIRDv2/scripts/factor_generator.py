#!/usr/bin/env python3
"""
Factor Generator (EmoBIRDv2)

- Uses the abstract to propose at least 3 psychological/social factors
- Prompt template: prompts/factor_generator_prompt.txt
- Parses lines until END_OF_FACTORS sentinel
- Outputs a structured JSON list of factors
"""

import os
import re
import sys
from typing import List, Dict, Any

# Ensure project root is on sys.path so `EmoBIRDv2` is importable when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from EmoBIRDv2.utils.constants import PROMPT_DIR
from EmoBIRDv2.scripts.abstract_generator import call_openrouter


def load_prompt() -> str:
    path = os.path.join(PROMPT_DIR, "factor_generator_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_user_prompt(template: str, abstract: str) -> str:
    return template.format(abstract=abstract.strip())


def parse_factor_block(raw_text: str) -> List[Dict[str, Any]]:
    """
    Extract factor lines preceding END_OF_FACTORS and parse entries of the form:
      1. Factor name: Description (value1/value2)
    Returns a list of dicts: {name, description, possible_values}
    """
    # Isolate lines before END_OF_FACTORS (case-insensitive)
    lines = raw_text.splitlines()
    out_lines: List[str] = []
    for line in lines:
        if line.strip().upper() == "END_OF_FACTORS":
            break
        if line.strip():
            out_lines.append(line.rstrip())

    factors: List[Dict[str, Any]] = []
    # Regex to capture: optional leading numbering, then name: desc (vals)
    pat = re.compile(r"^\s*(?:\d+\.\s*)?(?P<name>[^:()\n]+?)\s*:\s*(?P<desc>.*?)\s*\((?P<vals>[^)]+)\)\s*$")
    for line in out_lines:
        m = pat.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        desc = m.group("desc").strip()
        vals = [v.strip() for v in m.group("vals").split("/") if v.strip()]
        # Keep at most 2 values for now
        if len(vals) >= 2:
            vals = vals[:2]
        elif len(vals) == 1:
            # Duplicate if only one is found to ensure two options
            vals = [vals[0], vals[0]]
        else:
            continue
        factors.append({
            "name": name,
            "description": desc,
            "possible_values": vals,
        })

    # Return top 3 if more
    return factors[:3] if len(factors) > 3 else factors
