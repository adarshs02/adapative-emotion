"""
Validation and cleaning utilities for EmoBIRD LLM pipeline.

- strip_noise: remove meta-notes, code fences, and trailing chatter
- extract_first_json: keep only first top-level JSON object
- parse_and_validate: parse JSON and validate with a Pydantic model
- clean_parse_validate: end-to-end cleaning and validation routine
"""
from __future__ import annotations
import json
import re
from typing import Type, Tuple
from pydantic import ValidationError


CODE_FENCE_RE = re.compile(r"```+\w*\n|```+", re.MULTILINE)

# Delimiters that often introduce meta content/trailing chatter
DELIMS = [
    "\n--",  # common markdown separators
    "\nNote:",
    "\nNotes:",
    "\nNOTE:",
    "\nEND",
    "\n###",
]


def strip_noise(text: str) -> str:
    """Remove obvious noise before parsing.

    Strategy: never truncate inside the JSON region.

    - If a '{' exists, only clean the prefix before the first '{' (remove code
      fences, comment/meta lines, and known delimiters). Leave the JSON suffix
      untouched to avoid cutting inside JSON strings.
    - If no '{' exists, perform light cleanup over the whole text.
    """
    if not text:
        return text

    start = text.find("{")
    if start == -1:
        # No JSON detected; light cleanup over entire text
        t = CODE_FENCE_RE.sub("", text)
        # Optionally truncate at delimiters when no JSON present
        cut_at = len(t)
        for d in DELIMS:
            idx = t.find(d)
            if idx != -1:
                cut_at = min(cut_at, idx)
        t = t[:cut_at]

        cleaned_lines = []
        for line in t.splitlines():
            l = line.strip()
            if not l:
                continue
            if l.startswith("#"):
                continue
            if l.startswith("Note:") or l.startswith("NOTE:") or l.startswith("Notes:"):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    # Clean only the prefix before the JSON starts
    prefix = text[:start]
    suffix = text[start:]

    prefix = CODE_FENCE_RE.sub("", prefix)
    cut_at = len(prefix)
    for d in DELIMS:
        idx = prefix.find(d)
        if idx != -1:
            cut_at = min(cut_at, idx)
    prefix = prefix[:cut_at]

    cleaned_prefix_lines = []
    for line in prefix.splitlines():
        l = line.strip()
        if not l:
            continue
        if l.startswith("#"):
            continue
        if l.startswith("Note:") or l.startswith("NOTE:") or l.startswith("Notes:"):
            continue
        cleaned_prefix_lines.append(line)
    cleaned_prefix = "\n".join(cleaned_prefix_lines).strip()

    return (cleaned_prefix + ("\n" if cleaned_prefix else "")) + suffix


def extract_first_json(text: str) -> str:
    """Return the first complete top-level JSON object from text.

    Raises ValueError if none is found or braces are unmatched.
    """
    if not text:
        raise ValueError("empty text")

    # Find first opening brace
    start = text.find("{")
    if start == -1:
        raise ValueError("no opening brace found")

    brace = 0
    end = None
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            # Characters inside strings (including braces) are ignored for brace matching
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    end = i + 1
                    break

    if end is None or brace != 0:
        raise ValueError("unmatched braces in JSON")

    candidate = text[start:end]
    # Validate that it's parseable JSON
    json.loads(candidate)
    return candidate


def parse_and_validate(json_text: str, model_cls: Type) -> Tuple[dict, object]:
    """Parse json_text and instantiate model_cls (Pydantic model).

    Returns (parsed_dict, model_instance). Raises on errors.
    """
    data = json.loads(json_text)
    model = model_cls.model_validate(data)
    return data, model


def clean_parse_validate(raw: str, model_cls: Type):
    """Convenience: strip noise → first JSON → model validate.
    Returns model instance. Raises on errors.
    """
    cleaned = strip_noise(raw)
    first = extract_first_json(cleaned)
    _, model = parse_and_validate(first, model_cls)
    return model
