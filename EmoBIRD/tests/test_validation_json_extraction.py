import json
import os
import sys
import pytest

# Add parent directory to path to import top-level modules like validation.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import strip_noise, extract_first_json


def test_extract_json_with_braces_in_string():
    raw = '{"text": "a { b } c", "x": 1}'
    cleaned = strip_noise(raw)
    first = extract_first_json(cleaned)
    data = json.loads(first)
    assert data["text"] == "a { b } c"
    assert data["x"] == 1


def test_extract_json_with_escaped_quotes_and_braces():
    raw = '{"t": "He said: \"{ ok }\"", "n": 2}'
    cleaned = strip_noise(raw)
    first = extract_first_json(cleaned)
    data = json.loads(first)
    assert data["t"] == 'He said: "{ ok }"'
    assert data["n"] == 2


def test_extract_json_from_code_fence_block():
    raw = """```json
{
  "t": "inside fence { }",
  "k": 3
}
```
extraneous text here
"""
    cleaned = strip_noise(raw)
    first = extract_first_json(cleaned)
    data = json.loads(first)
    assert data["t"] == "inside fence { }"
    assert data["k"] == 3


def test_extract_json_with_trailing_chatter():
    raw = '{"ok": true, "list": [1,2,3]}\nNote: model chatter after JSON should be ignored.'
    cleaned = strip_noise(raw)
    first = extract_first_json(cleaned)
    data = json.loads(first)
    assert data["ok"] is True
    assert data["list"] == [1, 2, 3]


def test_strip_noise_no_json_then_delims():
    raw = """```json
not actually json
```
--
header
{ "a": 1 } and more
"""
    # The function should not cut inside JSON; only prefix before first '{' is cleaned.
    cleaned = strip_noise(raw)
    assert cleaned.strip().startswith('{')
    first = extract_first_json(cleaned)
    data = json.loads(first)
    assert data == {"a": 1}


def test_extract_raises_on_unmatched_braces():
    bad = '{"a": {"b": 1}'  # missing closing brace
    cleaned = strip_noise(bad)
    with pytest.raises(ValueError):
        extract_first_json(cleaned)


def test_extract_raises_on_no_brace():
    bad = "no json here"
    cleaned = strip_noise(bad)
    with pytest.raises(ValueError):
        extract_first_json(cleaned)
