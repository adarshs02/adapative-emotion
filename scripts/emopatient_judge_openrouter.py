#!/usr/bin/env python3
"""
Empathy Judge via OpenRouter → JSON output (baseline vs RECAP, score_recap field)
=================================================================================

- Input JSON may be:
  (a) a list of items, or
  (b) an object with exactly one list value (e.g. {"qa":[...]}).
- For each item, compares two responses labeled to the judge as:
    baseline  (your baseline response)
    RECAP     (your renamed EmoBIRD response)
- Appends to each original item:
    winner, score_baseline, score_recap, margin, judge_remarks
- Writes a single JSON output that mirrors the input’s top-level shape.

Auth:
  export OPENROUTER_API_KEY="sk-..."

Example:
  python emopatient_judge_openrouter.py \
    --input /path/to/oss20b_combined.json \
    --output /path/to/empathy_results.json \
    --model openai/gpt-5

If autodetect fails, override fields:
  --field-question q --field-baseline baseline --field-emo RECAP
(You can also use --field-recap RECAP as an alias.)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

# ---------- Robust JSON extraction (defensive) ----------

_JSON_RE = re.compile(r'(\{.*\})', re.DOTALL)

def extract_first_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON object found in response.")
    frag = m.group(1)
    try:
        return json.loads(frag)
    except Exception as e:
        last = frag.rfind('}')
        if last != -1:
            try:
                return json.loads(frag[: last + 1])
            except Exception:
                pass
        raise ValueError(f"Could not parse JSON: {e}\nRaw (truncated): {text[:2000]}")

# ---------- Field autodetection helpers ----------

QUESTION_KEYS = ["question", "prompt", "q", "query", "user_question", "text"]
BASELINE_KEYS = ["baseline", "base", "control", "baseline_response"]
RECAP_KEYS    = ["recap", "emobird", "emo", "emo_response", "emobird_response", "treatment"]

def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}

def autodetect_fields(example: Dict[str, Any]) -> Tuple[str, str, str]:
    low = _lower_keys(example)
    q_key = next((k for k in QUESTION_KEYS if k in low), None)
    b_key = next((k for k in BASELINE_KEYS if k in low), None)
    r_key = next((k for k in RECAP_KEYS if k in low), None)
    if not (q_key and b_key and r_key):
        # Try nested shape: {question:..., responses:{baseline:..., recap:...}}
        for k, v in low.items():
            if isinstance(v, dict):
                vlow = _lower_keys(v)
                if not b_key:
                    b_key = next((k2 for k2 in BASELINE_KEYS if k2 in vlow), b_key)
                if not r_key:
                    r_key = next((k2 for k2 in RECAP_KEYS if k2 in vlow), r_key)
                if b_key and r_key and not q_key:
                    q_key = next((k2 for k2 in QUESTION_KEYS if k2 in low), q_key)
                if q_key and b_key and r_key:
                    b_key = f"{k}.{b_key}" if '.' not in b_key else b_key
                    r_key = f"{k}.{r_key}" if '.' not in r_key else r_key
                    return q_key, b_key, r_key
        raise KeyError(
            f"Autodetect failed. Example keys: {list(example.keys())}. "
            f"Expected {QUESTION_KEYS} and {BASELINE_KEYS}/{RECAP_KEYS} (possibly nested)."
        )
    return q_key, b_key, r_key

def get_nested(d: Dict[str, Any], dotted_key: str) -> Any:
    if '.' not in dotted_key:
        if dotted_key in d:
            return d[dotted_key]
        low = _lower_keys(d)
        return low.get(dotted_key.lower())
    cur = d
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict):
            return None
        if part in cur:
            cur = cur[part]
        else:
            low = _lower_keys(cur)
            if part in low:
                cur = low[part]
            else:
                return None
    return cur

def coerce_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)

# ---------- Judge prompt (baseline/RECAP labels) ----------

SYSTEM_PROMPT = """You are an expert in clinical communication and empathy.
Compare two responses labeled **baseline** and **RECAP** to the SAME patient question.
Score EMPATHY ONLY (not medical correctness).

Empathy rubric (0–10 overall):
1) Emotion naming/validation
2) Personalization (identity/goals/context)
3) Shared control (options, invites preferences)
4) Compassionate, actionable next steps
5) Warm, non-judgmental tone (no false hope)

Output STRICT JSON:
{
  "scores": { "baseline": 0-10, "RECAP": 0-10 },
  "winner": "baseline" | "RECAP" | "tie",
  "combined_remarks": "3–6 sentences explaining comparatively why the winner is more empathic; reference 2–3 rubric factors."
}
Return ONLY the JSON object.
"""

USER_TEMPLATE = """Question:
{question}

Response (baseline):
{resp_baseline}

Response (RECAP):
{resp_recap}

Instructions:
- Judge purely on EMPATHY using the rubric.
- Use the provided labels ("baseline", "RECAP") in your JSON keys.
- Return ONLY the JSON object requested.
"""

# ---------- OpenRouter call ----------

def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    json_mode: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 600,
    seed: Optional[int] = 7,
    base_url: str = "https://openrouter.ai/api/v1",
    connect_timeout: int = 20,
    read_timeout: int = 180,
    debug: bool = False,
    loose_json: bool = False,
    parse_retries: int = 3,
) -> Dict[str, Any]:
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if os.environ.get("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
    if os.environ.get("OPENROUTER_X_TITLE"):
        headers["X-Title"] = os.environ["OPENROUTER_X_TITLE"]

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if seed is not None:
        body["seed"] = seed

    parse_fail_count = 0
    for attempt in range(6):
        resp = requests.post(url, headers=headers, json=body, timeout=(connect_timeout, read_timeout))
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                if debug:
                    print("[DEBUG] Non-JSON HTTP 200 body:", resp.text[:2000], file=sys.stderr)
                raise
            if debug:
                try:
                    print("[DEBUG] RAW RESPONSE JSON:", json.dumps(data)[:2000], file=sys.stderr)
                except Exception:
                    pass
            message_obj = data.get("choices", [{}])[0].get("message", {})
            content = message_obj.get("content", "")
            refusal = message_obj.get("refusal")
            if debug and refusal:
                try:
                    print("[DEBUG] REFUSAL:", str(refusal)[:2000], file=sys.stderr)
                except Exception:
                    pass
            if debug:
                print("[DEBUG] RAW MESSAGE CONTENT:", (content or "<empty>")[:2000], file=sys.stderr)
            # Attempt 1: parse JSON from content
            try:
                if content:
                    return extract_first_json(content)
            except Exception as e:
                if debug:
                    print(f"[DEBUG] JSON parse failed from content: {e}", file=sys.stderr)
            # Attempt 2: some providers put text in 'reasoning'
            reasoning = message_obj.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                if debug:
                    print("[DEBUG] Trying to parse JSON from 'reasoning' field...", file=sys.stderr)
                try:
                    return extract_first_json(reasoning)
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] JSON parse failed from reasoning: {e}", file=sys.stderr)
            # Attempt 3: optional loose retry without response_format
            if loose_json:
                if debug:
                    print("[DEBUG] Retrying once without response_format (loose JSON mode)...", file=sys.stderr)
                body2 = dict(body)
                body2.pop("response_format", None)
                # Strengthen instruction inline for JSON-only output
                strengthened = (
                    user_content
                    + "\n\nReturn ONLY a compact JSON object with keys: 'scores' (with 'baseline' and 'RECAP' numeric 0-10), "
                      "'winner' (baseline|RECAP|tie), and 'combined_remarks' (string). No code fences, no prose."
                )
                body2["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": strengthened},
                ]
                resp2 = requests.post(url, headers=headers, json=body2, timeout=(connect_timeout, read_timeout))
                if resp2.status_code == 200:
                    try:
                        data2 = resp2.json()
                        if debug:
                            print("[DEBUG] RAW RESPONSE JSON (loose):", json.dumps(data2)[:2000], file=sys.stderr)
                        msg2 = data2.get("choices", [{}])[0].get("message", {})
                        content2 = msg2.get("content") or msg2.get("reasoning") or ""
                        if debug:
                            print("[DEBUG] RAW MESSAGE CONTENT (loose):", (content2 or "<empty>")[:2000], file=sys.stderr)
                        return extract_first_json(content2)
                    except Exception as e:
                        if debug:
                            print(f"[DEBUG] Loose parse failed: {e}", file=sys.stderr)
                else:
                    if debug:
                        try:
                            print("[DEBUG] Loose retry HTTP", resp2.status_code, "body:", resp2.text[:2000], file=sys.stderr)
                        except Exception:
                            pass
            # If all parse attempts fail, either retry (if under parse_retries) or raise
            parse_fail_count += 1
            if parse_fail_count < parse_retries:
                if debug:
                    try:
                        print(f"[DEBUG] No JSON found (parse attempt {parse_fail_count}/{parse_retries}). Retrying...", file=sys.stderr)
                    except Exception:
                        pass
                continue
            raise ValueError("No JSON object found in response.")
        elif resp.status_code in (429, 500, 502, 503, 504):
            retry_after = int(resp.headers.get("Retry-After", "0"))
            wait = max(retry_after, min(2 ** attempt, 30))
            time.sleep(wait)
            continue
        else:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            if debug:
                print(f"[DEBUG] HTTP {resp.status_code} error body: {str(err)[:2000]}", file=sys.stderr)
            raise RuntimeError(f"HTTP {resp.status_code}: {err}")
    raise RuntimeError("Exceeded retry attempts to OpenRouter")

# ---------- Input/output shaping ----------

def load_items_and_wrapper(path: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Returns (items, wrapper_key).
    - If input is a list: wrapper_key=None
    - If input is an object with exactly one list value: wrapper_key=<that key>
    - Special-case support for nested shape: {"scenarios": [{..., "qa": [ {q, baseline, RECAP, ...}, ... ]}, ...]}
      In this case we flatten all QA rows into a single list and return a wrapper dict
      so that we can reconstruct the original nested structure on write.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None
    list_keys = [k for k, v in data.items() if isinstance(v, list)]
    if len(list_keys) == 1:
        key = list_keys[0]
        lst = data[key]
        # Detect nested scenarios[].qa[] pattern and flatten
        if key.lower() == "scenarios" and all(isinstance(x, dict) for x in lst):
            has_qa = any(isinstance(s.get("qa"), list) for s in lst)
            if has_qa:
                flat: List[Dict[str, Any]] = []
                for s in lst:
                    qa_list = s.get("qa")
                    if isinstance(qa_list, list):
                        for qa in qa_list:
                            if isinstance(qa, dict):
                                flat.append(qa)
                wrapper_info = {"mode": "scenarios.qa", "root_key": key, "original": data}
                return flat, wrapper_info  # type: ignore[return-value]
        # Fallback: single list value at top-level
        return lst, key
    raise ValueError("Input must be a list OR an object with exactly one list value.")

def write_json_like_input(path: str, augmented_items: List[Dict[str, Any]], wrapper_key: Optional[str], pretty: bool = True) -> None:
    # Reconstruct original structure if we flattened scenarios[].qa[]
    if isinstance(wrapper_key, dict) and wrapper_key.get("mode") == "scenarios.qa":
        original = wrapper_key["original"]
        key = wrapper_key["root_key"]
        scenarios = original[key]
        idx = 0
        for s in scenarios:
            qa_list = s.get("qa")
            if isinstance(qa_list, list):
                new_qa = []
                for _ in qa_list:
                    if idx >= len(augmented_items):
                        break
                    new_qa.append(augmented_items[idx])
                    idx += 1
                s["qa"] = new_qa
        out_obj = original
    else:
        if wrapper_key is None:
            out_obj: Union[List[Dict[str, Any]], Dict[str, Any]] = augmented_items
        else:
            out_obj = {wrapper_key: augmented_items}  # type: ignore[index]

    # Ensure output directory exists
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        else:
            json.dump(out_obj, f, ensure_ascii=False, separators=(",", ":"))

# ---------- Main judging loop ----------

@dataclasses.dataclass
class JudgeAppend:
    score_baseline: float
    score_recap: float
    winner: str           # "baseline" | "recap" | "tie"
    margin: float
    judge_remarks: str

def resolve_fields(example: Dict[str, Any], q_field: Optional[str], b_field: Optional[str], r_field: Optional[str]) -> Tuple[str, str, str]:
    if q_field and b_field and r_field:
        return q_field, b_field, r_field
    auto_q, auto_b, auto_r = autodetect_fields(example)
    return q_field or auto_q, b_field or auto_b, r_field or auto_r

def coerce_item(verdict: Dict[str, Any]) -> JudgeAppend:
    # Accept preferred keys ("baseline","RECAP") and fall back to A/B if needed.
    scores_obj = verdict.get("scores", {})
    scores_lower = {str(k).lower(): v for k, v in scores_obj.items()}
    s_base  = scores_lower.get("baseline", scores_lower.get("a"))
    s_recap = scores_obj.get("RECAP") if "RECAP" in scores_obj else scores_lower.get("recap", scores_lower.get("b"))

    if s_base is None or s_recap is None:
        raise RuntimeError(f"Judge JSON missing required scores. Got: {verdict}")

    sA = float(s_base)
    sB = float(s_recap)

    win_raw = str(verdict.get("winner", "")).strip().lower()
    if win_raw not in ("baseline", "recap", "a", "b", "tie"):
        win_raw = "baseline" if sA > sB else "recap" if sB > sA else "tie"

    if win_raw in ("a", "baseline"):
        winner = "baseline"
    elif win_raw in ("b", "recap"):
        winner = "recap"
    else:
        winner = "tie"

    margin = round(abs(sA - sB), 3)
    remarks = verdict.get("combined_remarks", "").strip()
    return JudgeAppend(sA, sB, winner, margin, remarks)

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Empathy Judge → JSON output (baseline vs RECAP, score_recap field)")
    ap.add_argument("--input", default="/mnt/shared/adarsh/data/clinician_judge_datasets/oss20b_combined.json")
    ap.add_argument("--output", default="/mnt/shared/adarsh/datasets/EmoPatient/results/oss20b_judged.json")
    ap.add_argument("--model", default="openai/gpt-5", help="OpenRouter model (e.g., meta-llama/llama-3.1-8b-instruct)")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="OpenRouter base URL")
    ap.add_argument("--max-items", type=int, default=0, help="Limit items for a quick run (0 = all)")
    ap.add_argument("--field-question", type=str, default=None, help="Override: question field")
    ap.add_argument("--field-baseline", type=str, default=None, help="Override: baseline field")
    ap.add_argument("--field-emo", type=str, default=None, help="Override: RECAP field (legacy name kept for compatibility)")
    ap.add_argument("--field-recap", type=str, default=None, help="Alias: RECAP field")
    ap.add_argument("--seed", type=int, default=7, help="Seed (if supported)")
    ap.add_argument("--compact", action="store_true", help="Write compact JSON (no pretty indent)")
    ap.add_argument("--debug", action="store_true", help="Print raw API outputs and parse errors")
    ap.add_argument("--connect-timeout", type=int, default=20, help="HTTP connect timeout (seconds)")
    ap.add_argument("--read-timeout", type=int, default=180, help="HTTP read timeout (seconds)")
    ap.add_argument("--max-tokens", type=int, default=2046, help="Max tokens for judge response")
    ap.add_argument("--loose-json", action="store_true", help="Retry once without response_format and parse more loosely if JSON parse fails")
    ap.add_argument("--parse-retries", type=int, default=3, help="Number of parse retries before quitting when no JSON is found")
    args = ap.parse_args(argv)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: Please set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(2)

    items, wrapper_key = load_items_and_wrapper(args.input)
    if not items:
        print("No items found in input.", file=sys.stderr)
        sys.exit(1)

    # Choose recap field override precedence: --field-recap > --field-emo > autodetect
    recap_field_override = args.field_recap or args.field_emo
    qf, bf, rf = resolve_fields(items[0], args.field_question, args.field_baseline, recap_field_override)
    print(f"Using fields => question: '{qf}' | baseline: '{bf}' | recap: '{rf}'")

    augmented_items: List[Dict[str, Any]] = []
    limit = len(items) if args.max_items == 0 else min(args.max_items, len(items))

    for i in range(limit):
        row = items[i]
        q = coerce_str(get_nested(row, qf))
        baseline = coerce_str(get_nested(row, bf))
        recap    = coerce_str(get_nested(row, rf))

        if args.debug:
            try:
                print(
                    f"[DEBUG] Item {i+1}/{limit} lens => q:{len(q)} baseline:{len(baseline)} recap:{len(recap)}",
                    file=sys.stderr,
                )
            except Exception:
                pass

        user_msg = USER_TEMPLATE.format(
            question=q,
            resp_baseline=baseline,
            resp_recap=recap
        )
        verdict = call_openrouter(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_content=user_msg,
            json_mode=True,
            temperature=0.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
            base_url=args.base_url,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            debug=args.debug,
            loose_json=args.loose_json,
            parse_retries=args.parse_retries,
        )

        app = coerce_item(verdict)
        augmented = dict(row)  # keep ALL existing fields
        augmented["winner"] = app.winner
        augmented["score_baseline"] = app.score_baseline
        augmented["score_recap"] = app.score_recap
        augmented["margin"] = app.margin
        augmented["judge_remarks"] = app.judge_remarks

        augmented_items.append(augmented)

        if ((i + 1) % 10) == 0 or (i + 1) == limit:
            print(f"Scored {i + 1} / {limit}")

    write_json_like_input(args.output, augmented_items, wrapper_key, pretty=(not args.compact))
    print(f"Done. Wrote augmented JSON to {args.output}")

if __name__ == "__main__":
    main()
