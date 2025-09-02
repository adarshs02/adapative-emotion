#!/usr/bin/env python3
"""
SECEU evaluation using OpenRouter (OpenAI-compatible) API only.
No EmoBIRD dependency, no local HF model load. Pure remote calls.

Environment variables:
- OPENROUTER_API_KEY: required unless --api-key is provided
- OPENROUTER_BASE_URL: optional (default: https://openrouter.ai/api/v1)
- REMOTE_MODEL or OPENROUTER_MODEL: optional if --model not provided

Example:
OPENROUTER_API_KEY=sk-... \
python scripts/seceu_eval_openrouter.py \
  --model meta-llama/llama-3.1-8b-instruct \
  --temperature 0.0 --iterations 1 --max-items 0
"""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import re

import numpy as np
from scipy.stats import pearsonr

# Visual progress bar (with safe fallback)
try:
    from tqdm.auto import tqdm
except Exception:
    class _TqdmFallback:
        def __call__(self, iterable, **kwargs):
            return iterable
        @staticmethod
        def write(msg: str):
            print(msg)
    tqdm = _TqdmFallback()

# Optional requests; fallback to urllib if missing
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None
    import urllib.request
    import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "seceu"
RESULTS_DIR = PROJECT_ROOT / "results" / "seceu"

# Default to Meta-Llama 3.1 8B Instruct per user preference
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_standard(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_scores_with_info(text: str, fallback: List[float] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract four numeric scores from model text robustly and normalize to sum 10.
    Strategy (improved):
    - Prefer the first 4 numbers immediately after the FIRST occurrence of 'Final Scores:' (case-insensitive).
    - Collect additional candidates: after each subsequent 'Final Scores:' marker, the FIRST 4 numbers; also the first 4 anywhere and the last 4 anywhere.
    - Avoid all-zero groups when a non-zero candidate exists.
    - If multiple candidates remain, choose the one with sum > 0 that is closest to 10; if none > 0, choose the one with the largest (non-negative) sum; else fallback.
    - Finally, clip negatives to 0 and rescale to make the sum exactly 10.
    Returns (scores_array, info_dict) where info includes method and raw numbers.
    """
    if fallback is None:
        fallback = [2.5, 2.5, 2.5, 2.5]
    info: Dict[str, Any] = {}
    try:
        float_pattern = re.compile(r"[-+]?\d*\.?\d+")

        # Gather raw numbers across the whole text
        raw_numbers_all = [float(x) for x in float_pattern.findall(text)]

        candidates: List[Tuple[str, List[float]]] = []

        # Candidates from 'Final Scores:' markers (take the FIRST 4 after each marker)
        markers = list(re.finditer(r"final\s*scores\s*:\s*", text, flags=re.IGNORECASE))
        for i, m in enumerate(markers):
            tail = text[m.end():]
            nums_after = [float(x) for x in float_pattern.findall(tail)]
            if len(nums_after) >= 4:
                label = "after_final_scores_first" if i == 0 else f"after_final_scores_{i+1}"
                candidates.append((label, nums_after[:4]))

        # First four anywhere and last four anywhere
        if len(raw_numbers_all) >= 4:
            candidates.append(("first_four_anywhere", raw_numbers_all[:4]))
            candidates.append(("last_four_anywhere", raw_numbers_all[-4:]))

        # If no candidates, fallback immediately
        if not candidates:
            arr = np.array(fallback, dtype=float)
            info.update({
                "method": "no_candidates_fallback",
                "raw_numbers": [],
                "chosen": fallback,
                "before_sum": float(arr.sum()),
                "normalized_sum": 10.0,
            })
            return arr, info

        # Remove all-zero candidates if any non-zero exists
        sums = [sum(vals) for _lab, vals in candidates]
        any_non_zero = any(s > 0 for s in sums)
        filtered = []
        for (lab, vals), s in zip(candidates, sums):
            if any_non_zero and s == 0:
                continue
            filtered.append((lab, vals))
        if filtered:
            candidates = filtered

        # Selection policy
        chosen_label = None
        chosen_numbers: List[float] = []

        # 1) Direct preference: first marker group if present and non-zero
        for lab, vals in candidates:
            if lab == "after_final_scores_first" and sum(vals) > 0:
                chosen_label, chosen_numbers = lab, vals
                break

        # 2) If not chosen, pick sum>0 closest to 10
        if not chosen_numbers:
            best = None
            best_lab = None
            for lab, vals in candidates:
                s = sum(vals)
                if s > 0:
                    diff = abs(s - 10.0)
                    if (best is None) or (diff < best[0]):
                        best = (diff, vals)
                        best_lab = lab
            if best is not None:
                chosen_label, chosen_numbers = best_lab, best[1]

        # 3) If still not chosen, pick the largest-sum candidate (could be zero)
        if not chosen_numbers:
            best = None
            best_lab = None
            for lab, vals in candidates:
                s = sum(vals)
                if (best is None) or (s > best[0]):
                    best = (s, vals)
                    best_lab = lab
            if best is not None:
                chosen_label, chosen_numbers = best_lab, best[1]

        # Final safety: if all failed for some reason
        if not chosen_numbers:
            arr = np.array(fallback, dtype=float)
            info.update({
                "method": "selection_fallback",
                "raw_numbers": raw_numbers_all,
                "chosen": fallback,
                "before_sum": float(arr.sum()),
                "normalized_sum": 10.0,
            })
            return arr, info

        # Clip negatives to 0, then rescale to sum 10
        arr = np.maximum(np.array(chosen_numbers, dtype=float), 0.0)
        s = float(arr.sum())
        if s <= 0:
            before_sum = 0.0
            arr = np.array(fallback, dtype=float)
            method = f"{chosen_label}_zero_sum_fallback"
        else:
            before_sum = s
            arr = arr / s * 10.0
            method = chosen_label

        info.update({
            "method": method,
            "raw_numbers": raw_numbers_all,
            "chosen": chosen_numbers,
            "before_sum": before_sum,
            "normalized_sum": float(arr.sum()),
        })
        return arr, info
    except Exception:
        arr = np.array(fallback, dtype=float)
        info.update({
            "method": "exception_fallback",
            "raw_numbers": [],
            "chosen": fallback,
            "before_sum": float(arr.sum()),
            "normalized_sum": 10.0,
        })
        return arr, info


def extract_scores(text: str, fallback: List[float] | None = None) -> np.ndarray:
    scores, _ = extract_scores_with_info(text, fallback=fallback)
    return scores


def build_prompt(story: str, options: List[str]) -> str:
    opts = ", ".join(f"({i+1}) {opt}" for i, opt in enumerate(options))
    return f'''Read the story and score each of the four emotion options.

Rules:
- Each score is between 0 and 10 (inclusive).
- The four scores must sum to exactly 10.
- You are an empathetic, supportive clinician. Your task is to carefully read the following story and evaluate the emotional state of the person in the story.

Story:
{story}

Emotion Options:
{opts}

Do not show your reasoning. Think silently and output only the final scores.
Output format: a single line with four numbers separated by spaces (e.g., 1.5 3.0 4.5 1.0).

Final Scores:'''

# 6.  Pay attention to user emotions and try to understand the user's emotional state before answering.

# -------------------- Remote OpenAI-compatible helpers --------------------

def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    if requests is not None:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:  # type: ignore[attr-defined]
            status = getattr(r, "status_code", "unknown")
            text = ""
            try:
                text = r.text
            except Exception:
                pass
            raise RuntimeError(f"HTTP {status} error: {text[:500]}") from e
    else:  # urllib fallback
        req = urllib.request.Request(url, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, data=body, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")


def _http_post_json_with_retries(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float, retries: int) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    attempts = max(0, int(retries)) + 1
    for attempt in range(attempts):
        try:
            return _http_post_json(url, headers, payload, timeout=timeout)
        except Exception as e:
            last_err = e
            if attempt < attempts - 1:
                # simple exponential backoff: 1s, 2s, 4s (cap at 8s)
                sleep_s = min(2.0 ** attempt, 8.0)
                try:
                    time.sleep(sleep_s)
                except Exception:
                    pass
            else:
                raise last_err


def _extract_message_text_from_openai_response(data: Dict[str, Any]) -> str:
    """Return assistant text from an OpenAI-compatible response.
    Tries, in order:
    - chat message.content
    - legacy choices[0].text
    - reasoning fields (message.reasoning or message.reasoning_details[*].text) as a last resort
    """
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        c0 = choices[0] or {}
        # Chat-style content
        msg = c0.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        # Legacy text completion
        txt = c0.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        # Reasoning fallback (may help if model returns only reasoning)
        reasoning = msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
        rdet = msg.get("reasoning_details")
        if isinstance(rdet, list):
            parts = [p.get("text", "") for p in rdet if isinstance(p, dict)]
            joined = "\n".join([p for p in parts if isinstance(p, str) and p.strip()])
            if joined.strip():
                return joined.strip()
        return ""
    except Exception:
        return ""

def _extract_text_from_responses_api(data: Dict[str, Any]) -> str:
    """Extract text from OpenRouter Responses API response.
    Looks for data['output'][i]['content'][j]['text'] fields.
    """
    try:
        out = data.get("output")
        if isinstance(out, list):
            texts: List[str] = []
            for block in out:
                content = block.get("content") if isinstance(block, dict) else None
                if isinstance(content, list):
                    for part in content:
                        txt = part.get("text") if isinstance(part, dict) else None
                        if isinstance(txt, str) and txt.strip():
                            texts.append(txt.strip())
            if texts:
                return "\n".join(texts)
        # Fallback if some providers nest differently
        cont = data.get("content")
        if isinstance(cont, list) and cont and isinstance(cont[0], dict):
            txt = cont[0].get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        return ""
    except Exception:
        return ""


def remote_chat_complete(base_url: str, api_key: str, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float = 120.0, retries: int = 0) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Optional OpenRouter headers (recommended)
    referer = os.environ.get("OPENROUTER_REFERER") or os.environ.get("HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    headers["X-Title"] = os.environ.get("OPENROUTER_TITLE", "SECEU Evaluation")
    messages = [
        {"role": "system", "content": "You are an empathetic assistant."},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": max(0.0, float(temperature)),
        "max_tokens": int(max_tokens),
    }
    data = _http_post_json_with_retries(url, headers, payload, timeout=timeout, retries=retries)
    text = _extract_message_text_from_openai_response(data)
    if text is None:
        text = ""
    try:
        # If still empty, raise with snippet for debugging
        if not str(text).strip():
            raise RuntimeError(f"Empty content in response; snippet: {json.dumps(data)[:300]}")
        return str(text).strip()
    except Exception as e:
        raise RuntimeError(f"Malformed response from OpenAI-compatible API: {e}; got: {json.dumps(data)[:500]}")


def remote_chat_complete_with_data(base_url: str, api_key: str, model: str, prompt: str, temperature: float, max_tokens: int, timeout: float = 120.0, retries: int = 0) -> Tuple[str, Dict[str, Any]]:
    """Same as remote_chat_complete, but also returns the raw JSON response."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_REFERER") or os.environ.get("HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    headers["X-Title"] = os.environ.get("OPENROUTER_TITLE", "SECEU Evaluation")
    messages = [
        {"role": "system", "content": "You are an empathetic assistant."},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": max(0.0, float(temperature)),
        "max_tokens": int(max_tokens),
    }
    data = _http_post_json_with_retries(url, headers, payload, timeout=timeout, retries=retries)
    text = _extract_message_text_from_openai_response(data)
    if text is None:
        text = ""
    return str(text).strip(), data

def remote_responses_complete_with_data(base_url: str, api_key: str, model: str, prompt: str, temperature: float, max_output_tokens: int, timeout: float = 120.0, retries: int = 0) -> Tuple[str, Dict[str, Any]]:
    """Call OpenRouter Responses API as a fallback when chat content is empty.
    This uses the unified responses endpoint and extracts text from the 'output' blocks.
    """
    url = base_url.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_REFERER") or os.environ.get("HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    headers["X-Title"] = os.environ.get("OPENROUTER_TITLE", "SECEU Evaluation")

    # Keep it simple: plain text input
    payload = {
        "model": model,
        "input": prompt,
        "temperature": max(0.0, float(temperature)),
        "max_output_tokens": int(max_output_tokens),
    }
    data = _http_post_json_with_retries(url, headers, payload, timeout=timeout, retries=retries)
    text = _extract_text_from_responses_api(data)
    if text is None:
        text = ""
    return str(text).strip(), data


# -------------------- CLI and main --------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SECEU with OpenRouter API (no EmoBIRD, no local HF)")
    p.add_argument("--items", type=str, default="", help="Path to seceu_items.json (defaults to data/seceu/seceu_items.json)")
    p.add_argument("--standard", type=str, default="", help="Path to seceu_standard.json (defaults to data/seceu/seceu_standard.json)")
    p.add_argument("--max-items", type=int, default=0, help="If >0, limit the number of items")
    p.add_argument("--iterations", type=int, default=1, help="Repeat full set and average distances")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (recommend 0.0)")
    p.add_argument("--max-tokens", type=int, default=256, help="Max tokens for the numeric reply (recommend >= 64 to avoid truncation)")

    # Remote OpenAI-compatible configuration
    p.add_argument("--model", default=None, help="Remote model name/id (env REMOTE_MODEL or OPENROUTER_MODEL if omitted)")
    p.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible API (env OPENROUTER_BASE_URL)")
    p.add_argument("--api-key", default=None, help="API key (env OPENROUTER_API_KEY)")
    p.add_argument("--http-timeout", type=float, default=120.0, help="HTTP timeout in seconds (default: 120)")
    p.add_argument("--http-retries", type=int, default=1, help="Number of retry attempts on transient failures (default: 1)")

    # Output control
    p.add_argument("--output-preds", type=str, default="", help="Path to save predictions JSON")
    p.add_argument("--output-results", type=str, default="", help="Path to save results JSON")
    p.add_argument("--save-raw", action="store_true", help="If set, save raw model outputs and parsing details alongside predictions")
    return p


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    default_items = PROJECT_ROOT / "data" / "seceu" / "seceu_items.json"
    default_std = PROJECT_ROOT / "data" / "seceu" / "seceu_standard.json"
    items_path = Path(args.items) if args.items else (default_items if default_items.exists() else (DATA_ROOT / "seceu_items.json"))
    std_path = Path(args.standard) if args.standard else (default_std if default_std.exists() else (DATA_ROOT / "seceu_standard.json"))
    if not items_path.exists():
        raise FileNotFoundError(f"Items file not found: {items_path}")
    if not std_path.exists():
        raise FileNotFoundError(f"Standard file not found: {std_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_out = Path(args.output_preds) if args.output_preds else (RESULTS_DIR / f"seceu_openrouter_predictions_{ts}.json")
    results_out = Path(args.output_results) if args.output_results else (RESULTS_DIR / f"seceu_openrouter_results_{ts}.json")
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.parent.mkdir(parents=True, exist_ok=True)
    return items_path, std_path, preds_out, results_out


def main() -> None:
    args = build_parser().parse_args()

    base_url = (
        args.base_url
        or os.environ.get("OPENROUTER_BASE_URL")
        or DEFAULT_BASE_URL
    )
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    model = (
        args.model
        or os.environ.get("REMOTE_MODEL")
        or os.environ.get("OPENROUTER_MODEL")
        or DEFAULT_MODEL
    )

    if not api_key:
        raise SystemExit("Missing API key. Provide --api-key or set OPENROUTER_API_KEY.")

    items_path, std_path, preds_out, results_out = resolve_paths(args)

    # Load data
    items = load_items(items_path)
    standard = load_standard(std_path)
    standard_scores = np.array(standard["standard_scores"], dtype=float)
    human_pattern = np.array(standard["human_pattern"], dtype=float)
    population_mean = float(standard["population_mean"])
    population_std = float(standard["population_std"])

    item_list = list(items)
    if args.max_items and args.max_items > 0:
        item_list = item_list[: args.max_items]

    all_iter_distances: List[np.ndarray] = []
    all_iter_preds: List[List[List[float]]] = []
    all_iter_raw: List[List[str]] = []
    all_iter_parse_info: List[List[Dict[str, Any]]] = []
    all_iter_api: List[List[Dict[str, Any]]] = []

    for it in range(1, args.iterations + 1):
        iter_preds: List[List[float]] = []
        iter_raw: List[str] = []
        iter_info: List[Dict[str, Any]] = []
        iter_api: List[Dict[str, Any]] = []
        distances_iter: List[float] = []
        for idx, item in enumerate(tqdm(item_list, desc=f"Processing items (iter {it})")):
            story = item["story"]
            options = item["options"]
            prompt = build_prompt(story, options)

            if args.save_raw:
                try:
                    completion, api_data = remote_chat_complete_with_data(
                        base_url, api_key, model, prompt, args.temperature, args.max_tokens,
                        timeout=args.http_timeout, retries=args.http_retries
                    )
                except Exception as e:
                    tqdm.write(f"❌ Error calling remote API on item {idx}: {e}")
                    completion = ""
                    api_data = {"error": str(e)}

                # Detect truncation via finish_reason == 'length'
                was_truncated = False
                try:
                    ch0 = (api_data.get("choices") or [None])[0] or {}
                    fin = ch0.get("finish_reason")
                    was_truncated = (str(fin).lower() == "length")
                except Exception:
                    was_truncated = False

                # Fallback to /responses if empty content or truncated
                if was_truncated or not str(completion).strip():
                    try:
                        fb_tokens = max(int(args.max_tokens) * 2, 64)
                        fb_text, fb_data = remote_responses_complete_with_data(
                            base_url, api_key, model, prompt, args.temperature, fb_tokens,
                            timeout=args.http_timeout, retries=args.http_retries
                        )
                        api_data = {
                            "primary": api_data,
                            "fallback": {
                                "used": True,
                                "endpoint": "/responses",
                                "reason": "truncated" if was_truncated else "empty",
                                "max_output_tokens": fb_tokens,
                                "data": fb_data,
                            },
                        }
                        completion = fb_text
                    except Exception as fe:
                        api_data = {
                            "primary": api_data,
                            "fallback": {
                                "used": True,
                                "endpoint": "/responses",
                                "reason": "truncated" if was_truncated else "empty",
                                "error": str(fe),
                            },
                        }

                iter_raw.append(completion)
                iter_api.append(api_data)
            else:
                try:
                    completion, api_data = remote_chat_complete_with_data(
                        base_url, api_key, model, prompt, args.temperature, args.max_tokens,
                        timeout=args.http_timeout, retries=args.http_retries
                    )
                except Exception as e:
                    tqdm.write(f" Error calling remote API on item {idx}: {e}")
                    completion = ""
                    api_data = {"error": str(e)}

                # Detect truncation via finish_reason == 'length'
                was_truncated = False
                try:
                    ch0 = (api_data.get("choices") or [None])[0] or {}
                    fin = ch0.get("finish_reason")
                    was_truncated = (str(fin).lower() == "length")
                except Exception:
                    was_truncated = False

                # Fallback to /responses if empty or truncated
                if was_truncated or not str(completion).strip():
                    try:
                        fb_tokens = max(int(args.max_tokens) * 2, 64)
                        fb_text, _fb_data = remote_responses_complete_with_data(
                            base_url, api_key, model, prompt, args.temperature, fb_tokens,
                            timeout=args.http_timeout, retries=args.http_retries
                        )
                        if str(fb_text).strip():
                            reason = "truncated" if was_truncated else "empty"
                            tqdm.write(f" Fallback /responses used on item {idx} due to {reason} (max_output_tokens={fb_tokens})")
                        completion = fb_text
                    except Exception as fe:
                        tqdm.write(f" Fallback /responses failed on item {idx}: {fe}")

            scores, info = extract_scores_with_info(completion)
            iter_preds.append(scores.tolist())
            if args.save_raw:
                iter_info.append(info)
            distances_iter.append(float(np.linalg.norm(scores - np.array(standard_scores[idx], dtype=float), ord=2)))

        all_iter_preds.append(iter_preds)
        all_iter_distances.append(np.array(distances_iter, dtype=float))
        if args.save_raw:
            all_iter_raw.append(iter_raw)
            all_iter_parse_info.append(iter_info)
            all_iter_api.append(iter_api)

    # Aggregate across iterations
    distances_mat = np.vstack(all_iter_distances)  # shape: (iterations, items)
    # Per-item mean and std across iterations
    distances = np.mean(distances_mat, axis=0)
    distances_std = np.std(distances_mat, axis=0, ddof=0)

    # Per-iteration SECEU and EQ scores
    seceu_scores_per_iter = np.mean(distances_mat, axis=1)
    seceu_score = float(np.mean(seceu_scores_per_iter))
    seceu_score_std = float(np.std(seceu_scores_per_iter, ddof=0)) if args.iterations > 1 else 0.0

    eq_scores_per_iter = 15.0 * ((population_mean - seceu_scores_per_iter) / population_std) + 100.0
    eq_score = float(np.mean(eq_scores_per_iter))
    eq_score_std = float(np.std(eq_scores_per_iter, ddof=0)) if args.iterations > 1 else 0.0

    # Pattern similarity (aggregate and per-iteration)
    hp_slice = human_pattern[: distances.shape[0]]
    if distances.shape[0] < 2:
        pattern_similarity = float("nan")
    else:
        pattern_similarity = float(pearsonr(distances, hp_slice)[0])

    pattern_similarity_per_iter: List[float] = []
    if distances_mat.shape[1] >= 2:
        for i in range(distances_mat.shape[0]):
            try:
                pattern_similarity_per_iter.append(float(pearsonr(distances_mat[i], hp_slice)[0]))
            except Exception:
                pattern_similarity_per_iter.append(float("nan"))
    else:
        pattern_similarity_per_iter = [float("nan")] * distances_mat.shape[0]
    # nan-aware std
    try:
        pattern_similarity_std = float(np.nanstd(np.array(pattern_similarity_per_iter, dtype=float))) if args.iterations > 1 else 0.0
    except Exception:
        pattern_similarity_std = float("nan")

    # Save predictions (keep only final iteration if single iteration, else save all)
    preds_payload: Dict[str, Any] = {"predictions": all_iter_preds[-1] if args.iterations == 1 else all_iter_preds}
    if args.save_raw:
        preds_payload["raw_outputs"] = all_iter_raw[-1] if args.iterations == 1 else all_iter_raw
        preds_payload["parse_info"] = all_iter_parse_info[-1] if args.iterations == 1 else all_iter_parse_info
        preds_payload["api_responses"] = all_iter_api[-1] if args.iterations == 1 else all_iter_api
    with preds_out.open("w", encoding="utf-8") as f:
        json.dump(preds_payload, f, indent=2)

    # Helper to safely round floats, preserving NaN as None for JSON compatibility
    def _safe_round(x: float, ndigits: int) -> Any:
        try:
            if not np.isfinite(x):
                return None
            return round(float(x), ndigits)
        except Exception:
            return None

    results = {
        "seceu_score": _safe_round(seceu_score, 3),
        "seceu_score_std": _safe_round(seceu_score_std, 3),
        "eq_score": _safe_round(eq_score, 2),
        "eq_score_std": _safe_round(eq_score_std, 2),
        "pattern_similarity": _safe_round(pattern_similarity, 3),
        "pattern_similarity_std": _safe_round(pattern_similarity_std, 3),
        "per_iteration": {
            "seceu_score": [ _safe_round(x, 3) for x in seceu_scores_per_iter.tolist() ],
            "eq_score": [ _safe_round(x, 2) for x in eq_scores_per_iter.tolist() ],
            "pattern_similarity": [ _safe_round(x, 3) for x in pattern_similarity_per_iter ],
        },
        "distances": {
            "mean": [ _safe_round(x, 3) for x in distances.tolist() ],
            "std": [ _safe_round(x, 3) for x in distances_std.tolist() ],
        },
        "meta": {
            "items_path": str(items_path),
            "standard_path": str(std_path),
            "iterations": args.iterations,
            "items_count": int(distances.shape[0]),
            "model_name": model,
            "sampling_temperature": args.temperature,
            "max_tokens": int(args.max_tokens),
            "base_url": base_url,
            "http_timeout": args.http_timeout,
            "http_retries": args.http_retries,
        },
    }
    with results_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ SECEU evaluation complete. SECEU Score: {results['seceu_score']}")
    print(f"📄 Predictions saved to: {preds_out}")
    print(f"📄 Results saved to: {results_out}")


if __name__ == "__main__":
    main()
