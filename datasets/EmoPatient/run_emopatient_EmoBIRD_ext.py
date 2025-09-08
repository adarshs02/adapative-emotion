#!/usr/bin/env python3
"""
Extended EmoPatient runner (hardened):
- Mode 1 (default): runs EmoBIRD pipeline locally (vLLM)
- Mode 2: runs remote OpenAI-compatible (e.g., OpenRouter/Lambda) model via API key

What's new in this version:
- Robust HTTP JSON posting with retries/backoff for 429/5xx/timeouts
- Treats non-JSON bodies (empty/HTML/CF pages) as transient, with readable error snippets
- Optional small inter-call delay to be gentle on providers
- Accept header set explicitly; connection pooling via requests.Session()

Environment variables (used if flags omitted):
- EMOBIRD_MODEL: default local model for EmoBIRD (prefers Llama)
- OPENROUTER_API_KEY (preferred) or LAMBDA_API_KEY
- OPENROUTER_BASE_URL (preferred) or LAMBDA_API_BASE  e.g., https://openrouter.ai/api/v1
- OPENROUTER_CONNECT_TIMEOUT (default 20), OPENROUTER_READ_TIMEOUT (default 180)
- OPENROUTER_INTERCALL_DELAY_MS (default 250), OPENROUTER_MAX_RETRIES (default 4)
- OPENROUTER_TITLE, HTTP_REFERER (optional OpenRouter niceties)

Examples
- Local EmoBIRD (default):
  CUDA_VISIBLE_DEVICES=0 python run_emopatient_EmoBIRD_ext.py --provider emo

- Remote OpenRouter/OpenAI-compatible:
  OPENROUTER_API_KEY=... OPENROUTER_BASE_URL=https://openrouter.ai/api/v1 \
  python run_emopatient_EmoBIRD_ext.py --provider openai \
    --remote-model meta-llama/llama-3.1-8b-instruct \
    --openrouter-read-timeout 240 --max-retries 4 --inter-call-delay-ms 300
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Repo root on path
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional requests; fallback to urllib if missing
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None
    import urllib.request
    import urllib.error  # type: ignore

# Visual progress bar (with safe fallback)
try:
    from tqdm.auto import tqdm  # prefer rich display when available
except Exception:
    class _TqdmFallback:
        def __call__(self, iterable, **kwargs):
            return iterable
        @staticmethod
        def write(msg: str):
            print(msg)
    tqdm = _TqdmFallback()

# EmoBIRD imports
from EmoBIRD.emobird_poc import Emobird
from EmoBIRD.logger import EmobirdLogger, set_logger, get_logger, close_logger

DATASET_PATH = HERE.parent / "scenarios_30.json"
RESULTS_DIR = HERE.parent / "results"
LOGS_DIR = HERE.parent / "logs"
DEFAULT_REMOTE_MODEL = "openai/gpt-oss-20b"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run EmoPatient with EmoBIRD (local) or OpenAI-compatible remote API")
    p.add_argument("--provider", choices=["emo", "openai"], default="emo",
                   help="'emo' = EmoBIRD local vLLM, 'openai' = remote OpenAI-compatible API")

    # Remote OpenAI-compatible configuration
    p.add_argument("--remote-model", default=None, help="Remote model name/id")
    p.add_argument("--remote-base-url", default=None, help="Base URL for OpenAI-compatible API (env OPENROUTER_BASE_URL or LAMBDA_API_BASE)")
    p.add_argument("--api-key", default=None, help="API key (env OPENROUTER_API_KEY or LAMBDA_API_KEY)")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=220)
    # OpenRouter timeout controls
    p.add_argument("--openrouter-connect-timeout", type=int, default=None,
                   help="TCP connect timeout in seconds (env OPENROUTER_CONNECT_TIMEOUT, default 20)")
    p.add_argument("--openrouter-read-timeout", type=int, default=None,
                   help="Read/response timeout in seconds (env OPENROUTER_READ_TIMEOUT, default 180)")

    # NEW: polite pacing + retries
    p.add_argument("--inter-call-delay-ms", type=int,
                   default=int(os.environ.get("OPENROUTER_INTERCALL_DELAY_MS", "250")),
                   help="Sleep this many ms between remote calls")
    p.add_argument("--max-retries", type=int,
                   default=int(os.environ.get("OPENROUTER_MAX_RETRIES", "4")),
                   help="Max transient retries for 429/5xx/timeouts/parse errors")

    # Execution controls
    p.add_argument("--scenario-index", type=int, default=None,
                   help="Run only this scenario (0-based). If omitted, runs all.")
    p.add_argument("--scenarios", default=str(DATASET_PATH),
                   help="Path to scenarios JSON (default: datasets/EmoPatient/scenarios_30.json)")
    # Local EmoBIRD model override
    p.add_argument("--model", default=None,
                   help="Local EmoBIRD model name/id (env EMOBIRD_MODEL). Only used when --provider emo")
    return p


def load_scenarios(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data.get("scenarios") or data.get("scenarios_30") or []
    if not scenarios:
        raise ValueError("No scenarios found in dataset")
    return scenarios


def compose_situation_text(scn: Dict[str, Any], question: str) -> str:
    title = scn.get("title", "")
    diagnosis = scn.get("diagnosis", "")
    tplan = scn.get("treatment_plan", "")
    narrative = scn.get("narrative", "")
    situation = (
        f"Context (clinical scenario):\n"
        f"Title: {title}\n"
        f"Diagnosis: {diagnosis}\n"
        f"Treatment plan: {tplan}\n\n"
        f"Patient narrative:\n{narrative}\n\n"
        f"Question:\n{question}\n\n"
        f"TASK: You are an empathetic AI assistant. Please provide a concise, empathetic, medically grounded answer tailored to the above context."
    )
    return situation


# -------------------- Remote OpenAI-compatible helpers --------------------

_SESSION = None
def _session():
    """Shared requests session for connection pooling."""
    global _SESSION
    if _SESSION is None and requests is not None:
        _S = requests.Session()
        _S.headers.update({"Accept": "application/json"})
        _SESSION = _S
    return _SESSION

_TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 522, 524}


def _post_json_with_retries(url: str,
                            headers: Dict[str, str],
                            payload: Dict[str, Any],
                            *,
                            connect_timeout: int,
                            read_timeout: int,
                            max_retries: int) -> Dict[str, Any]:
    """
    POST JSON with robust error handling:
    - JSON encode (so requests sets proper headers)
    - Retries on 429/5xx/timeouts and on non-JSON bodies
    - Raises RuntimeError with a helpful snippet when giving up
    """
    last_err: Optional[str] = None
    sess = _session()

    for attempt in range(max_retries + 1):
        try:
            if requests is None:
                # urllib fallback
                req = urllib.request.Request(url, method="POST")  # type: ignore[attr-defined]
                for k, v in headers.items():
                    req.add_header(k, v)
                req.add_header("Content-Type", "application/json")
                req.add_header("Accept", "application/json")
                body = json.dumps(payload).encode("utf-8")
                try:
                    with urllib.request.urlopen(req, data=body, timeout=read_timeout) as resp:  # type: ignore[attr-defined]
                        raw = resp.read().decode("utf-8", errors="ignore")
                        try:
                            return json.loads(raw)
                        except Exception:
                            last_err = f"Non-JSON response (status {getattr(resp, 'status', '?')}). First 500 chars:\n{raw[:500]}"
                            # treat as transient and retry
                            raise RuntimeError(last_err)
                except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
                    status = getattr(e, "code", 0)
                    body = e.read().decode("utf-8", errors="ignore")
                    if status in _TRANSIENT_STATUSES and attempt < max_retries:
                        time.sleep(min(10.0, (2 ** attempt) * 0.5 + random.uniform(0, 0.25)))
                        continue
                    raise RuntimeError(f"HTTP {status} error:\n{body[:500]}") from e

            else:
                r = sess.post(url, headers=headers, json=payload,
                              timeout=(connect_timeout, read_timeout))
                status = r.status_code
                if status in _TRANSIENT_STATUSES:
                    retry_after = r.headers.get("Retry-After")
                    if attempt < max_retries:
                        sleep_s = float(retry_after) if (retry_after and retry_after.isdigit()) else min(
                            10.0, (2 ** attempt) * 0.5 + random.uniform(0, 0.25)
                        )
                        time.sleep(sleep_s)
                        continue
                    # fallthrough to raise below

                # Raise for non-transient HTTP client errors
                r.raise_for_status()

                # Parse JSON safely
                try:
                    return r.json()
                except Exception:
                    ct = r.headers.get("Content-Type", "")
                    txt = ""
                    try:
                        txt = r.text
                    except Exception:
                        pass
                    last_err = f"Non-JSON response (HTTP {status}, Content-Type={ct}). First 500 chars:\n{txt[:500]}"
                    if attempt < max_retries:
                        time.sleep(min(10.0, (2 ** attempt) * 0.5 + random.uniform(0, 0.25)))
                        continue
                    raise RuntimeError(last_err)

        except (requests.exceptions.Timeout if requests else tuple(),  # type: ignore
                requests.exceptions.ConnectionError if requests else tuple()) as e:  # type: ignore
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(min(10.0, (2 ** attempt) * 0.5 + random.uniform(0, 0.25)))
                continue
            raise RuntimeError(last_err) from e

        except RuntimeError as e:
            # If transient-like error and attempts remain, we already slept; else propagate
            if attempt >= max_retries:
                raise

            time.sleep(min(10.0, (2 ** attempt) * 0.5 + random.uniform(0, 0.25)))

    raise RuntimeError(last_err or "Unknown error during POST")


def _extract_message_text_from_openai_response(data: Dict[str, Any]) -> str:
    """Best-effort extraction of assistant text from OpenAI-compatible response."""
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
        # Reasoning fallback fields if present
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
    """Extract text from OpenRouter unified /responses response."""
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
        cont = data.get("content")
        if isinstance(cont, list) and cont and isinstance(cont[0], dict):
            txt = cont[0].get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        return ""
    except Exception:
        return ""


def remote_chat_complete(base_url: str,
                         api_key: str,
                         model: str,
                         prompt: str,
                         temperature: float,
                         max_tokens: int,
                         *,
                         connect_timeout: int,
                         read_timeout: int,
                         max_retries: int) -> str:
    """Call chat/completions with robust fallback to /responses when truncated/empty."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # Optional OpenRouter headers (recommended)
    referer = os.environ.get("OPENROUTER_REFERER") or os.environ.get("HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    headers["X-Title"] = os.environ.get("OPENROUTER_TITLE", "EmoPatient")

    messages = [
        {"role": "system", "content": "You are an empathetic, clinically careful assistant."},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": max(0.0, float(temperature)),
        "max_tokens": int(max_tokens),
    }

    data = _post_json_with_retries(url, headers, payload,
                                   connect_timeout=connect_timeout,
                                   read_timeout=read_timeout,
                                   max_retries=max_retries)

    text = _extract_message_text_from_openai_response(data).strip()
    truncated = False
    try:
        ch0 = (data.get("choices") or [None])[0] or {}
        fin = str(ch0.get("finish_reason", "")).lower()
        truncated = (fin == "length")
    except Exception:
        pass

    if truncated or not text:
        fb_url = base_url.rstrip("/") + "/responses"
        fb_payload = {
            "model": model,
            "input": prompt,
            "temperature": max(0.0, float(temperature)),
            "max_output_tokens": int(max(64, int(max_tokens) * 2)),
        }
        fb_data = _post_json_with_retries(fb_url, headers, fb_payload,
                                          connect_timeout=connect_timeout,
                                          read_timeout=read_timeout,
                                          max_retries=max_retries)
        fb_text = _extract_text_from_responses_api(fb_data).strip()
        if fb_text:
            return fb_text

    return text


# -------------------- Main flows --------------------

def run_remote_openai(args: argparse.Namespace):
    base_url = (
        args.remote_base_url
        or os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("LAMBDA_API_BASE")
        or "https://openrouter.ai/api/v1"
    )
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LAMBDA_API_KEY")
    model = args.remote_model or os.environ.get("REMOTE_MODEL") or DEFAULT_REMOTE_MODEL
    # Timeouts: CLI -> ENV -> defaults
    connect_timeout = args.openrouter_connect_timeout if args.openrouter_connect_timeout is not None else int(os.environ.get("OPENROUTER_CONNECT_TIMEOUT", "20"))
    read_timeout = args.openrouter_read_timeout if args.openrouter_read_timeout is not None else int(os.environ.get("OPENROUTER_READ_TIMEOUT", "180"))
    if not api_key:
        raise SystemExit("Missing API key. Provide --api-key or set OPENROUTER_API_KEY (or LAMBDA_API_KEY).")
    if not base_url:
        raise SystemExit("Missing base URL. Provide --remote-base-url or set OPENROUTER_BASE_URL (or LAMBDA_API_BASE).")

    scenarios = load_scenarios(Path(args.scenarios))
    # Optional single-scenario selection
    if args.scenario_index is not None:
        if args.scenario_index < 0 or args.scenario_index >= len(scenarios):
            raise SystemExit(f"--scenario-index {args.scenario_index} out of range (0..{len(scenarios)-1})")
        scenarios = [scenarios[args.scenario_index]]
        print(f"Running only scenario index {args.scenario_index}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for s_idx, scenario in enumerate(tqdm(scenarios, desc="Scenarios", unit="scn"), start=1):
        qa_list = scenario.get("qa", [])
        if not qa_list:
            tqdm.write(f"⚠️ Scenario {s_idx} has no QA items; skipping.")
            continue

        scn_id = str(scenario.get("id") or f"S{s_idx}").replace(" ", "_")
        title = scenario.get("title", "")
        print(f"\n🚦 Running questions (remote) for Scenario {scn_id} — {title}\n")

        items: List[Dict[str, Any]] = []
        for idx, qa in enumerate(tqdm(qa_list, desc="QAs", unit="q"), start=1):
            q = (qa.get("q") or "").strip()
            if not q:
                continue
            situation = compose_situation_text(scenario, q)

            tqdm.write("=" * 80)
            tqdm.write(f"Q{idx}: {q}")
            tqdm.write("-" * 80)

            try:
                answer = remote_chat_complete(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    prompt=situation,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    max_retries=args.max_retries,
                )
            except Exception as e:
                tqdm.write(f"❌ Error calling remote API: {e}")
                continue

            tqdm.write("Answer:\n")
            tqdm.write(answer)

            items.append({
                "index": idx,
                "question": q,
                "answer": answer,
                "emotions": {},
                "top_emotions": [],
                "explanation": None,
                "metadata": {
                    "method": "openai_remote_chat",
                    "pooling": "none",
                    "abstract_length": 0,
                    "num_crucial_emotions": 0,
                    "num_factors": 0,
                    "processing_steps": ["remote_chat_completion"],
                },
            })

            # Be polite to the provider between calls
            if args.inter_call_delay_ms and args.inter_call_delay_ms > 0:
                time.sleep(args.inter_call_delay_ms / 1000.0)

        payload = {
            "scenario_id": scenario.get("id"),
            "title": scenario.get("title"),
            "num_questions": len(qa_list),
            "run_id": run_id,
            "model_info": {
                "status": "remote",
                "model_name": model,
                "sampling_temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "base_url": base_url,
                "provider": "openai-compatible",
                "connect_timeout": connect_timeout,
                "read_timeout": read_timeout,
            },
            "log_info": None,
            "items": items,
        }
        out_path = RESULTS_DIR / f"{scn_id}_results_{run_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Saved results to: {out_path}")

    print("\n✅ Completed remote run across all scenarios.")


def run_local_emobird(args: argparse.Namespace):
    print("\n📄 Loading dataset ...")
    scenarios = load_scenarios(Path(args.scenarios))
    # Optional single-scenario selection
    if args.scenario_index is not None:
        if args.scenario_index < 0 or args.scenario_index >= len(scenarios):
            raise SystemExit(f"--scenario-index {args.scenario_index} out of range (0..{len(scenarios)-1})")
        scenarios = [scenarios[args.scenario_index]]
        print(f"Running only scenario index {args.scenario_index}")
    if not scenarios:
        raise ValueError("No scenarios found in dataset")

    # Initialize structured logger
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = EmobirdLogger(log_dir=str(LOGS_DIR))
    set_logger(logger)
    run_id = logger.session_id

    # Initialize EmoBIRD
    print("🐦 Initializing EmoBIRD engine ...")
    # Apply local model override via environment, if provided
    if getattr(args, "model", None):
        os.environ["EMOBIRD_MODEL"] = args.model
        print(f"🔧 Using local model override from --model: {args.model}")
    emo = Emobird()

    # Prepare results dir and metadata
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model_info = emo.vllm_wrapper.get_model_info()  # type: ignore[attr-defined]
    except Exception:
        model_info = None

    for s_idx, scenario in enumerate(tqdm(scenarios, desc="Scenarios", unit="scn"), start=1):
        qa_list = scenario.get("qa", [])
        if not qa_list:
            tqdm.write(f"⚠️ Scenario {s_idx} has no QA items; skipping.")
            continue

        scn_id = str(scenario.get("id") or f"S{s_idx}").replace(" ", "_")
        title = scenario.get("title", "")
        print(f"\n🚦 Running questions for Scenario {scn_id} — {title}\n")

        results: List[Dict[str, Any]] = []
        for idx, qa in enumerate(tqdm(qa_list, desc="QAs", unit="q"), start=1):
            q = qa.get("q", "").strip()
            if not q:
                continue
            situation = compose_situation_text(scenario, q)

            tqdm.write("=" * 80)
            tqdm.write(f"Q{idx}: {q}")
            tqdm.write("-" * 80)

            try:
                result = emo.analyze_emotion(situation)
            except Exception as e:
                tqdm.write(f"❌ Error running EmoBIRD: {e}")
                continue

            response = result.get("response") or "(no response generated)"
            tqdm.write("Answer:\n")
            tqdm.write(response)

            emotions = result.get("emotions") or {}
            if emotions:
                tqdm.write("\nTop emotions (model):")
                for em, p in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]:
                    tqdm.write(f"  - {em}: {p:.3f}")

            top5 = (
                sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]
                if emotions else []
            )
            rec = {
                "index": idx,
                "question": q,
                "answer": response,
                "emotions": emotions,
                "top_emotions": top5,
                "explanation": result.get("explanation"),
                "metadata": result.get("metadata"),
            }
            results.append(rec)

            try:
                get_logger().log_analysis_result(situation, result)
            except Exception as e:
                tqdm.write(f"⚠️ Failed to log analysis result: {e}")

        try:
            log_info = get_logger().get_session_info()
        except Exception:
            log_info = None

        payload = {
            "scenario_id": scenario.get("id"),
            "title": scenario.get("title"),
            "num_questions": len(qa_list),
            "run_id": run_id,
            "model_info": model_info,
            "log_info": log_info,
            "items": results,
        }
        out_path = RESULTS_DIR / f"{scn_id}_results_{run_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Saved results to: {out_path}")

    try:
        close_logger()
    except Exception as e:
        print(f"⚠️ Failed to close logger: {e}")

    print("\n✅ Completed local EmoBIRD run across all scenarios.")


if __name__ == "__main__":
    args = build_parser().parse_args()

    # If scenarios not explicitly provided, auto-detect in cwd
    try:
        if not args.scenarios:
            cand = Path("scenarios_30.json")
            if cand.exists():
                args.scenarios = str(cand)
                print(f"Detected scenarios file: {args.scenarios}")
    except Exception:
        pass

    if args.provider == "openai":
        run_remote_openai(args)
    else:
        run_local_emobird(args)
