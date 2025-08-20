#!/usr/bin/env python3
"""
Extended EmoPatient runner:
- Mode 1 (default): runs EmoBIRD pipeline locally (vLLM)
- Mode 2: runs remote OpenAI-compatible (e.g., Lambda) model via API key

Why: sometimes you want to try other hosted models without standing up vLLM locally.

Environment variables (used if flags omitted):
- EMOBIRD_MODEL: default local model for EmoBIRD (prefers Llama)
- LAMBDA_API_KEY (or OPENAI_API_KEY)
- LAMBDA_API_BASE (or OPENAI_BASE_URL/OPENAI_API_BASE)  e.g., https://api.lambdalabs.com/v1

Examples
- Local EmoBIRD (default):
  CUDA_VISIBLE_DEVICES=0 python run_emopatient_EmoBIRD_ext.py --provider emo

- Remote OpenAI-compatible (Lambda/OpenRouter/etc):
  LAMBDA_API_KEY=... LAMBDA_API_BASE=https://api.openrouter.ai/v1 \
  python run_emopatient_EmoBIRD_ext.py --provider openai \
    --remote-model meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import json
import os
import sys
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
    import urllib.error

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
DEFAULT_REMOTE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run EmoPatient with EmoBIRD (local) or OpenAI-compatible remote API")
    p.add_argument("--provider", choices=["emo", "openai"], default="emo", help="'emo' = EmoBIRD local vLLM, 'openai' = remote OpenAI-compatible API")

    # Remote OpenAI-compatible configuration
    p.add_argument("--remote-model", default=None, help="Remote model name/id")
    p.add_argument("--remote-base-url", default=None, help="Base URL for OpenAI-compatible API (env LAMBDA_API_BASE/OPENAI_BASE_URL)")
    p.add_argument("--api-key", default=LAMBDA_API_KEY, help="API key (env LAMBDA_API_KEY/OPENAI_API_KEY)")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=220)

    # Execution controls
    p.add_argument("--scenario-index", type=int, default=None, help="Run only this scenario (0-based). If omitted, runs all.")
    p.add_argument("--scenarios", default=str(DATASET_PATH), help="Path to scenarios JSON (default: datasets/EmoPatient/scenarios_30.json)")
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
        f"Please provide a concise, empathetic, medically grounded answer tailored to the above context."
    )
    return situation


# -------------------- Remote OpenAI-compatible helpers --------------------

def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    if requests is not None:
        r = requests.post(url, headers=headers, data=body, timeout=120)
        r.raise_for_status()
        return r.json()
    else:  # urllib fallback
        req = urllib.request.Request(url, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, data=body, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # pragma: no cover
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")


def remote_chat_complete(base_url: str, api_key: str, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
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
    data = _http_post_json(url, headers, payload)
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        raise RuntimeError(f"Malformed response from OpenAI-compatible API: {e}; got: {json.dumps(data)[:500]}")


# -------------------- Main flows --------------------

def run_remote_openai(args: argparse.Namespace):
    base_url = args.remote_base_url or os.environ.get("LAMBDA_API_BASE") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = args.api_key or os.environ.get("LAMBDA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = args.remote_model or os.environ.get("REMOTE_MODEL") or DEFAULT_REMOTE_MODEL
    if not api_key:
        raise SystemExit("Missing API key. Provide --api-key or set LAMBDA_API_KEY/OPENAI_API_KEY.")
    if not base_url:
        raise SystemExit("Missing base URL. Provide --remote-base-url or set LAMBDA_API_BASE/OPENAI_BASE_URL.")

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
                answer = remote_chat_complete(base_url, api_key, model, situation, args.temperature, args.max_tokens)
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
