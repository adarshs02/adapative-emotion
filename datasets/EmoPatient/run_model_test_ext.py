#!/usr/bin/env python3
"""
Extended model tester: local HF or remote OpenAI-compatible (e.g., Lambda) models.

- Mirrors `run_model_test.py` behavior for HuggingFace models
- Adds `--provider openai` mode to call a remote OpenAI-compatible API
- Reads API config from environment or CLI flags

Environment variables (used if flags omitted):
- LAMBDA_API_KEY (or OPENAI_API_KEY)
- LAMBDA_API_BASE (or OPENAI_BASE_URL)  e.g., https://api.lambdalabs.com/v1
- EMOBIRD_MODEL (default HF model when provider=hf)

Examples:
  # Local HuggingFace model (default)
  CUDA_VISIBLE_DEVICES=0 python run_model_test_ext.py \
    --provider hf \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --scenarios ./scenarios_30.json

  # Remote Lambda (OpenAI-compatible) model
  LAMBDA_API_KEY=... LAMBDA_API_BASE=https://api.openrouter.ai/v1 \
  python run_model_test_ext.py \
    --provider openai \
    --remote-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --scenarios ./scenarios_30.json

  # Single prompt via remote
  python run_model_test_ext.py --provider openai --remote-model "gpt-4o-mini" \
    --prompt "Write a brief, supportive reply to someone awaiting biopsy results."
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Optional torch/transformers for local HF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# requests is convenient; fallback to urllib if unavailable
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # we'll fallback to urllib
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

# Defaults
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HERE = Path(__file__).resolve()
RESULTS_DIR = HERE.parent / "results"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test HF or OpenAI-compatible LLMs on prompts or EmoPatient scenarios")
    p.add_argument("--provider", choices=["hf", "openai"], default="hf", help="Inference provider: local HF or remote OpenAI-compatible API")

    # Local HF
    p.add_argument("--model", help="HF model id (provider=hf)", default=None)

    # Remote OpenAI-compatible
    p.add_argument("--remote-model", help="Remote model name/id (provider=openai)", default=None)
    p.add_argument("--remote-base-url", help="OpenAI-compatible base URL (env LAMBDA_API_BASE/OPENAI_BASE_URL)", default=None)
    p.add_argument("--api-key", help="API key (env LAMBDA_API_KEY/OPENAI_API_KEY)", default=LAMBDA_API_KEY)

    # Workload
    p.add_argument("--prompt", help="Single prompt to run", default="Please provide a concise, empathetic, medically grounded answer tailored to the above context.")
    p.add_argument("--scenarios", help="Path to scenarios_30.json to run QAs (default: run all scenarios)")
    p.add_argument("--scenario-index", type=int, default=None, help="Run only this scenario index (0-based). If omitted, runs all scenarios.")

    # Sampling
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, help="Random seed")
    return p


# -------------------- Local HF path --------------------

def load_pipe(model_id: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    textgen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        torch_dtype=dtype,
        device_map="auto",
    )
    return textgen, tok


def apply_chat_or_plain(tokenizer, prompt: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        pass
    return prompt


def run_single_prompt_hf(pipe, tok, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    text = apply_chat_or_plain(tok, prompt)
    out = pipe(
        text,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_full_text=False,
    )
    return out[0]["generated_text"].strip()


# -------------------- Remote OpenAI-compatible path --------------------

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


def run_single_prompt_openai(base_url: str, api_key: str, model: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
    # Use Chat Completions schema
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
        "max_tokens": int(max_new_tokens),
    }
    data = _http_post_json(url, headers, payload)
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        raise RuntimeError(f"Malformed response from OpenAI-compatible API: {e}; got: {json.dumps(data)[:500]}")


def run_scenarios_openai(base_url: str, api_key: str, model: str, scenarios_path: Path, scenario_index: Optional[int], max_new_tokens: int, temperature: float):
    data = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        scenarios = data.get("scenarios") or data.get("scenarios_30") or []
    elif isinstance(data, list):
        scenarios = data
    else:
        scenarios = []
    if not scenarios:
        raise ValueError("No scenarios found in file; expected key 'scenarios' or 'scenarios_30'")

    if scenario_index is None:
        indices = list(range(len(scenarios)))
        outer = tqdm(indices, desc="Scenarios", unit="scn")
    else:
        if not (0 <= scenario_index < len(scenarios)):
            raise ValueError(f"scenario-index {scenario_index} out of range (0..{len(scenarios)-1})")
        indices = [scenario_index]
        outer = indices

    overall_results: List[Dict[str, Any]] = []

    for s_idx in outer:
        scn = scenarios[s_idx]
        qa_list = scn.get("qa", [])
        if not qa_list:
            tqdm.write(f"⚠️ Scenario {s_idx} has no QA items; skipping.")
            continue

        tqdm.write(f"\nScenario {s_idx}: {scn.get('id', '(unknown)')} — {scn.get('title', '')}")
        scenario_items: List[Dict[str, Any]] = []
        for i, qa in enumerate(tqdm(qa_list, desc="QAs", unit="q"), start=1):
            qid = (qa.get("qid") or "").strip()
            q = (qa.get("q") or "").strip()
            if not q:
                continue
            prompt = compose_from_scenario(scn, q)
            tqdm.write("=" * 80)
            header = f"Q{i}"
            if qid:
                header += f" [{qid}]"
            tqdm.write(f"{header}: {q}")
            tqdm.write("-" * 80)
            ans = run_single_prompt_openai(base_url, api_key, model, prompt, max_new_tokens, temperature)
            tqdm.write(ans)

            scenario_items.append({
                "index": i,
                "qid": qid,
                "question": q,
                "answer": ans,
            })

        overall_results.append({
            "scenario_index": s_idx,
            "id": scn.get("id"),
            "title": scn.get("title"),
            "num_questions": len(qa_list),
            "qas": scenario_items,
        })

    # Save combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"modeltest_results_{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "scenarios_path": str(scenarios_path),
            "generated_at": run_id,
            "num_scenarios": len(overall_results),
            "items": overall_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📝 Saved results to: {out_path}")


# -------------------- Shared utilities --------------------

def compose_from_scenario(scn: Dict[str, Any], q: str) -> str:
    title = scn.get("title", "")
    diagnosis = scn.get("diagnosis", "")
    tplan = scn.get("treatment_plan", "")
    narrative = scn.get("narrative", "")
    return (
        f"Context (clinical scenario):\n"
        f"Title: {title}\n"
        f"Diagnosis: {diagnosis}\n"
        f"Treatment plan: {tplan}\n\n"
        f"Patient narrative:\n{narrative}\n\n"
        f"Question:\n{q}\n\n"
        f"Please provide a concise, empathetic, medically grounded answer tailored to the above context.\n\n"
        f"Answer:"
    )


def run_scenarios_hf(pipe, tok, scenarios_path: Path, scenario_index: Optional[int], max_new_tokens: int, temperature: float, top_p: float):
    data = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        scenarios = data.get("scenarios") or data.get("scenarios_30") or []
    elif isinstance(data, list):
        scenarios = data
    else:
        scenarios = []
    if not scenarios:
        raise ValueError("No scenarios found in file; expected key 'scenarios' or 'scenarios_30'")

    if scenario_index is None:
        indices = list(range(len(scenarios)))
        outer = tqdm(indices, desc="Scenarios", unit="scn")
    else:
        if not (0 <= scenario_index < len(scenarios)):
            raise ValueError(f"scenario-index {scenario_index} out of range (0..{len(scenarios)-1})")
        indices = [scenario_index]
        outer = indices

    overall_results: List[Dict[str, Any]] = []

    for s_idx in outer:
        scn = scenarios[s_idx]
        qa_list = scn.get("qa", [])
        if not qa_list:
            tqdm.write(f"⚠️ Scenario {s_idx} has no QA items; skipping.")
            continue

        tqdm.write(f"\nScenario {s_idx}: {scn.get('id', '(unknown)')} — {scn.get('title', '')}")
        scenario_items: List[Dict[str, Any]] = []
        for i, qa in enumerate(tqdm(qa_list, desc="QAs", unit="q"), start=1):
            qid = (qa.get("qid") or "").strip()
            q = (qa.get("q") or "").strip()
            if not q:
                continue
            prompt = compose_from_scenario(scn, q)
            tqdm.write("=" * 80)
            header = f"Q{i}"
            if qid:
                header += f" [{qid}]"
            tqdm.write(f"{header}: {q}")
            tqdm.write("-" * 80)
            ans = run_single_prompt_hf(pipe, tok, prompt, max_new_tokens, temperature, top_p)
            tqdm.write(ans)

            scenario_items.append({
                "index": i,
                "qid": qid,
                "question": q,
                "answer": ans,
            })

        overall_results.append({
            "scenario_index": s_idx,
            "id": scn.get("id"),
            "title": scn.get("title"),
            "num_questions": len(qa_list),
            "qas": scenario_items,
        })

    # Save combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"modeltest_results_{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "scenarios_path": str(scenarios_path),
            "generated_at": run_id,
            "num_scenarios": len(overall_results),
            "items": overall_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📝 Saved results to: {out_path}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.seed is not None:
        try:
            torch.manual_seed(args.seed)
        except Exception:
            pass

    # Auto-detect scenarios_30.json in cwd if not provided
    try:
        if not args.scenarios:
            cand = Path("scenarios_30.json")
            if cand.exists():
                args.scenarios = str(cand)
                print(f"Detected scenarios file: {args.scenarios}")
    except Exception:
        pass

    if args.provider == "hf":
        # EMOBIRD_MODEL env > --model > default
        model_id = os.environ.get("EMOBIRD_MODEL") or (args.model or DEFAULT_MODEL)
        print(f"Using local HF model: {model_id}")
        pipe, tok = load_pipe(model_id)
        if args.scenarios:
            run_scenarios_hf(pipe, tok, Path(args.scenarios), args.scenario_index, args.max_new_tokens, args.temperature, args.top_p)
        else:
            prompt: Optional[str] = args.prompt
            if not prompt:
                try:
                    prompt = input("Enter prompt: ")
                except EOFError:
                    raise SystemExit("No prompt provided.")
            print("\n--- Prompt ---\n" + prompt + "\n---------------\n")
            ans = run_single_prompt_hf(pipe, tok, prompt, args.max_new_tokens, args.temperature, args.top_p)
            print("\n--- Output ---\n" + ans + "\n---------------")
    else:
        # Provider: openai-compatible (e.g., Lambda)
        base_url = args.remote_base_url or os.environ.get("LAMBDA_API_BASE") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        api_key = args.api_key or os.environ.get("LAMBDA_API_KEY") or os.environ.get("OPENAI_API_KEY")
        remote_model = args.remote_model or os.environ.get("REMOTE_MODEL") or (args.model or DEFAULT_MODEL)
        if not api_key:
            raise SystemExit("Missing API key. Provide --api-key or set LAMBDA_API_KEY/OPENAI_API_KEY.")
        if not base_url:
            raise SystemExit("Missing base URL. Provide --remote-base-url or set LAMBDA_API_BASE/OPENAI_BASE_URL.")
        print(f"Using remote OpenAI-compatible model: {remote_model} @ {base_url}")
        if args.scenarios:
            run_scenarios_openai(base_url, api_key, remote_model, Path(args.scenarios), args.scenario_index, args.max_new_tokens, args.temperature)
        else:
            prompt: Optional[str] = args.prompt
            if not prompt:
                try:
                    prompt = input("Enter prompt: ")
                except EOFError:
                    raise SystemExit("No prompt provided.")
            print("\n--- Prompt ---\n" + prompt + "\n---------------\n")
            ans = run_single_prompt_openai(base_url, api_key, remote_model, prompt, args.max_new_tokens, args.temperature)
            print("\n--- Output ---\n" + ans + "\n---------------")
