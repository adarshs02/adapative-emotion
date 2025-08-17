#!/usr/bin/env python3
"""
Generic model tester: run any HF causal LLM on a prompt or on EmoPatient scenarios.

Notes:
- If run with no flags inside the EmoPatient scenarios folder, the script will auto-use
  ./scenarios.json when present. Otherwise it falls back to a single default prompt.

Examples:
  # Single prompt
  CUDA_VISIBLE_DEVICES=6 python run_model_test.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --prompt "Write a brief, supportive reply to someone awaiting biopsy results."

  # Use scenarios.json (S1 style)
  CUDA_VISIBLE_DEVICES=6 python run_model_test.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --scenarios ./scenarios.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Default model used when neither --model nor EMOBIRD_MODEL is provided.
# Edit this to change the script's built-in default.
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test any HF causal LLM on prompts or EmoPatient scenarios")
    p.add_argument("--model", help="HF model id, e.g., meta-llama/Meta-Llama-3.1-8B-Instruct", default=None)
    p.add_argument("--prompt", help="Single prompt to run", default="Please provide a concise, empathetic, medically grounded answer tailored to the above context.")
    p.add_argument("--scenarios", help="Path to EmoPatient scenarios.json to run first scenario's QAs")
    p.add_argument("--scenario-index", type=int, default=0, help="Scenario index to use from scenarios file (default 0)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, help="Random seed")
    return p


def load_pipe(model_id: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        # Many LLMs don't have a pad token; use EOS
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
    # If the tokenizer has a chat template, use it for instruction-tuned models.
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


def run_single_prompt(pipe, tok, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
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


def run_scenarios(pipe, tok, scenarios_path: Path, scenario_index: int, max_new_tokens: int, temperature: float, top_p: float):
    data = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])
    if not scenarios:
        raise ValueError("No scenarios found in file")
    if not (0 <= scenario_index < len(scenarios)):
        raise ValueError(f"scenario-index {scenario_index} out of range (0..{len(scenarios)-1})")

    scn = scenarios[scenario_index]
    qa_list = scn.get("qa", [])
    if not qa_list:
        raise ValueError("Selected scenario has no QA items")

    print(f"\nScenario: {scn.get('id', '(unknown)')} — {scn.get('title', '')}")
    for i, qa in enumerate(qa_list, start=1):
        q = (qa.get("q") or "").strip()
        if not q:
            continue
        prompt = compose_from_scenario(scn, q)
        print("=" * 80)
        print(f"Q{i}: {q}")
        print("-" * 80)
        ans = run_single_prompt(pipe, tok, prompt, max_new_tokens, temperature, top_p)
        print(ans)


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.seed is not None:
        try:
            torch.manual_seed(args.seed)
        except Exception:
            pass

    # Choose model priority: EMOBIRD_MODEL env > CLI --model > DEFAULT_MODEL
    model_id = os.environ.get("EMOBIRD_MODEL") or (args.model or DEFAULT_MODEL)
    # Auto-detect scenarios.json in current working directory if not provided
    try:
        default_scen = Path("scenarios.json")
        if not args.scenarios and default_scen.exists():
            args.scenarios = str(default_scen)
            print(f"Detected scenarios file: {args.scenarios}")
    except Exception:
        pass

    print(f"Using model: {model_id}")
    pipe, tok = load_pipe(model_id)

    if args.scenarios:
        run_scenarios(pipe, tok, Path(args.scenarios), args.scenario_index, args.max_new_tokens, args.temperature, args.top_p)
    else:
        # Resolve prompt: from --prompt or fallback to input()
        prompt: Optional[str] = args.prompt
        if not prompt:
            try:
                prompt = input("Enter prompt: ")
            except EOFError:
                raise SystemExit("No prompt provided.")
        print("\n--- Prompt ---\n" + prompt + "\n---------------\n")
        ans = run_single_prompt(pipe, tok, prompt, args.max_new_tokens, args.temperature, args.top_p)
        print("\n--- Output ---\n" + ans + "\n---------------")
