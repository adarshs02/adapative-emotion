"""
Evaluate EmoBench (EA & EU) using the base LLaMA model via vLLM, without the EmoBIRD pipeline.

This script:
- Loads EmoBench EA and EU datasets
- Formats prompts and queries the base LLaMA model using VLLMWrapper
- Parses predictions
- Computes exact-match accuracy and a policy-based closeness score (partial credit)
- Saves detailed results to eval_results/

Usage:
  python evaluate_emobench_llama.py [limit] [--quiet|-q] [--model MODEL_NAME]

Examples:
  python evaluate_emobench_llama.py --quiet
  python evaluate_emobench_llama.py 50 --model meta-llama/Llama-3.1-8B-Instruct
"""

import json
import os
import re
import sys
import atexit
from pathlib import Path
from typing import Dict, List, Any, Tuple

from config import EmobirdConfig
from vllm_wrapper import VLLMWrapper


# -----------------------------
# Utilities
# -----------------------------

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_ea_prompt(item: Dict[str, Any]) -> str:
    """Format EA dataset item as multiple choice prompt."""
    scenario = item['scenario']
    subject = item['subject']
    choices = item['choices']

    choice_text = ""
    for i, choice in enumerate(choices):
        choice_text += f"{chr(97 + i)}) {choice}\n"

    prompt = f"""Scenario: {scenario}
As {subject}, how would you respond?
Choices:
{choice_text}Answer:"""

    return prompt


def format_eu_prompt(item: Dict[str, Any]) -> str:
    """Format EU dataset item as emotion understanding prompt."""
    scenario = item['scenario']
    prompt = f"""Please identify the emotion and its cause from the scenario. Respond ONLY in JSON format.

Scenario: {scenario}
Answer:"""
    return prompt


def parse_ea_response(response: str, choices: List[str]) -> str:
    """Parse EA response to extract choice text."""
    try:
        if isinstance(response, dict):
            if 'text' in response:
                response = str(response['text'])
            elif 'content' in response:
                response = str(response['content'])
            elif 'response' in response:
                response = str(response['response'])
            else:
                response = str(response)
        elif response is None:
            response = ""
        else:
            response = str(response)
        response = response.strip()
    except Exception:
        response = ""

    # Strict pattern first
    choice_pattern = r'[Aa]nswer:\s*([a-d])\)'
    match = re.search(choice_pattern, response)
    if match:
        choice_letter = match.group(1).lower()
        idx = ord(choice_letter) - ord('a')
        if 0 <= idx < len(choices):
            return choices[idx]

    # Fallbacks
    for i, choice in enumerate(choices):
        letter = chr(97 + i)
        if f"{letter})" in response.lower():
            return choice
    for choice in choices:
        if choice.lower() in response.lower():
            return choice

    return choices[0] if choices else ""


def parse_eu_response(response: str) -> Tuple[str, str]:
    """Parse EU response to extract emotion and cause."""
    try:
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        m = re.search(json_pattern, response)
        if m:
            js = m.group(0)
            parsed = json.loads(js)
            emo = parsed.get('emo_label', parsed.get('emotion', ''))
            cause = parsed.get('cause_label', parsed.get('cause', ''))
            return emo, cause
    except Exception:
        pass

    emo = ""
    cause = ""
    emo_patterns = [
        r'emotion["\']?\s*:\s*["\']([^"\']+)["\']',
        r'emo_label["\']?\s*:\s*["\']([^"\']+)["\']',
        r'emotion.*?["\']([^"\']+)["\']'
    ]
    for p in emo_patterns:
        m = re.search(p, response, re.IGNORECASE)
        if m:
            emo = m.group(1)
            break

    cause_patterns = [
        r'cause["\']?\s*:\s*["\']([^"\']+)["\']',
        r'cause_label["\']?\s*:\s*["\']([^"\']+)["\']',
        r'cause.*?["\']([^"\']+)["\']'
    ]
    for p in cause_patterns:
        m = re.search(p, response, re.IGNORECASE)
        if m:
            cause = m.group(1)
            break

    return emo, cause


# -----------------------------
# Policy scoring (closeness)
# -----------------------------

def _call_policy_scorer(vllm_wrapper: VLLMWrapper, system_prompt: str, user_prompt: str) -> Tuple[float, str]:
    """Call the policy model via vLLM wrapper, expect JSON with keys: score, justification."""
    try:
        prompt = f"{system_prompt}\n\n{user_prompt}\n\nReturn only a compact JSON object with keys: score (0..1 float) and justification (string)."
        resp = vllm_wrapper.generate(prompt, component="policy_scorer", interaction_type="evaluation")
        import re as _re, json as _json
        m = _re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', str(resp))
        if not m:
            return 0.0, "No JSON found in scorer response"
        data = _json.loads(m.group(0))
        score = float(data.get('score', 0.0))
        score = max(0.0, min(1.0, score))
        just = str(data.get('justification', ''))
        return score, just
    except Exception as e:
        return 0.0, f"Scorer exception: {e}"


def score_ea_with_policy_model(vllm_wrapper: VLLMWrapper, item: Dict[str, Any], pred_choice_text: str, gt_choice_text: str, raw_output: str) -> Tuple[float, str]:
    """If not an exact match, ask policy model how close the prediction was (0..1)."""
    system_prompt = (
        "You are a strict but fair evaluator for empathetic multiple-choice decisions. "
        "Assess how close the model's chosen option is to the ground-truth option, considering empathy, appropriateness, and safety."
    )
    choices = item.get('choices', [])
    choice_lines = [f"{chr(97+i)}) {c}" for i, c in enumerate(choices)]
    choices_block = "\n".join(choice_lines)
    user_prompt = f"""
Scenario: {item.get('scenario','')}
Subject: {item.get('subject','')}
Choices:
{choices_block}

Ground truth choice text: {gt_choice_text}
Model predicted choice text: {pred_choice_text}
Model raw output: {raw_output}

Scoring rubric:
- 1.0 if prediction is effectively equivalent in meaning and quality to ground-truth.
- 0.6-0.9 if generally appropriate and empathetic but misses some key aspect.
- 0.3-0.5 if partially appropriate but with notable issues.
- 0.1-0.2 if weakly related or slightly inappropriate.
- 0.0 if unrelated, clearly inappropriate, or harmful.
"""
    return _call_policy_scorer(vllm_wrapper, system_prompt, user_prompt)


def score_eu_with_policy_model(vllm_wrapper: VLLMWrapper, item: Dict[str, Any], pred_emo: str, pred_cause: str, gt_emo: str, gt_cause: str, raw_output: str) -> Tuple[float, str]:
    """Policy scoring for EU: evaluate closeness of emotion and cause. Return a combined score 0..1 and justification."""
    system_prompt = (
        "You are an evaluator for emotion understanding. Score the closeness between the predicted emotion/cause "
        "and the ground truth. Consider synonyms for emotions and reasonable paraphrases for causes."
    )
    user_prompt = f"""
Scenario: {item.get('scenario','')}
Subject: {item.get('subject','')}

Ground truth emotion: {gt_emo}
Predicted emotion: {pred_emo}

Ground truth cause: {gt_cause}
Predicted cause: {pred_cause}

Model raw output: {raw_output}

Scoring rubric (produce a single overall score 0..1):
- Start from emotion similarity (semantic equivalence and category closeness) as 60% of the score.
- Cause similarity (semantic overlap and relevance) as 40% of the score.
- 1.0 if both are essentially equivalent; 0.0 if both are unrelated.
"""
    return _call_policy_scorer(vllm_wrapper, system_prompt, user_prompt)


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_ea_dataset(vllm_wrapper: VLLMWrapper, dataset: List[Dict[str, Any]], quiet: bool = False) -> List[Dict[str, Any]]:
    results = []
    correct_count = 0

    if not quiet:
        print(f"\n🎭 Evaluating EA Dataset ({len(dataset)} items)")
        print("=" * 60)

    for idx, item in enumerate(dataset):
        if not quiet:
            print(f"\n📝 Item {idx + 1}/{len(dataset)}")
        prompt = format_ea_prompt(item)
        if not quiet:
            print(f"📋 Scenario: {item['scenario'][:100]}...")

        try:
            raw_output = vllm_wrapper.generate(prompt, component="ea_base_llama", interaction_type="ea_prompt")
            if not quiet:
                try:
                    print(f"💭 Raw output: {str(raw_output)[:150]}...")
                except Exception:
                    pass

            pred_choice = parse_ea_response(raw_output, item['choices'])
            gt_choice = item['label']

            try:
                pred_choice_str = str(pred_choice).strip().lower() if pred_choice is not None else ""
                gt_choice_str = str(gt_choice).strip().lower() if gt_choice is not None else ""
                is_correct = pred_choice_str == gt_choice_str
            except Exception:
                is_correct = False

            if is_correct:
                correct_count += 1

            if not quiet:
                print(f"✅ Predicted Choice: {pred_choice}")
                print(f"🎯 Ground Truth: {gt_choice}")
                print(f"{'✓' if is_correct else '✗'} Correct: {is_correct}")

            results.append({
                "idx": idx,
                "prompt": prompt,
                "raw_model_output": raw_output,
                "pred_choice_text": pred_choice,
                "gt_choice_text": gt_choice,
                "is_correct": is_correct,
            })
        except Exception as e:
            if not quiet:
                print(f"❌ Error processing item {idx}: {e}")
            results.append({
                "idx": idx,
                "prompt": prompt,
                "raw_model_output": f"Error: {str(e)}",
                "pred_choice_text": item['choices'][0] if item['choices'] else "",
                "gt_choice_text": item['label'],
                "is_correct": False,
            })

    # Policy closeness scoring
    total_points = 0.0
    for r in results:
        try:
            if r.get('is_correct'):
                r['policy_score'] = 1.0
                r['policy_justification'] = ''
                r['final_score'] = 1.0
                total_points += 1.0
            else:
                item = dataset[r['idx']]
                ps, pj = score_ea_with_policy_model(
                    vllm_wrapper,
                    item,
                    r.get('pred_choice_text', ''),
                    r.get('gt_choice_text', ''),
                    r.get('raw_model_output', ''),
                )
                r['policy_score'] = float(ps)
                r['policy_justification'] = pj
                r['final_score'] = float(ps)
                total_points += float(ps)
        except Exception as _e:
            if r.get('is_correct'):
                r['policy_score'] = 1.0
                r['policy_justification'] = ''
                r['final_score'] = 1.0
                total_points += 1.0
            else:
                r['policy_score'] = 0.0
                r['policy_justification'] = f'Policy scoring failed: {_e}'
                r['final_score'] = 0.0
                total_points += 0.0

    accuracy = correct_count / len(dataset) if dataset else 0.0
    avg_score = total_points / len(dataset) if dataset else 0.0

    if not quiet:
        print(f"\n📊 EA Dataset Results:")
        print(f"   Total Items: {len(dataset)}")
        print(f"   Correct: {correct_count}")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Average policy score: {avg_score:.4f} ({avg_score*100:.2f}%)")

    return results


def evaluate_eu_dataset(vllm_wrapper: VLLMWrapper, dataset: List[Dict[str, Any]], quiet: bool = False) -> List[Dict[str, Any]]:
    results = []
    correct_count = 0
    emo_correct_count = 0

    if not quiet:
        print(f"\n😊 Evaluating EU Dataset ({len(dataset)} items)")
        print("=" * 60)

    for idx, item in enumerate(dataset):
        if not quiet:
            print(f"\n📝 Item {idx + 1}/{len(dataset)}")
        prompt = format_eu_prompt(item)
        if not quiet:
            print(f"Prompt: {prompt[:100]}...")

        try:
            raw_output = vllm_wrapper.generate(prompt, component="eu_base_llama", interaction_type="eu_prompt")
            pred_emo, pred_cause = parse_eu_response(str(raw_output))
            gt_emo = item['emotion_label']
            gt_cause = item['cause_label']

            try:
                pred_emo_str = str(pred_emo).strip().lower() if pred_emo is not None else ""
                gt_emo_str = str(gt_emo).strip().lower() if gt_emo is not None else ""
                pred_cause_str = str(pred_cause).strip().lower() if pred_cause is not None else ""
                gt_cause_str = str(gt_cause).strip().lower() if gt_cause is not None else ""
                is_emo_correct = pred_emo_str == gt_emo_str
                is_correct = is_emo_correct and pred_cause_str == gt_cause_str
            except Exception:
                is_emo_correct = False
                is_correct = False

            if is_correct:
                correct_count += 1
            if is_emo_correct:
                emo_correct_count += 1

            if not quiet:
                print(f"😊 Predicted Emotion: {pred_emo}")
                print(f"🎯 Ground Truth Emotion: {gt_emo}")
                print(f"📋 Predicted Cause: {pred_cause}")
                print(f"🎯 Ground Truth Cause: {gt_cause}")
                print(f"{'✓' if is_emo_correct else '✗'} Emotion Correct: {is_emo_correct}")
                print(f"{'✓' if is_correct else '✗'} Overall Correct: {is_correct}")

            results.append({
                "idx": idx,
                "prompt": prompt,
                "raw_model_output": str(raw_output),
                "pred_emo": pred_emo,
                "pred_cause": pred_cause,
                "gt_emo": gt_emo,
                "gt_cause": gt_cause,
                "is_correct": is_correct,
                "is_emo_correct": is_emo_correct,
            })
        except Exception as e:
            if not quiet:
                print(f"❌ Error processing item {idx}: {e}")
            results.append({
                "idx": idx,
                "prompt": prompt,
                "raw_model_output": f"Error: {str(e)}",
                "pred_emo": "",
                "pred_cause": "",
                "gt_emo": item['emotion_label'],
                "gt_cause": item['cause_label'],
                "is_correct": False,
                "is_emo_correct": False,
            })

    # Policy closeness scoring
    total_points = 0.0
    for r in results:
        try:
            if r.get('is_correct'):
                r['policy_score'] = 1.0
                r['policy_justification'] = ''
                r['final_score'] = 1.0
                total_points += 1.0
            else:
                item = dataset[r['idx']]
                ps, pj = score_eu_with_policy_model(
                    vllm_wrapper,
                    item,
                    r.get('pred_emo', ''),
                    r.get('pred_cause', ''),
                    r.get('gt_emo', ''),
                    r.get('gt_cause', ''),
                    r.get('raw_model_output', ''),
                )
                r['policy_score'] = float(ps)
                r['policy_justification'] = pj
                r['final_score'] = float(ps)
                total_points += float(ps)
        except Exception as _e:
            if r.get('is_correct'):
                r['policy_score'] = 1.0
                r['policy_justification'] = ''
                r['final_score'] = 1.0
                total_points += 1.0
            else:
                r['policy_score'] = 0.0
                r['policy_justification'] = f'Policy scoring failed: {_e}'
                r['final_score'] = 0.0
                total_points += 0.0

    accuracy = correct_count / len(dataset) if dataset else 0.0
    emo_accuracy = emo_correct_count / len(dataset) if dataset else 0.0
    avg_score = total_points / len(dataset) if dataset else 0.0

    if not quiet:
        print(f"\n📊 EU Dataset Results:")
        print(f"   Total Items: {len(dataset)}")
        print(f"   Emotion Correct: {emo_correct_count}")
        print(f"   Overall Correct: {correct_count}")
        print(f"   Emotion Accuracy: {emo_accuracy:.4f} ({emo_accuracy*100:.2f}%)")
        print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Average policy score: {avg_score:.4f} ({avg_score*100:.2f}%)")

    return results


# -----------------------------
# Entrypoint
# -----------------------------

def cleanup_resources():
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def main(limit_entries: int = None, quiet: bool = False, model_name: str = None):
    atexit.register(cleanup_resources)

    # Paths
    dataset_dir = Path("/mnt/shared/adarsh/datasets/EmoBench/data")
    eval_results_dir = Path("eval_results")
    eval_results_dir.mkdir(exist_ok=True)

    # Load datasets
    if not quiet:
        print("📂 Loading datasets...")
    ea_dataset = load_dataset(dataset_dir / "EA.jsonl")
    eu_dataset = load_dataset(dataset_dir / "EU.jsonl")

    # Limit datasets if specified
    if limit_entries:
        ea_dataset = ea_dataset[:limit_entries]
        eu_dataset = eu_dataset[:limit_entries]
        if not quiet:
            print(f"   📊 Limited to {limit_entries} entries each for testing")

    if not quiet:
        print(f"   EA Dataset: {len(ea_dataset)} items")
        print(f"   EU Dataset: {len(eu_dataset)} items")

    # Initialize base LLaMA
    cfg = EmobirdConfig()
    if model_name:
        cfg.update_config(llm_model_name=model_name)
    if not quiet:
        print(f"🐪 Initializing base LLaMA model: {cfg.llm_model_name}")
    vllm = VLLMWrapper(cfg)

    # Evaluate EA and EU with base model
    ea_results = evaluate_ea_dataset(vllm, ea_dataset, quiet=quiet)
    eu_results = evaluate_eu_dataset(vllm, eu_dataset, quiet=quiet)

    # Save results
    if not quiet:
        print(f"\n💾 Saving results to {eval_results_dir}/")

    ea_output_path = eval_results_dir / "llama_ea_results.json"
    with open(ea_output_path, 'w', encoding='utf-8') as f:
        json.dump(ea_results, f, indent=2, ensure_ascii=False)
    if not quiet:
        print(f"   EA results saved to: {ea_output_path}")

    eu_output_path = eval_results_dir / "llama_eu_results.json"
    with open(eu_output_path, 'w', encoding='utf-8') as f:
        json.dump(eu_results, f, indent=2, ensure_ascii=False)
    if not quiet:
        print(f"   EU results saved to: {eu_output_path}")

    if not quiet:
        print("\n🎉 Base LLaMA Evaluation Complete!")


if __name__ == "__main__":
    # CLI parsing: [limit] [--quiet|-q] [--model MODEL_NAME]
    limit_entries = None
    quiet = False
    model_name = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--quiet", "-q"):
            quiet = True
            print("🔇 Quiet mode enabled")
            i += 1
        elif arg == "--model" and i + 1 < len(args):
            model_name = args[i + 1]
            print(f"🧩 Overriding model: {model_name}")
            i += 2
        else:
            try:
                limit_entries = int(arg)
                print(f"📊 Running with limit: {limit_entries} entries per dataset")
                i += 1
            except ValueError:
                print(f"⚠️ Invalid argument: {arg}")
                i += 1

    if limit_entries is None:
        limit_entries = 20
        if not quiet:
            print("📊 Defaulting to 20 entries per dataset for quick testing")

    main(limit_entries, quiet, model_name)
