"""
EmoBench Evaluation Script for EmoBIRD

Evaluates EmoBIRD framework on EmoBench EA (Empathy Actions) and EU (Emotion Understanding) datasets.
Provides real-time output and saves results in specified JSON format.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import atexit

# Add EmoBIRD modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emobird_poc import Emobird


def _call_policy_scorer(vllm_wrapper, system_prompt: str, user_prompt: str):
    """Call the policy model via existing vLLM wrapper, expecting JSON with keys: score, justification.
    Returns (score_float, justification_str). Falls back to (0.0, reason) on any error.
    """
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


def score_ea_with_policy_model(emobird, item, pred_choice_text: str, gt_choice_text: str, raw_output: str):
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
    return _call_policy_scorer(emobird.vllm_wrapper, system_prompt, user_prompt)


def score_eu_with_policy_model(emobird, item, pred_emo: str, pred_cause: str, gt_emo: str, gt_cause: str, raw_output: str):
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
    return _call_policy_scorer(emobird.vllm_wrapper, system_prompt, user_prompt)



def cleanup_resources():
    """Clean up distributed computing resources to prevent warnings"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass


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
    
    # Format choices as a), b), c), d)
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
    subject = item['subject']
    
    prompt = f"""Please identify the emotion and its cause from the scenario. Respond ONLY in JSON format.

Scenario: {scenario}
Answer:"""
    
    return prompt


def parse_ea_response(response: str, choices: List[str]) -> str:
    """Parse EA response to extract choice text."""
    # Robust type checking to prevent 'dict' object has no attribute 'strip' errors
    try:
        if isinstance(response, dict):
            # If response is a dict, try to extract text content
            if 'text' in response:
                response = str(response['text'])
            elif 'content' in response:
                response = str(response['content'])
            elif 'response' in response:
                response = str(response['response'])
            else:
                # Fallback: convert entire dict to string
                response = str(response)
        elif response is None:
            response = ""
        else:
            # Ensure it's a string
            response = str(response)
            
        response = response.strip()
    except Exception as e:
        print(f"⚠️ Error processing EA response: {e}")
        response = ""  # Fallback to empty string
    
    # Look for choice letters (a, b, c, d) with optional colon/paren
    choice_pattern = r'[Aa]nswer[:\s]*([a-dA-D])\)?'
    match = re.search(choice_pattern, response)
    
    if match:
        choice_letter = match.group(1).lower()
        choice_index = ord(choice_letter) - ord('a')
        if 0 <= choice_index < len(choices):
            return choices[choice_index]
    
    # Fallback: look for choice letter anywhere in response
    for i, choice in enumerate(choices):
        choice_letter = chr(97 + i)  # a, b, c, d
        if f"{choice_letter})" in response.lower():
            return choice
    
    # Last resort: try to match choice text directly
    for choice in choices:
        if choice.lower() in response.lower():
            return choice
    
    # Default fallback
    return choices[0] if choices else ""


def parse_eu_response(response: str) -> tuple:
    """Parse EU response to extract emotion and cause."""
    # First, try to extract strict JSON
    try:
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_match = re.search(json_pattern, response)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            emotion = parsed.get('emo_label', parsed.get('emotion', ''))
            cause = parsed.get('cause_label', parsed.get('cause', ''))
            return emotion, cause
    except Exception:
        # Fall through to regex-based fallback
        pass
    
    # Fallback parsing (regex-based, tolerant)
    emotion = ""
    cause = ""
    
    # Look for emotion patterns
    emotion_patterns = [
        r'emotion["\']?\s*:\s*["\']([^"\']+)["\']',
        r'emo_label["\']?\s*:\s*["\']([^"\']+)["\']',
        r'emotion.*?["\']([^"\']+)["\']'
    ]
    
    for pattern in emotion_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            emotion = match.group(1)
            break
    
    # Look for cause patterns  
    cause_patterns = [
        r'cause["\']?\s*:\s*["\']([^"\']+)["\']',
        r'cause_label["\']?\s*:\s*["\']([^"\']+)["\']',
        r'cause.*?["\']([^"\']+)["\']'
    ]
    
    for pattern in cause_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            cause = match.group(1)
            break
    
    return emotion, cause

def predict_eu_labels_with_choices(emobird: Emobird, item: Dict[str, Any]) -> tuple:
    """Use the LLM (via robust json_call) to select emotion and cause strictly from provided choices.
    Returns (pred_emo, pred_cause, raw_output_json_str).
    """
    scenario = item.get('scenario', '')
    subject = item.get('subject', '')
    emo_choices = item.get('emotion_choices', [])
    cause_choices = item.get('cause_choices', [])

    # Build strict instruction to force verbatim selection from choices
    choices_block = (
        "Emotion choices:\n- " + "\n- ".join(str(c) for c in emo_choices) + "\n\n" +
        "Cause choices:\n- " + "\n- ".join(str(c) for c in cause_choices)
    )
    prompt = (
        "You are EmoBIRD. Read the scenario and choose exactly ONE emotion and ONE cause from the provided choices. "
        "Return STRICT JSON only (no markdown, no code fences) with keys 'emo_label' and 'cause_label'. "
        "The values MUST be verbatim copies of one option from the respective choices. Do not invent new options.\n\n"
        f"Subject: {subject}\n"
        f"Scenario: {scenario}\n\n"
        f"{choices_block}\n\n"
        "Output format strictly: {\"emo_label\": \"<one of emotion choices>\", \"cause_label\": \"<one of cause choices>\"}"
    )

    try:
        data = emobird.vllm_wrapper.json_call(
            prompt=prompt,
            component="eu_predictor",
            interaction_type="eu_prediction",
            max_retries=getattr(emobird.config, 'allow_format_only_retry', 1),
            schema_model=None,
            temperature_override=0.0,
            max_tokens_override=256,
        )
        pred_emo = str(data.get('emo_label') or data.get('emotion') or '').strip()
        pred_cause = str(data.get('cause_label') or data.get('cause') or '').strip()

        # Enforce membership; if invalid, attempt a light-weight reformat prompt once
        def _coerce_to_choice(val: str, choices: List[str]) -> str:
            if val in choices:
                return val
            # Try case-insensitive exact match
            lower_map = {c.lower(): c for c in choices}
            if val.lower() in lower_map:
                return lower_map[val.lower()]
            # Fallback to closest match using difflib
            try:
                import difflib
                m = difflib.get_close_matches(val, choices, n=1, cutoff=0.0)
                return m[0] if m else (choices[0] if choices else '')
            except Exception:
                return choices[0] if choices else ''

        pred_emo = _coerce_to_choice(pred_emo, emo_choices)
        pred_cause = _coerce_to_choice(pred_cause, cause_choices)

        raw_output = json.dumps({"emo_label": pred_emo, "cause_label": pred_cause}, ensure_ascii=False)
        return pred_emo, pred_cause, raw_output
    except Exception as _e:
        # As a fallback, derive from EmoBIRD's unified analysis: map top emotion to choices; pick closest cause choice by semantic proximity to scenario via difflib
        try:
            analysis = emobird.analyze_emotion(scenario)
            emotions = analysis.get('emotions', {})
            top_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else ''
        except Exception:
            top_emotion = ''
        # Coerce emotion to closest choice
        def _closest(val: str, choices: List[str]) -> str:
            try:
                import difflib
                m = difflib.get_close_matches(val, choices, n=1, cutoff=0.0)
                return m[0] if m else (choices[0] if choices else '')
            except Exception:
                return choices[0] if choices else ''
        pred_emo = _closest(top_emotion, emo_choices)
        # Choose cause by nearest to scenario text (very rough but bounded to choices)
        pred_cause = _closest(scenario, cause_choices)
        raw_output = json.dumps({"emo_label": pred_emo, "cause_label": pred_cause}, ensure_ascii=False)
        return pred_emo, pred_cause, raw_output


def generate_ea_choice_with_emotions(emobird: Emobird, scenario: str, subject: str, choices: List[str]) -> str:
    """Generate EA choice selection using EmoBIRD emotional insights."""
    try:
        # Run EmoBIRD emotional analysis
        emotion_result = emobird.analyze_emotion(scenario)
        
        # Extract key insights
        emotions = emotion_result.get('emotions', {})
        factors = emotion_result.get('factors', {})
        crucial_emotions = emotion_result.get('crucial_emotions', [])
        
        # Format emotion insights for reasoning
        emotion_context = ""
        if emotions and isinstance(emotions, dict):
            try:
                # Get top 3 emotions by probability
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                emotion_list = [f"{emotion} ({prob:.2f})" for emotion, prob in sorted_emotions]
                emotion_context += f"Predicted emotions: {', '.join(emotion_list)}. "
            except (TypeError, AttributeError, ValueError) as e:
                print(f"⚠️ Error processing emotions for EA: {e}")
                emotion_context += "Unable to process emotion predictions. "
        
        if factors:
            factor_list = [f"{factor}: {value}" for factor, value in factors.items()]
            emotion_context += f"Key factors: {', '.join(factor_list)}. "
        
        if crucial_emotions:
            emotion_context += f"Crucial emotions identified: {', '.join(crucial_emotions)}."
        
        # Build choice selection prompt
        choice_text = ""
        for i, choice in enumerate(choices):
            choice_text += f"{chr(97 + i)}) {choice}\n"
        
        reasoning_prompt = f"""You are an empathetic decision-maker. Analyze this scenario and choose the most appropriate response.

SCENARIO: {scenario}

EMOTIONAL ANALYSIS: {emotion_context}

As {subject}, consider the emotional context and choose the most empathetic and appropriate response:

{choice_text}
Based on the emotional analysis, which choice would be most appropriate? Consider:
- The emotional state of all people involved
- The potential emotional impact of each choice
- The most empathetic and constructive approach

Respond with the letter (a, b, c, or d) followed by brief reasoning.

Answer:"""
        
        # Generate choice using vLLM
        response = emobird.vllm_wrapper.generate(
            reasoning_prompt,
            component="ea_choice_generator",
            interaction_type="empathetic_choice_selection"
        )
        
        return response
        
    except Exception as e:
        print(f"⚠️ Error in emotion-based choice generation: {e}")
        # Fallback to simple reasoning
        return f"Unable to analyze emotions. Default reasoning for scenario: {scenario[:100]}..."


def predict_ea_choice_with_choices(emobird: Emobird, item: Dict[str, Any]) -> Tuple[str, str]:
    """Select EA choice strictly from provided options using robust JSON selection.
    Returns (pred_choice_text, raw_output_str).
    """
    choices = item.get('choices', [])
    scenario = item.get('scenario', '')
    subject = item.get('subject', 'user')

    # Prepare choice block
    choice_lines = [f"{chr(97+i)}) {c}" for i, c in enumerate(choices)]
    choices_block = "\n".join(choice_lines)

    # Optional: include EmoBIRD analysis context to guide the selection
    emotion_ctx = ""
    try:
        emo = emobird.analyze_emotion(scenario)
        if isinstance(emo, dict):
            emod = emo.get('emotions', {})
            if isinstance(emod, dict) and emod:
                try:
                    top3 = sorted(emod.items(), key=lambda x: x[1], reverse=True)[:3]
                    emotion_ctx += "Predicted emotions: " + ", ".join([f"{k} ({v:.2f})" for k, v in top3]) + ". "
                except Exception:
                    pass
            factors = emo.get('factors', {})
            if isinstance(factors, dict) and factors:
                try:
                    flist = [f"{k}: {v}" for k, v in list(factors.items())[:5]]
                    emotion_ctx += "Key factors: " + ", ".join(flist) + ". "
                except Exception:
                    pass
            ce = emo.get('crucial_emotions', [])
            if isinstance(ce, list) and ce:
                emotion_ctx += "Crucial emotions identified: " + ", ".join([str(x) for x in ce]) + "."
    except Exception as _e:
        # Non-fatal; proceed without context
        emotion_ctx = ""

    system = (
        "You are an empathetic decision-maker. Select exactly one option that is most empathetic, supportive, and safe. "
        "Consider emotional wellbeing, de-escalation, and respect."
    )
    user = f"""
Scenario: {scenario}
Subject: {subject}

Choices (select exactly one):
{choices_block}

Emotional analysis context (optional): {emotion_ctx}

Return STRICT JSON only with keys:
{{
  "choice_letter": "a|b|c|d",
  "choice_text": "<exact text copied from one of the choices>"
}}

Rules:
- The choice_text must be exactly one of the provided choices (verbatim match).
- Do not include any text before or after the JSON object.
- If unsure, pick the safest, most empathetic option.
"""

    schema = {
        "required": ["choice_text"],
        "properties": {
            "choice_letter": {"type": "string"},
            "choice_text": {"type": "string"}
        }
    }

    try:
        data = emobird.vllm_wrapper.json_call(
            prompt=f"{system}\n\n{user}",
            component="ea_choice_selector",
            interaction_type="ea_choice_selection",
            schema=schema,
            temperature_override=0.0,
            max_tokens_override=256,
        )
        # Coerce to valid choice
        letter = str(data.get('choice_letter', '')).strip().lower() if isinstance(data, dict) else ''
        txt = str(data.get('choice_text', '')).strip() if isinstance(data, dict) else ''

        # Prefer exact letter mapping if valid
        if letter in ['a', 'b', 'c', 'd']:
            idx = ord(letter) - ord('a')
            if 0 <= idx < len(choices):
                return choices[idx], json.dumps(data)

        # Else, try exact text match (case-insensitive)
        for c in choices:
            if txt.lower() == c.strip().lower():
                return c, json.dumps(data)

        # Else, try substring containment heuristic
        for c in choices:
            cl = c.strip().lower()
            if cl in txt.lower() or txt.lower() in cl:
                return c, json.dumps(data)

        # Last resort: fall back to freeform generator + parser
        raw = generate_ea_choice_with_emotions(emobird, scenario, subject, choices)
        return parse_ea_response(raw, choices), json.dumps(data)
    except Exception as e:
        print(f"⚠️ EA strict JSON selection failed, falling back: {e}")
        raw = generate_ea_choice_with_emotions(emobird, scenario, subject, choices)
        return parse_ea_response(raw, choices), str(raw)

def evaluate_ea_dataset(emobird: Emobird, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate EA dataset using EmoBIRD emotional insights."""
    results = []
    correct_count = 0
    total_points = 0.0
    
    print(f"\n🎭 Evaluating EA Dataset ({len(dataset)} items)")
    print("=" * 60)
    
    for idx, item in enumerate(dataset):
        print(f"\n📝 Item {idx + 1}/{len(dataset)}")
        
        # Format prompt for display
        prompt = format_ea_prompt(item)
        print(f"📋 Scenario: {item['scenario'][:100]}...")
        
        try:
            # Select choice via strict JSON helper (with robust fallbacks)
            print("🧠 Selecting choice via strict JSON...")
            pred_choice, raw_output = predict_ea_choice_with_choices(emobird, item)

            # Safely display raw output with proper type checking
            try:
                if isinstance(raw_output, str):
                    print(f"💭 Raw selection output: {raw_output[:150]}...")
                else:
                    print(f"💭 Raw selection output: {str(raw_output)[:150]}...")
            except Exception as e:
                print(f"⚠️ Error displaying raw output: {e}")

            gt_choice = item['label']
            # Robust string comparison with type checking
            try:
                pred_choice_str = str(pred_choice).strip().lower() if pred_choice is not None else ""
                gt_choice_str = str(gt_choice).strip().lower() if gt_choice is not None else ""
                is_correct = pred_choice_str == gt_choice_str
            except Exception as e:
                print(f"⚠️ Error comparing choices: {e}")
                is_correct = False
            
            if is_correct:
                correct_count += 1
            
            print(f"✅ Predicted Choice: {pred_choice}")
            print(f"🎯 Ground Truth: {gt_choice}")
            print(f"{'✓' if is_correct else '✗'} Correct: {is_correct}")
            
            # Format result
            result_item = {
                "idx": idx,
                "prompt": prompt,
                "raw_model_output": raw_output,
                "pred_choice_text": pred_choice,
                "gt_choice_text": gt_choice,
                "is_correct": is_correct
            }
            
            results.append(result_item)
            
        except Exception as e:
            print(f"❌ Error processing item {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add error result
            result_item = {
                "idx": idx,
                "prompt": prompt,
                "raw_model_output": f"Error: {str(e)}",
                "pred_choice_text": item['choices'][0] if item['choices'] else "",
                "gt_choice_text": item['label'],
                "is_correct": False
            }
            results.append(result_item)
    

    # Compute policy-based scores for EA (closeness) and aggregate total_points
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
                    emobird,
                    item,
                    r.get('pred_choice_text', ''),
                    r.get('gt_choice_text', ''),
                    r.get('raw_model_output', '')
                )
                r['policy_score'] = float(ps)
                r['policy_justification'] = pj
                r['final_score'] = float(ps)
                total_points += float(ps)
        except Exception as _e:
            # On any error, fall back to 0 for incorrect and 1 for correct
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
    avg_score = total_points / len(dataset) if dataset else 0.0

    accuracy = correct_count / len(dataset) if dataset else 0
    print(f"\n📊 EA Dataset Results:")
    print(f"   Total Items: {len(dataset)}")
    print(f"   Correct: {correct_count}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Average policy score: {avg_score:.4f} ({avg_score*100:.2f}%)")
    
    return results


def evaluate_eu_dataset(emobird: Emobird, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate EU dataset."""
    results = []
    correct_count = 0
    emo_correct_count = 0
    total_points = 0.0
    
    print(f"\n😊 Evaluating EU Dataset ({len(dataset)} items)")
    print("=" * 60)
    
    for idx, item in enumerate(dataset):
        print(f"\n📝 Item {idx + 1}/{len(dataset)}")
        
        # Format prompt
        prompt = format_eu_prompt(item)
        print(f"Prompt: {prompt[:100]}...")
        
        try:
            # Predict using strict choice selection via LLM
            pred_emo, pred_cause, raw_output = predict_eu_labels_with_choices(emobird, item)
            gt_emo = item['emotion_label']
            gt_cause = item['cause_label']
            
            # Robust string comparison with type checking
            try:
                pred_emo_str = str(pred_emo).strip().lower() if pred_emo is not None else ""
                gt_emo_str = str(gt_emo).strip().lower() if gt_emo is not None else ""
                pred_cause_str = str(pred_cause).strip().lower() if pred_cause is not None else ""
                gt_cause_str = str(gt_cause).strip().lower() if gt_cause is not None else ""
                
                is_emo_correct = pred_emo_str == gt_emo_str
                is_correct = is_emo_correct and pred_cause_str == gt_cause_str
            except Exception as e:
                print(f"⚠️ Error comparing EU predictions: {e}")
                is_emo_correct = False
                is_correct = False
            
            # Simple scoring: 1.0 for exact overall match, else 0.0
            policy_score = 1.0 if is_correct else 0.0
            total_points += policy_score
            
            if is_correct:
                correct_count += 1
            if is_emo_correct:
                emo_correct_count += 1
            
            print(f"😊 Predicted Emotion: {pred_emo}")
            print(f"🎯 Ground Truth Emotion: {gt_emo}")
            print(f"📋 Predicted Cause: {pred_cause}")
            print(f"🎯 Ground Truth Cause: {gt_cause}")
            print(f"{'✓' if is_emo_correct else '✗'} Emotion Correct: {is_emo_correct}")
            print(f"{'✓' if is_correct else '✗'} Overall Correct: {is_correct}")
            
            # Format result
            result_item = {
                "idx": idx,
                "prompt": prompt,
                "raw_model_output": raw_output,
                "pred_emo": pred_emo,
                "pred_cause": pred_cause,
                "gt_emo": gt_emo,
                "gt_cause": gt_cause,
                "is_correct": is_correct,
                "is_emo_correct": is_emo_correct,
                "final_score": policy_score
            }
            
            results.append(result_item)
            
        except Exception as e:
            print(f"❌ Error processing item {idx}: {e}")
            # Add error result
            result_item = {
                "idx": idx,
                "prompt": prompt,
                "raw_model_output": f"Error: {str(e)}",
                "pred_emo": "",
                "pred_cause": "",
                "gt_emo": item['emotion_label'],
                "gt_cause": item['cause_label'],
                "is_correct": False,
                "is_emo_correct": False
            }
            results.append(result_item)
    
    accuracy = correct_count / len(dataset) if dataset else 0
    emo_accuracy = emo_correct_count / len(dataset) if dataset else 0
    
    # Compute policy-based closeness scores for EU predictions (emotion + cause)
    # This mirrors EA's post-processing: exact matches get 1.0; otherwise use the policy scorer.
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
                    emobird,
                    item,
                    r.get('pred_emo', ''),
                    r.get('pred_cause', ''),
                    r.get('gt_emo', ''),
                    r.get('gt_cause', ''),
                    r.get('raw_model_output', '')
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
    avg_score = total_points / len(dataset) if dataset else 0.0
    
    print(f"\n📊 EU Dataset Results:")
    print(f"   Total Items: {len(dataset)}")
    print(f"   Emotion Correct: {emo_correct_count}")
    print(f"   Overall Correct: {correct_count}")
    print(f"   Emotion Accuracy: {emo_accuracy:.4f} ({emo_accuracy*100:.2f}%)")
    print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    # Average policy score across items (policy-based closeness)
    print(f"   Average policy score: {avg_score:.4f} ({avg_score*100:.2f}%)")
    
    return results


def main(limit_entries: int = None, quiet: bool = False):
    """Main evaluation function.
    
    Args:
        limit_entries: If specified, limit each dataset to this many entries for testing
        quiet: If True, suppress verbose EmoBIRD pipeline output
    """
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    print("🐦 EmoBIRD EmoBench Evaluation")
    if limit_entries:
        print(f"📊 Testing Mode: Limited to {limit_entries} entries per dataset")
    print("=" * 50)
    
    # Paths
    dataset_dir = Path("/mnt/shared/adarsh/datasets/EmoBench/data")
    eval_results_dir = Path("eval_results")
    eval_results_dir.mkdir(exist_ok=True)
    
    # Load datasets
    print("📂 Loading datasets...")
    ea_dataset = load_dataset(dataset_dir / "EA.jsonl")
    eu_dataset = load_dataset(dataset_dir / "EU.jsonl")
    
    # Limit datasets if specified
    if limit_entries:
        ea_dataset = ea_dataset[:limit_entries]
        eu_dataset = eu_dataset[:limit_entries]
        print(f"   📊 Limited to {limit_entries} entries each for testing")
    
    print(f"   EA Dataset: {len(ea_dataset)} items")
    print(f"   EU Dataset: {len(eu_dataset)} items")
    
    # Initialize EmoBIRD system
    print("🐦 Initializing EmoBIRD system...")
    emobird = Emobird()
    
    # Set verbose mode
    if hasattr(emobird, 'set_verbose'):
        emobird.set_verbose(not quiet)
    
    # Evaluate EA dataset
    ea_results = evaluate_ea_dataset(emobird, ea_dataset)
    
    # Evaluate EU dataset  
    eu_results = evaluate_eu_dataset(emobird, eu_dataset)
    
    # Save results
    print(f"\n💾 Saving results to {eval_results_dir}/")
    
    # Save EA results
    ea_output_path = eval_results_dir / "emobird_ea_results.json"
    with open(ea_output_path, 'w', encoding='utf-8') as f:
        json.dump(ea_results, f, indent=2, ensure_ascii=False)
    print(f"   EA results saved to: {ea_output_path}")
    
    # Save EU results
    eu_output_path = eval_results_dir / "emobird_eu_results.json"
    with open(eu_output_path, 'w', encoding='utf-8') as f:
        json.dump(eu_results, f, indent=2, ensure_ascii=False)
    print(f"   EU results saved to: {eu_output_path}")
    
    # Final summary
    print(f"\n🎉 Evaluation Complete!")
    print("=" * 50)
    
    # EA accuracy and score
    ea_correct = sum(1 for r in ea_results if r.get('is_correct'))
    ea_accuracy = ea_correct / len(ea_results) if ea_results else 0
    ea_avg_score = sum(float(r.get('final_score', 1.0 if r.get('is_correct') else 0.0)) for r in ea_results) / len(ea_results) if ea_results else 0.0
    print(f"📊 EA Dataset Final Accuracy: {ea_accuracy:.4f} ({ea_accuracy*100:.2f}%)")
    print(f"   Correct: {ea_correct}/{len(ea_results)}")
    print(f"   Average policy score: {ea_avg_score:.4f} ({ea_avg_score*100:.2f}%)")
    
    # EU accuracy and score
    eu_correct = sum(1 for r in eu_results if r.get('is_correct'))
    eu_emo_correct = sum(1 for r in eu_results if r.get('is_emo_correct'))
    eu_accuracy = eu_correct / len(eu_results) if eu_results else 0
    eu_emo_accuracy = eu_emo_correct / len(eu_results) if eu_results else 0
    eu_avg_score = sum(float(r.get('final_score', 1.0 if r.get('is_correct') else 0.0)) for r in eu_results) / len(eu_results) if eu_results else 0.0
    print(f"📊 EU Dataset Final Accuracy:")
    print(f"   Emotion Accuracy: {eu_emo_accuracy:.4f} ({eu_emo_accuracy*100:.2f}%)")
    print(f"   Overall Accuracy: {eu_accuracy:.4f} ({eu_accuracy*100:.2f}%)")
    print(f"   Emotion Correct: {eu_emo_correct}/{len(eu_results)}")
    print(f"   Average policy score: {eu_avg_score:.4f} ({eu_avg_score*100:.2f}%)")
    print(f"   Overall Correct: {eu_correct}/{len(eu_results)}")


if __name__ == "__main__":
    import sys
    
    # Check for arguments
    limit_entries = None
    quiet = False
    
    for arg in sys.argv[1:]:
        if arg == '--quiet' or arg == '-q':
            quiet = True
            print("🔇 Quiet mode enabled - suppressing verbose output")
        else:
            try:
                limit_entries = int(arg)
                print(f"📊 Running with limit: {limit_entries} entries per dataset")
            except ValueError:
                print(f"⚠️ Invalid argument: {arg}")
    
    # Default to 20 for quick testing
    if limit_entries is None:
        limit_entries = 20
        if not quiet:
            print("📊 Defaulting to 20 entries per dataset for quick testing")
    
    main(limit_entries, quiet)
