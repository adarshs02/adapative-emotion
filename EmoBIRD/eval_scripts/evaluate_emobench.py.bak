"""
EmoBench Evaluation Script for EmoBIRD

Evaluates EmoBIRD framework on EmoBench EA (Empathy Actions) and EU (Emotion Understanding) datasets.
Provides real-time output and saves results in specified JSON format.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any
import sys
import atexit

# Add EmoBIRD modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emobird_poc import Emobird


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
    
    # Look for choice letters (a, b, c, d)
    choice_pattern = r'[Aa]nswer:\s*([a-d])\)'
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
    try:
        # Try to extract JSON from response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_match = re.search(json_pattern, response)
        
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            emotion = parsed.get('emo_label', parsed.get('emotion', ''))
            cause = parsed.get('cause_label', parsed.get('cause', ''))
            
            return emotion, cause
    except:
        pass
    
    # Fallback parsing
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


def evaluate_ea_dataset(emobird: Emobird, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate EA dataset using EmoBIRD emotional insights."""
    results = []
    correct_count = 0
    
    print(f"\n🎭 Evaluating EA Dataset ({len(dataset)} items)")
    print("=" * 60)
    
    for idx, item in enumerate(dataset):
        print(f"\n📝 Item {idx + 1}/{len(dataset)}")
        
        # Format prompt for display
        prompt = format_ea_prompt(item)
        print(f"📋 Scenario: {item['scenario'][:100]}...")
        
        try:
            # Generate choice using EmoBIRD emotional insights
            print("🧠 Running EmoBIRD emotional analysis...")
            raw_output = generate_ea_choice_with_emotions(
                emobird, item['scenario'], item['subject'], item['choices']
            )
            
            # Safely display raw output with proper type checking
            try:
                if isinstance(raw_output, str):
                    print(f"💭 Raw reasoning output: {raw_output[:150]}...")
                else:
                    print(f"💭 Raw reasoning output: {str(raw_output)[:150]}...")
            except Exception as e:
                print(f"⚠️ Error displaying raw output: {e}")
            
            # Parse prediction
            pred_choice = parse_ea_response(raw_output, item['choices'])
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
    
    accuracy = correct_count / len(dataset) if dataset else 0
    print(f"\n📊 EA Dataset Results:")
    print(f"   Total Items: {len(dataset)}")
    print(f"   Correct: {correct_count}")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return results


def evaluate_eu_dataset(emobird: Emobird, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate EU dataset."""
    results = []
    correct_count = 0
    emo_correct_count = 0
    
    print(f"\n😊 Evaluating EU Dataset ({len(dataset)} items)")
    print("=" * 60)
    
    for idx, item in enumerate(dataset):
        print(f"\n📝 Item {idx + 1}/{len(dataset)}")
        
        # Format prompt
        prompt = format_eu_prompt(item)
        print(f"Prompt: {prompt[:100]}...")
        
        try:
            # Run EmoBIRD pipeline
            result = emobird.analyze_emotion(item['scenario'])
            
            # Extract top emotions and create EU-style response
            emotion_probs = result.get('emotions', {})
            if emotion_probs:
                # Get the top emotion (highest probability)
                top_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
                # Simple cause extraction from scenario context
                cause = f"Context from the scenario: {item['scenario'][:50]}..."
            else:
                top_emotion = "Neutral"
                cause = "Unable to determine cause"
            
            # Format as JSON response
            raw_output = f'{{"emo_label": "{top_emotion}", "cause_label": "{cause}"}}'
            
            # Parse prediction
            pred_emo, pred_cause = parse_eu_response(raw_output)
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
                "is_emo_correct": is_emo_correct
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
    
    print(f"\n📊 EU Dataset Results:")
    print(f"   Total Items: {len(dataset)}")
    print(f"   Emotion Correct: {emo_correct_count}")
    print(f"   Overall Correct: {correct_count}")
    print(f"   Emotion Accuracy: {emo_accuracy:.4f} ({emo_accuracy*100:.2f}%)")
    print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
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
    
    # EA accuracy
    ea_correct = sum(1 for r in ea_results if r['is_correct'])
    ea_accuracy = ea_correct / len(ea_results) if ea_results else 0
    print(f"📊 EA Dataset Final Accuracy: {ea_accuracy:.4f} ({ea_accuracy*100:.2f}%)")
    print(f"   Correct: {ea_correct}/{len(ea_results)}")
    
    # EU accuracy
    eu_correct = sum(1 for r in eu_results if r['is_correct'])
    eu_emo_correct = sum(1 for r in eu_results if r['is_emo_correct'])
    eu_accuracy = eu_correct / len(eu_results) if eu_results else 0
    eu_emo_accuracy = eu_emo_correct / len(eu_results) if eu_results else 0
    print(f"📊 EU Dataset Final Accuracy:")
    print(f"   Emotion Accuracy: {eu_emo_accuracy:.4f} ({eu_emo_accuracy*100:.2f}%)")
    print(f"   Overall Accuracy: {eu_accuracy:.4f} ({eu_accuracy*100:.2f}%)")
    print(f"   Emotion Correct: {eu_emo_correct}/{len(eu_results)}")
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
