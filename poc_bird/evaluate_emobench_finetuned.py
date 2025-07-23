#!/usr/bin/env python3
"""
Comprehensive Emobench evaluation using fine-tuned embedding model with tags.
Produces detailed logs in the exact format required for analysis.
"""

import sys
import re
import json
import datetime
import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import our fine-tuned system
from query_emotions import (
    get_finetuned_router, 
    load_llm_model, 
    get_factor_values_from_llm, 
    get_probabilities_for_factors,
    extract_json_from_response
)
import config

class FineTunedEmobenchEvaluator:
    """Evaluates fine-tuned model on Emobench dataset with detailed logging."""
    
    def __init__(self):
        print("🚀 Initializing Fine-tuned Emobench Evaluator...")
        
        # Load fine-tuned router (includes embedding model + HNSW index)
        self.router = get_finetuned_router()
        
        # Load LLM for factor extraction and choice prediction
        self.tokenizer, self.llm_model = load_llm_model()
        
        # Load scenarios for factor extraction
        with open("atomic-scenarios.json", 'r') as f:
            scenarios_data = json.load(f)
            self.all_scenarios = scenarios_data.get("scenarios", scenarios_data)
        
        self.cpt_dir = config.CPT_DIR
        print("✅ Fine-tuned evaluator initialized successfully!")
    
    def normalize_text(self, text):
        """Normalize text for comparison."""
        if not isinstance(text, str):
            return ""
        return text.strip().lower()
    
    def handle_unfamiliar_scenario_eu(self, situation: str, best_match: dict) -> dict:
        """Handle unfamiliar scenarios with similarity < 0.82 using fixed prompt to decoder LLM."""
        # Log to evaluation-specific unfamiliar scenarios file
        filename = "logs/evaluation_unfamiliar_scenarios.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        data = []
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = []
        
        # Check for duplicates - skip logging if same situation already exists
        for existing_entry in data:
            if existing_entry.get("situation", "").strip() == situation.strip():
                print(f"📋 Skipping duplicate evaluation unfamiliar scenario (already logged)")
                break  # Skip logging but continue with analysis
        else:
            # Only log if no duplicate was found (else clause of for loop)
            unfamiliar_entry = {
                "situation": situation,
                "best_match_id": best_match['scenario']['id'],
                "similarity_score": best_match['score'],
                "timestamp": "evaluation_run"
            }
            
            data.append(unfamiliar_entry)
            
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            except IOError:
                pass  # Continue evaluation even if logging fails
        
        # Fixed prompt for decoder LLM
        fixed_prompt = f"""Please read the following situation carefully and analyze the emotions involved. Pay close attention to all details and context.

Situation: {situation}

Based on your careful analysis, what emotions are likely present? Please provide your reasoning and the most probable emotions with confidence levels."""
        
        # Use LLM to analyze unfamiliar scenario
        try:
            inputs = self.tokenizer.encode(fixed_prompt, return_tensors="pt").to(config.get_device())
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Parse response for emotions (simplified)
            # Try to extract any emotion words from response
            emotion_words = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'happiness', 'delight', 'frustration', 'anxiety']
            detected_emotions = {}
            
            for emotion in emotion_words:
                if emotion.lower() in response.lower():
                    detected_emotions[emotion] = 0.5  # Default confidence
            
            if not detected_emotions:
                detected_emotions = {'Unknown': 1.0}
            
            return {
                "method": "unfamiliar_scenario_fallback",
                "situation": situation,
                "similarity_score": best_match['score'],
                "best_scenario_id": best_match['scenario']['id'],
                "best_scenario_description": best_match['scenario']['description'],
                "fixed_prompt": fixed_prompt,
                "llm_response": response.strip(),
                "probabilities": detected_emotions,
                "factor_values": {},
                "logged": True
            }
            
        except Exception as e:
            return {
                "method": "unfamiliar_scenario_fallback",
                "situation": situation, 
                "similarity_score": best_match['score'],
                "error": f"LLM analysis failed: {str(e)}",
                "probabilities": {'Unknown': 1.0},
                "logged": True
            }
    
    def predict_emotion_eu(self, situation):
        """
        Predict emotion for EU task using full fine-tuned pipeline.
        Returns detailed prediction log matching required format.
        """
        prediction_log = {
            "situation": situation,
            "best_scenario_id": None,
            "best_scenario_description": None,
            "similarity_score": None,
            "user_tags": [],
            "combined_query": None,
            "factor_values": {},
            "probabilities": {},
            "error": None
        }
        
        try:
            # Step 1: Find best scenario using fine-tuned model + tags
            print(f"🔍 Processing: {situation[:60]}...")
            scenario_matches = self.router.find_top_scenarios(situation, top_k=1)
            
            if not scenario_matches:
                prediction_log["error"] = "No scenarios found"
                return prediction_log
            
            best_match = scenario_matches[0]
            prediction_log["best_scenario_id"] = best_match["scenario"]["id"]
            prediction_log["best_scenario_description"] = best_match["scenario"]["description"]
            prediction_log["similarity_score"] = best_match["score"]
            prediction_log["user_tags"] = best_match.get("user_tags", [])
            prediction_log["combined_query"] = best_match.get("combined_query", situation)
            
            # Check similarity threshold for unfamiliar scenario fallback
            SIMILARITY_THRESHOLD = 0.82
            if prediction_log["similarity_score"] < SIMILARITY_THRESHOLD:
                return self.handle_unfamiliar_scenario_eu(situation, best_match)
            
            # Step 2: Load CPT for the matched scenario
            scenario_id = best_match["scenario"]["id"]
            cpt_path = os.path.join(self.cpt_dir, f"{scenario_id}.json")
            
            if not os.path.exists(cpt_path):
                prediction_log["error"] = f"CPT file not found: {cpt_path}"
                return prediction_log
            
            with open(cpt_path, 'r') as f:
                cpt_data = json.load(f)
            
            # Step 3: Extract factor values using LLM
            selected_factors = get_factor_values_from_llm(
                situation,
                best_match["scenario"]["description"],
                cpt_data['factors'],
                self.tokenizer,
                self.llm_model
            )
            
            if not selected_factors:
                prediction_log["error"] = "Failed to extract factor values"
                return prediction_log
            
            prediction_log["factor_values"] = selected_factors
            
            # Step 4: Get emotion probabilities from CPT
            probabilities = get_probabilities_for_factors(cpt_data, selected_factors)
            
            if probabilities:
                prediction_log["probabilities"] = probabilities
            else:
                prediction_log["error"] = "No matching CPT entry found for factors"
                
        except Exception as e:
            prediction_log["error"] = f"Pipeline error: {str(e)}"
            print(f"❌ Error processing situation: {e}")
        
        return prediction_log
    
    def predict_choice_ea(self, prompt):
        """
        Predict choice for EA task using LLM.
        Returns detailed prediction log matching required format.
        """
        prediction_log = {
            "prompt": prompt,
            "raw_model_output": None,
            "pred_choice_text": None,
            "error": None
        }
        
        try:
            # Generate response using LLM
            messages = [{"role": "user", "content": prompt}]
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(config.get_device())

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.6,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            prediction_log["raw_model_output"] = raw_output
            prediction_log["pred_choice_text"] = raw_output  # Will be processed later
            
        except Exception as e:
            prediction_log["error"] = f"LLM prediction error: {str(e)}"
            print(f"❌ Error generating choice: {e}")
        
        return prediction_log
    
    def evaluate_emobench(self, data_dir="/mnt/shared/adarsh/EmoBench/data"):
        """
        Run full Emobench evaluation on EU and EA tasks.
        """
        print("🧪 Starting Emobench Evaluation with Fine-tuned Model...")
        print("=" * 80)
        
        # Setup logging
        log_dir = Path("logs/emobench_finetuned")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        task_results = {}
        
        for task in ["EU", "EA"]:
            print(f"\n🎯 Evaluating {task} Task...")
            print("-" * 60)
            
            # Load data
            data_path = Path(data_dir) / f"{task}.jsonl"
            if not data_path.exists():
                print(f"❌ Data file not found: {data_path}")
                continue
            
            df = pd.read_json(str(data_path), lines=True, encoding="utf-8")
            df = df[df["language"] == "en"]  # English only
            print(f"📊 Processing {len(df)} samples...")
            
            # Initialize counters for accuracy
            correct = 0
            eu_emo_correct = 0  # Separate counter for EU emotion correctness
            unfamiliar_count = 0  # Counter for unfamiliar scenario fallbacks
            total_samples = len(df)
            
            task_logs = []
            
            for idx, sample in tqdm(df.iterrows(), total=len(df), desc=f"{task}-en"):
                sample_log = {
                    "sample_id": idx,
                    "ground_truth": {},
                    "prediction_log": {},
                    "is_correct": False
                }
                
                if task == "EU":
                    # Emotion Understanding task
                    situation = sample['scenario']
                    gt_emotion = sample['emotion_label']
                    gt_cause = sample.get('cause_label', '')  # Ground truth cause
                    
                    sample_log["ground_truth"] = {
                        "emotion": gt_emotion,
                        "cause": gt_cause
                    }
                    
                    # Run our fine-tuned pipeline
                    prediction_log = self.predict_emotion_eu(situation)
                    sample_log["prediction_log"] = prediction_log
                    
                    # Check if unfamiliar scenario fallback was used
                    if prediction_log.get("method") == "unfamiliar_scenario_fallback":
                        unfamiliar_count += 1
                        sample_log["used_fallback"] = True
                    else:
                        sample_log["used_fallback"] = False
                    
                    # Determine predicted emotion (highest probability)
                    pred_emotion = "Unknown"  # Default fallback
                    pred_cause = "Unknown"    # Default fallback
                    
                    # Extract predicted emotion from probabilities
                    if prediction_log.get("probabilities") and prediction_log["probabilities"]:
                        try:
                            pred_emotion = max(
                                prediction_log["probabilities"], 
                                key=prediction_log["probabilities"].get
                            )
                        except (ValueError, TypeError):
                            pred_emotion = "Error in emotion extraction"
                    
                    # Extract predicted cause from scenario match
                    if prediction_log.get("best_scenario_id"):
                        pred_cause = f"Matched scenario: {prediction_log['best_scenario_id']}"
                    elif prediction_log.get("best_scenario_description"):
                        pred_cause = f"Scenario: {prediction_log['best_scenario_description']}"
                    elif prediction_log.get("similarity_score") is not None:
                        pred_cause = f"Best match with similarity: {prediction_log.get('similarity_score', 0):.3f}"
                    else:
                        pred_cause = "No scenario match found"
                    
                    # Add prediction fields matching your format (ALWAYS SET)
                    sample_log["prompt"] = f"Scenario: {situation}\nAnswer:"
                    sample_log["raw_model_output"] = str(prediction_log.get("probabilities", {}))
                    sample_log["pred_emo"] = pred_emotion
                    sample_log["pred_cause"] = pred_cause
                    sample_log["gt_emo"] = gt_emotion
                    sample_log["gt_cause"] = gt_cause
                    sample_log["is_emo_correct"] = False
                    
                    # Check correctness
                    if pred_emotion:
                        sample_log["is_emo_correct"] = self.normalize_text(pred_emotion) == self.normalize_text(gt_emotion)
                        sample_log["is_correct"] = sample_log["is_emo_correct"]  # Overall correctness based on emotion
                        
                        # Track EU emotion correctness separately
                        if sample_log["is_emo_correct"]:
                            eu_emo_correct += 1
                            correct += 1
                
                elif task == "EA":
                    # Emotion Action task
                    scenario = sample["scenario"]
                    subject = sample["subject"]
                    choices = sample["choices"]
                    gt_choice = sample['label']
                    
                    # Build prompt
                    prompt = f"Scenario: {scenario}\nAs {subject}, how would you respond?\nChoices:\n"
                    for i, choice in enumerate(choices):
                        prompt += f"{chr(97+i)}) {choice}\n"
                    prompt += "Answer:"
                    
                    sample_log["ground_truth"] = {"choice": gt_choice}
                    sample_log["prompt"] = prompt
                    
                    # Run choice prediction
                    prediction_log = self.predict_choice_ea(prompt)
                    sample_log["prediction_log"] = prediction_log
                    
                    # Extract predicted choice
                    raw_pred = prediction_log.get("raw_model_output", "")
                    
                    # Parse choice from response
                    letter_to_choice_map = {chr(97 + i): choice.strip() for i, choice in enumerate(choices)}
                    pred_choice = raw_pred  # Default fallback
                    
                    # Method 1: Look for "Answer: X)" pattern first
                    answer_pattern = re.search(r"Answer:\s*([a-d])\)\s*([^\n]+)", raw_pred, re.IGNORECASE)
                    if answer_pattern:
                        selected_letter = answer_pattern.group(1).lower()
                        if selected_letter in letter_to_choice_map:
                            pred_choice = letter_to_choice_map[selected_letter]
                        else:
                            # Extract the text after the letter
                            pred_choice = answer_pattern.group(2).strip()
                    else:
                        # Method 2: Look for standalone letter choice (a, b, c, d)
                        model_selected_letter = re.search(r"\b([a-d])\)", raw_pred, re.IGNORECASE)
                        if model_selected_letter:
                            selected_letter = model_selected_letter.group(1).lower()
                            if selected_letter in letter_to_choice_map:
                                pred_choice = letter_to_choice_map[selected_letter]
                        else:
                            # Method 3: Try to match choice text directly in the response
                            for letter, choice_text in letter_to_choice_map.items():
                                if self.normalize_text(choice_text) in self.normalize_text(raw_pred):
                                    pred_choice = choice_text
                                    break
                    
                    sample_log["pred_choice_text"] = pred_choice
                    sample_log["gt_choice_text"] = gt_choice
                    
                    # Check correctness
                    sample_log["is_correct"] = self.normalize_text(pred_choice) == self.normalize_text(gt_choice)
                    if sample_log["is_correct"]:
                        correct += 1
                
                task_logs.append(sample_log)
            
            # Save results
            log_filename = log_dir / f"finetuned_{task}_results_{timestamp}.json"
            with open(log_filename, 'w') as f:
                json.dump(task_logs, f, indent=2)
            
            # Calculate accuracy
            accuracy = correct / len(df) if len(df) > 0 else 0.0
            
            # Store results (include EU emotion accuracy for EU task)
            task_results[task] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(df),
                "log_file": str(log_filename)
            }
            
            # Add EU emotion accuracy and unfamiliar count for EU task
            if task == "EU":
                eu_emo_accuracy = eu_emo_correct / len(df) if len(df) > 0 else 0.0
                unfamiliar_rate = unfamiliar_count / len(df) if len(df) > 0 else 0.0
                task_results[task]["eu_emo_accuracy"] = eu_emo_accuracy
                task_results[task]["eu_emo_correct"] = eu_emo_correct
                task_results[task]["unfamiliar_count"] = unfamiliar_count
                task_results[task]["unfamiliar_rate"] = unfamiliar_rate
            
            print(f"✅ {task} Results:")
            print(f"   Accuracy: {correct}/{len(df)} = {accuracy:.4f}")
            if task == "EU":
                print(f"   EU-emo Correct: {eu_emo_correct}/{len(df)} = {eu_emo_accuracy:.4f}")
                print(f"   Unfamiliar Scenarios: {unfamiliar_count}/{len(df)} = {unfamiliar_rate:.4f} (similarity < 0.82)")
            print(f"   Log saved: {log_filename}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("🏆 FINE-TUNED MODEL EMOBENCH RESULTS")
        print("=" * 80)
        
        for task, results in task_results.items():
            print(f"🎯 {task} Task:")
            print(f"   📊 Accuracy: {results['accuracy']:.4f}")
            print(f"   ✅ Correct: {results['correct']}/{results['total']}")
            print(f"   📄 Logs: {results['log_file']}")
            print()
        
        print("🚀 Evaluation complete! Check log files for detailed analysis.")
        return task_results


def main():
    """Main evaluation function."""
    evaluator = FineTunedEmobenchEvaluator()
    results = evaluator.evaluate_emobench()
    return results


if __name__ == "__main__":
    main()
