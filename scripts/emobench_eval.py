import sys
import re
import json
from pathlib import Path
import torch
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent # Should be /mnt/shared/adarsh
lib_dir = project_root / "lib"
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from data import DataLoader
from utils import load_yaml, normalize_text
from scripts_utils import get_model_name, ModelInitializer, print_gpu_info
import pandas as pd
from tqdm import tqdm

# ---- Main ----
if __name__ == "__main__":
    # ---- Script Setup & Initializations ----
    
    # Model and Device Configuration
    MODEL_NAME = get_model_name()
    print_gpu_info() # Display GPU info early
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Model
    client = ModelInitializer(
        MODEL_NAME,
        DEVICE,
        default_max_new_tokens=512, # Default from previous scripts_utils.py
        default_temperature=0.6,    # Default from previous scripts_utils.py
        default_do_sample=False     # Default from previous scripts_utils.py
    )

    # Define Core Paths (project_root is globally defined and absolute)
    DATA_DIR = project_root / "EmoBench" / "data"
    RESULTS_BASE_DIR = project_root / "results" / "emobench"
    RESULTS_BASE_DIR.mkdir(parents=True, exist_ok=True) # Ensure base results/ directory exists

    # Initialize score tracking
    task_accuracies = {}
    task_emo_accuracies = {} # For EU emotion-only accuracy

    FEW_SHOT_EXAMPLES_EU = [
        {
            "scenario": "Lena has been trying to study hard the entire week for her crucial medical exam, but has not found the chance because of her social life. So, with high anxiety, she decided to prepare hours away from the exam. In the middle of her study, she got a notification about the cancellation of the exam due to a virus outbreak in her college.",
            "answer": {"emo_label": "Relief", "cause_label": "The medical exam was cancelled due to a virus outbreak"}
        },
        {
            "scenario": "After waiting for almost two hours for his date, Roger was considering leaving when he heard a familiar voice behind him. It was his date, Jessica, who frantically apologized for being late and explained she had been helping a family stuck on the highway.",
            "answer": {"emo_label": "Admiration", "cause_label": "Jessica was late because she was helping a family in need"}
        }
    ]

    for task in ["EA", "EU"]:
        for lang in ["en"]:
            results = []  # Initialize results list for each task-language pair
            # client = LlamaModelClient(MODEL_NAME, DEVICE) # Moved instantiation out of the loop
            data_path = str(DATA_DIR / f"{task}.jsonl")
            df = pd.read_json(data_path, lines=True, encoding="utf-8")
            df = df[df["language"] == lang]
            print(f"\nEvaluating {task}-{lang} ({len(df)} samples):\n")
            correct = 0
            emo_correct = 0 # Counter for emotion-only accuracy in EU task

            # Create few-shot prompt for EU task
            few_shot_prompt = ""
            if task == "EU":
                few_shot_prompt = "Please identify the emotion and its cause from the scenario. Respond ONLY in JSON format.\n\n"
                for ex in FEW_SHOT_EXAMPLES_EU:
                    few_shot_prompt += f"Scenario: {ex['scenario']}\n"
                    few_shot_prompt += f"Answer: {json.dumps(ex['answer'])}\n\n"
                few_shot_prompt += "--- End of examples ---\n\n"
            for idx, sample in tqdm(df.iterrows(), total=len(df), desc=f"{task}-{lang}"):
                if task == "EA":
                    # Build prompt from scenario, subject, and choices
                    scenario = sample["scenario"]
                    subject = sample["subject"]
                    choices = sample["choices"]
                    prompt = f"Scenario: {scenario}\nAs {subject}, how would you respond?\nChoices:\n"
                    for i, choice in enumerate(choices):
                        prompt += f"{chr(97+i)}) {choice}\n"
                    prompt += "Answer:"
                elif task == "EU":
                    scenario = sample["scenario"]
                    prompt = few_shot_prompt + f"Scenario: {scenario}\nAnswer:"
                else: # Fallback for other tasks, if any
                    prompt = sample["prompt"] if "prompt" in sample else sample.get("input", "")

                response_dict = client.gen_response(prompt, task=task)
                raw_pred = response_dict['completion'].strip() # Used by EA, EU uses fields from response_dict

                if task == "EA":
                    gt = sample['label'] # Ground truth is the text of the correct answer

                    # 1. Get the mapping from letter to choice text directly from sample data
                    choices_list_from_sample = sample['choices']
                    letter_to_choice_text_map = {}
                    for i, choice_text_val in enumerate(choices_list_from_sample):
                        letter_char = chr(97 + i)  # 'a', 'b', 'c', ...
                        letter_to_choice_text_map[letter_char] = choice_text_val.strip()

                    # 2. Determine model's selected letter from raw_pred
                    # Ensure raw_pred is defined from model output before this block (e.g., raw_pred = response_dict['completion'].strip())
                    model_selected_letter = None
                    m = re.search(r"(?:Answer:\s*)?([a-d])(?:\)|\b)", raw_pred, re.IGNORECASE)
                    if m:
                        model_selected_letter = m.group(1).lower()
                    
                    pred = raw_pred  # Fallback: Initialize pred with the full raw model output.
                                   # This will be overwritten if a more specific match is found.

                    if model_selected_letter and model_selected_letter in letter_to_choice_text_map:
                        pred = letter_to_choice_text_map[model_selected_letter]  # Use exact choice text for the matched letter
                    else:
                        # Fallback: If no specific letter (a, b, c, d) is identified in the model's output,
                        # or if the letter is not in our map,
                        # try to see if the model's raw output starts with any of the known choice texts.
                        normalized_raw_pred_full = normalize_text(raw_pred)
                        # Iterate through known choices to find a match
                        for _choice_letter_key, choice_text_from_map in letter_to_choice_text_map.items():
                            normalized_choice_from_map = normalize_text(choice_text_from_map)
                            if normalized_raw_pred_full.startswith(normalized_choice_from_map):
                                pred = choice_text_from_map  # Use the original casing from sample['choices']
                                break  # Found a match, stop searching
                    
                    pred_norm = normalize_text(pred)
                    gt_norm = normalize_text(gt)
                    is_correct = pred_norm == gt_norm
                    # Record result for EA task analysis
                    results.append({
                        "idx": int(idx),
                        "prompt": prompt,
                        "raw_model_output": response_dict.get("full_raw_output", ""),
                        "pred_choice_text": pred,
                        "gt_choice_text": gt,
                        "is_correct": is_correct
                    })
                elif task == "EU":
                    # Directly use pre-parsed labels from response_dict
                    # The gen_response method in scripts_utils.py already handles JSON parsing and fallbacks
                    model_emo_label = response_dict.get("emo_label", "").strip()
                    model_cause_label = response_dict.get("cause_label", "").strip()

                    # The following debug print is now primarily handled in scripts_utils.py's gen_response
                    # print(f"DEBUG EU Raw Completion from response_dict['completion']: >>>{response_dict.get('completion', '').strip()}<<< DEBUG END")

                    # Evaluate correctness
                    gt_emo_label = sample['emotion_label'].strip() # Use correct column name from data.py
                    gt_cause_label = sample['cause_label'].strip() # Use correct column name from data.py

                    is_correct = (model_emo_label.lower() == gt_emo_label.lower()) and (model_cause_label.lower() == gt_cause_label.lower())

                    # Emotion-only accuracy
                    is_emo_correct = (model_emo_label.lower() == gt_emo_label.lower())
                    if is_emo_correct:
                        emo_correct += 1
                    print(f"Sample {idx}:\nPrompt: {prompt}\nModel: emo_label={model_emo_label}, cause_label={model_cause_label}\nGT: emo_label={gt_emo_label}, cause_label={gt_cause_label}\nCorrect: {is_correct}, Emo Correct: {is_emo_correct}\n{'-'*40}")
                    # Record result for analysis
                    results.append({
                        "idx": int(idx),
                        "prompt": prompt,
                        "raw_model_output": response_dict.get("full_raw_output", ""),
                        "pred_emo": model_emo_label,
                        "pred_cause": model_cause_label,
                        "gt_emo": gt_emo_label, # Use updated variable
                        "gt_cause": gt_cause_label, # Use updated variable
                        "is_correct": is_correct,
                        "is_emo_correct": is_emo_correct
                    })
                if is_correct:
                    correct += 1

            accuracy = correct / len(df) if len(df) > 0 else 0
            task_accuracies[(task, lang)] = accuracy
            print(f"\n{task}-{lang} Combined Accuracy: {correct}/{len(df)} = {accuracy:.4f}")

            if task == "EU":
                emo_accuracy = emo_correct / len(df) if len(df) > 0 else 0
                task_emo_accuracies[(task, lang)] = emo_accuracy
                print(f"{task}-{lang} Emotion-Only Accuracy: {emo_correct}/{len(df)} = {emo_accuracy:.4f}\n")
            # Save results if any were collected for the current task and language
            if results:
                # Sanitize MODEL_NAME for use in filename (e.g., replace '/')
                safe_model_name = MODEL_NAME.replace('/', '_') 
                results_filename = f"emobench_{safe_model_name}_{task}_{lang}_results.json"
                
                # Define the directory to save results
                results_dir = RESULTS_BASE_DIR / task
                results_dir.mkdir(parents=True, exist_ok=True) # Create 'results/TASK_TYPE' dir if not exists
                
                save_path = results_dir / results_filename

                with open(save_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nResults for {task}-{lang} ({len(results)} samples) saved to {save_path}")
            else:
                print(f"\nNo results to save for {task}-{lang}.")
    print("\n--- Overall Model Scores ---")
    if task_accuracies:
        for (task_name, lang_name), acc in task_accuracies.items():
            print(f"  {MODEL_NAME} - {task_name}-{lang_name} Combined Accuracy: {acc:.4f}")
        for (task_name, lang_name), acc in task_emo_accuracies.items():
            print(f"  {MODEL_NAME} - {task_name}-{lang_name} Emotion-Only Accuracy: {acc:.4f}")
    else:
        print("  No tasks were evaluated to show a summary.")
    print("\nDone.")
