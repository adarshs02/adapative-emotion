import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys
import re

# Add lib/ to path for imports
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent # Should be /mnt/shared/adarsh
lib_dir = project_root / "lib"
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from data import DataLoader
from utils import load_yaml, save_gen_results, save_eval_results, normalize_text
from scripts_utils import get_model_name, ModelInitializer, print_gpu_info

# ---- Main ----
if __name__ == "__main__":
    MODEL_NAME = get_model_name()
    print_gpu_info() # Call the imported function
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate ModelInitializer once before the loops
    client = ModelInitializer(
        MODEL_NAME, 
        DEVICE,
        default_max_new_tokens=512, # Default from previous scripts_utils.py
        default_temperature=0.6,    # Default from previous scripts_utils.py
        default_do_sample=False     # Default from previous scripts_utils.py
    )

    # Choose task and language
    import pandas as pd
    from tqdm import tqdm

    for task in ["EA", "EU"]:
        results = []  # <-- Fix: initialize results list for each task
        for lang in ["en"]:
            # client = LlamaModelClient(MODEL_NAME, DEVICE) # Moved instantiation out of the loop
            data_path = str(project_root / "data" / f"{task}.jsonl")
            df = pd.read_json(data_path, lines=True, encoding="utf-8")
            df = df[df["language"] == lang]
            print(f"\nEvaluating {task}-{lang} ({len(df)} samples):\n")
            correct = 0
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
                    prompt = sample["scenario"] # Use scenario for EU task
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
                elif task == "EU":
                    completion = response_dict['completion'].strip()
                    print(f"DEBUG EU Completion: >>>{completion}<<< DEBUG END")

                    emo_labels = []
                    cause_labels = []
                    try:
                        # Use regex to find all JSON objects in the completion string.
                        # This handles cases where the model returns multiple JSONs.
                        json_strs = re.findall(r'\{.*?\}', completion, re.DOTALL)

                        for json_str in json_strs:
                            try:
                                parsed_json = json.loads(json_str)
                                emo_label = parsed_json.get("emo_label", "").strip()
                                cause_label = parsed_json.get("cause_label", "").strip()
                                if emo_label:
                                    emo_labels.append(emo_label)
                                if cause_label:
                                    cause_labels.append(cause_label)
                            except json.JSONDecodeError:
                                continue # Ignore malformed JSON objects

                    except Exception:
                        pass # Keep labels empty if parsing fails

                    model_emo_label = " & ".join(emo_labels)
                    model_cause_label = " & ".join(cause_labels)

                    # Evaluate correctness
                    gt_emo_label = sample['emotion_label'].strip() # Use correct column name from data.py
                    gt_cause_label = sample['cause_label'].strip() # Use correct column name from data.py

                    is_correct = (model_emo_label == gt_emo_label) and (model_cause_label == gt_cause_label)
                    print(f"Sample {idx}:\nPrompt: {prompt}\nModel: emo_label={model_emo_label}, cause_label={model_cause_label}\nGT: emo_label={gt_emo_label}, cause_label={gt_cause_label}\nCorrect: {is_correct}\n{'-'*40}")
                    # Record result for analysis
                    results.append({
                        "idx": int(idx),
                        "prompt": prompt,
                        "raw_model_output": response_dict.get("full_raw_output", ""),
                        "pred_emo": model_emo_label,
                        "pred_cause": model_cause_label,
                        "gt_emo": gt_emo_label, # Use updated variable
                        "gt_cause": gt_cause_label, # Use updated variable
                        "is_correct": is_correct
                    })
                if is_correct:
                    correct += 1
            print(f"\n{task}-{lang} Accuracy: {correct}/{len(df)} = {correct/len(df):.4f}\n")
            # Save results for EU task
            if task == "EU":
                import json
                with open("emobench_llama_results.json", "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    print("Done.")
