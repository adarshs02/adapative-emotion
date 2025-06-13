import json
import os

INPUT_MODEL_PATH = "/mnt/shared/adarsh/results/eqbench/eqbench_mistral_results.json"

def extract_scores(text_block):
    scores = {}
    for line in text_block.strip().split("\n"):
        if ":" in line:
            emotion, score_text_original = line.split(":", 1) # Split only on the first colon
            score_text = score_text_original.strip()

            if not score_text: # If the part after colon is empty or just whitespace, skip
                continue

            try:
                # Take the first "word" from the score text
                numeric_part = score_text.split(' ')[0]
                # Try to convert to float then int, to handle "9.5" -> 9 (by truncation)
                scores[emotion.strip()] = int(float(numeric_part))
            except ValueError:
                # This will be triggered if numeric_part is not a number (e.g. "<revised", "I've")
                print(f"Warning: Could not parse score for '{emotion.strip()}'. Raw score part: '{score_text_original.strip()}'")
    return scores

def compute_eq_score(model_path):
    with open(model_path, "r") as f:
        results = json.load(f)

    total_score = 0
    count = 0

    for entry in results:
        gt = entry["ground_truth"]
        model_output = entry["model_output"]

        # Extract revised scores from model output
        revised_section = model_output.split("Revised scores:")[-1].split("[End of answer]")[0]
        model_scores = extract_scores(revised_section)

        # Extract ground truth scores
        gt_scores = {
            gt["emotion1"]: gt["emotion1_score"],
            gt["emotion2"]: gt["emotion2_score"],
            gt["emotion3"]: gt["emotion3_score"],
            gt["emotion4"]: gt["emotion4_score"],
        }

        # Match order of emotions in ground truth
        l1_dist = 0
        for emo in gt_scores:
            pred = model_scores.get(emo, 0)
            l1_dist += abs(pred - gt_scores[emo])

        eq_score = 100 - (l1_dist / 40 * 100)  # max L1 distance is 4 * 10 = 40
        total_score += eq_score
        count += 1

    avg_eq_score = total_score / count if count > 0 else 0
    return avg_eq_score

def calculate_and_save_eq_score(input_model_path_arg):
    score = compute_eq_score(input_model_path_arg)

    output_dir = os.path.dirname(input_model_path_arg)
    base_name = os.path.basename(input_model_path_arg)
    
    # Construct output filename
    if "_results" in base_name:
        output_base_name = base_name.replace("_results", "_score")
    else:
        name_part, ext_part = os.path.splitext(base_name)
        output_base_name = f"{name_part}_score{ext_part}"
        
    output_file_path = os.path.join(output_dir, output_base_name)
    
    output_data = {"eq_bench_score": f"{score:.2f}"} # Store score as formatted string
    
    with open(output_file_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ EQ-Bench Score: {score:.2f}")
    print(f"💾 Score saved to: {output_file_path}")
    # return output_file_path # Optionally return path

if __name__ == "__main__":
    # This allows eqbench_score.py to be run directly
    # using the INPUT_MODEL_PATH defined at the top of the file.
    if not INPUT_MODEL_PATH:
        print("Error: INPUT_MODEL_PATH is not set or is empty in eqbench_score.py. Cannot run directly.")
    elif not os.path.exists(INPUT_MODEL_PATH):
        print(f"Error: INPUT_MODEL_PATH '{INPUT_MODEL_PATH}' does not exist. Cannot run directly.")
    else:
        print(f"Running eqbench_score.py directly for: {INPUT_MODEL_PATH}")
        calculate_and_save_eq_score(INPUT_MODEL_PATH)
