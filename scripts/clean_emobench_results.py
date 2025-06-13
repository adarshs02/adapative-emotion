import json
import os

def clean_prompt(prompt_text):
    """Removes the few-shot examples from the prompt."""
    marker = "--- End of examples ---"
    parts = prompt_text.split(marker)
    if len(parts) > 1:
        # The actual prompt is the last part, after the marker.
        # We also need to strip leading/trailing whitespace and newlines.
        scenario_prompt = parts[-1].strip()
        return scenario_prompt
    return prompt_text # Return original if marker not found

def clean_raw_model_output(raw_output_text):
    """Extracts the primary JSON response from the raw model output."""
    if not isinstance(raw_output_text, str):
        return raw_output_text

    marker = "--- End of examples ---"
    parts = raw_output_text.split(marker)
    
    content_after_marker = parts[-1] if len(parts) > 1 else raw_output_text

    search_text = content_after_marker.strip()
    
    idx = 0
    while idx < len(search_text):
        json_start_index = search_text.find('{', idx)
        if json_start_index == -1:
            break

        balance = 0
        potential_json_end = -1
        for i in range(json_start_index, len(search_text)):
            char = search_text[i]
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
                if balance == 0:
                    potential_json_end = i
                    break
        
        if potential_json_end != -1:
            json_candidate_str = search_text[json_start_index : potential_json_end + 1]
            try:
                json.loads(json_candidate_str)
                return json_candidate_str
            except json.JSONDecodeError:
                idx = json_start_index + 1 
        else:
            idx = json_start_index + 1
            
    return content_after_marker.strip() if content_after_marker else raw_output_text


def clean_results_file(input_path, output_path):
    """Reads a results JSON file, cleans the prompts, and saves to a new file."""
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {input_path}: {e}")
        return

    cleaned_data = []
    for item in data:
        if 'prompt' in item and isinstance(item['prompt'], str):
            item['prompt'] = clean_prompt(item['prompt'])
        if 'raw_model_output' in item and isinstance(item['raw_model_output'], str):
            item['raw_model_output'] = clean_raw_model_output(item['raw_model_output'])
        cleaned_data.append(item)

    try:
        with open(output_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        print(f"Successfully cleaned file and saved to {output_path}")
    except IOError as e:
        print(f"Error writing to {output_path}: {e}")

def main():
    """Main function to clean all specified emobench result files."""
    results_dir = "/mnt/shared/adarsh/results/emobench/EU/"
    files_to_clean = [
        "emobench_mistralai_Mistral-7B-Instruct-v0.3_EU_en_results.json",
        "emobench_Qwen_Qwen2.5-7B-Instruct_EU_en_results.json",
        "emobench_meta-llama_Llama-3.1-8B-Instruct_EU_en_results.json"
    ]

    for filename in files_to_clean:
        input_file_path = os.path.join(results_dir, filename)
        output_filename = filename.replace("_results.json", "_results_cleaned.json")
        output_file_path = os.path.join(results_dir, output_filename)
        
        print(f"Processing {input_file_path}...")
        clean_results_file(input_file_path, output_file_path)

if __name__ == "__main__":
    main()
