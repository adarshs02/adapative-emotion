#!/usr/bin/env python3
"""
Script to combine multiple evaluation JSON files into a single JSON file.
Adds a 'method_name' key at the sample (qa) level to distinguish between methods.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_method_name_to_samples(data: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """
    Add method_name key to each qa sample in the data structure.
    
    Args:
        data: The JSON data containing scenarios and qa items
        method_name: The name of the method to add to each sample
    
    Returns:
        Modified data structure with method_name added to each qa item
    """
    if 'scenarios' not in data:
        return data
    
    for scenario in data['scenarios']:
        if 'qa' in scenario:
            for qa_item in scenario['qa']:
                qa_item['method_name'] = method_name
    
    return data


def combine_eval_files(input_dir: Path, output_file: Path, file_configs: List[Dict[str, str]], 
                       max_scenarios: int = None, max_questions_per_scenario: int = None) -> None:
    """
    Combine multiple evaluation JSON files into a single file.
    
    Args:
        input_dir: Directory containing the input JSON files
        output_file: Path to the output combined JSON file
        file_configs: List of dicts with 'filename' and 'method_name' keys
        max_scenarios: Maximum number of scenarios to include (None for all)
        max_questions_per_scenario: Maximum questions per scenario per method (None for all)
    """
    combined_data = {
        "scenarios": []
    }
    
    # Track scenarios by ID to merge qa items
    scenario_map = {}
    
    for config in file_configs:
        filename = config['filename']
        method_name = config['method_name']
        filepath = input_dir / filename
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        print(f"Processing {filename} with method_name='{method_name}'...")
        data = load_json_file(filepath)
        data_with_method = add_method_name_to_samples(data, method_name)
        
        # Limit scenarios if specified
        scenarios_to_process = data_with_method.get('scenarios', [])
        if max_scenarios is not None:
            scenarios_to_process = scenarios_to_process[:max_scenarios]
        
        # Merge scenarios
        for scenario in scenarios_to_process:
            scenario_id = scenario.get('id')
            
            if scenario_id not in scenario_map:
                # First time seeing this scenario, add it
                scenario_map[scenario_id] = {
                    'id': scenario.get('id'),
                    'title': scenario.get('title'),
                    'diagnosis': scenario.get('diagnosis'),
                    'treatment_plan': scenario.get('treatment_plan'),
                    'narrative': scenario.get('narrative'),
                    'qa': []
                }
            
            # Get qa items, limit if specified
            qa_items = scenario.get('qa', [])
            if max_questions_per_scenario is not None:
                qa_items = qa_items[:max_questions_per_scenario]
            
            # Add qa items from this scenario
            scenario_map[scenario_id]['qa'].extend(qa_items)
    
    # Convert map to list
    combined_data['scenarios'] = list(scenario_map.values())
    
    # Write combined data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCombined {len(file_configs)} files into: {output_file}")
    print(f"Total scenarios: {len(combined_data['scenarios'])}")
    total_samples = sum(len(s['qa']) for s in combined_data['scenarios'])
    print(f"Total samples: {total_samples}")


def main():
    """Main function to run the combination script."""
    # Configuration
    input_dir = Path("/Users/adarsh/myProjects/thesis/adapative-emotion/EmoBIRDv2/RECAP_EVALS")
    output_file = Path("/Users/adarsh/myProjects/thesis/adapative-emotion/EmoBIRDv2/RECAP_EVALS/emopatient_combined.json")
    
    # Define the files to combine and their method names
    file_configs = [
        {'filename': 'emopatient_RECAP.json', 'method_name': 'RECAP'},
        {'filename': 'emopatient_basemodel.json', 'method_name': 'basemodel'},
        {'filename': 'emopatient_eprime.json', 'method_name': 'eprime'},
        {'filename': 'emopatient_persona.json', 'method_name': 'persona'},
    ]
    
    # Combine the files (limited to 25 scenarios and 1 question per scenario)
    combine_eval_files(input_dir, output_file, file_configs, 
                      max_scenarios=25, max_questions_per_scenario=1)


if __name__ == "__main__":
    main()
