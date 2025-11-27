import json
import sys
import numpy as np
from collections import defaultdict

def main():
    input_file = "eval_results/simulation_results_evaluated.json"
    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    scores = defaultdict(lambda: defaultdict(list))

    for item in data:
        evals = item.get("evaluation", {})
        for model_name, result in evals.items():
            if "error" in result:
                continue
            
            model_scores = result.get("scores", {})
            for criterion, score in model_scores.items():
                # Handle potential string scores or out of bounds
                try:
                    s = float(score)
                    scores[model_name][criterion].append(s)
                except (ValueError, TypeError):
                    pass

    print("Average Scores:")
    print("-" * 30)
    
    for model_name, criteria in scores.items():
        print(f"Model: {model_name}")
        for criterion, values in criteria.items():
            avg = np.mean(values)
            print(f"  {criterion}: {avg:.2f} (n={len(values)})")
        print("-" * 30)

if __name__ == "__main__":
    main()
