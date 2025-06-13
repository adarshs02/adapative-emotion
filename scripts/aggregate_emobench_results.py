import json
import os
from scripts_utils import get_model_name

TASKS = ["EA", "EU"]
MODEL_NAME = get_model_name()
LANG = "en"
N_RUNS = 5

for task in TASKS:
    scores = []
    for run in range(1, N_RUNS + 1):
        leaderboard_path = f"results/{task}/leaderboard_run{run}.json"
        if not os.path.exists(leaderboard_path):
            print(f"[Warning] {leaderboard_path} not found.")
            continue
        with open(leaderboard_path, "r") as f:
            data = json.load(f)
        try:
            score = data[MODEL_NAME][LANG]["Overall"]
            scores.append(score)
            print(f"{task} Run {run}: {score:.4f}")
        except Exception as e:
            print(f"[Error] Could not extract score from {leaderboard_path}: {e}")
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n{task} Average over {len(scores)} runs: {avg:.4f}\n")
    else:
        print(f"No scores found for {task}.")
