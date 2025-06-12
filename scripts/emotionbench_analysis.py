import json
from collections import defaultdict
from pathlib import Path
import numpy as np

RESULTS_PATH = Path("/mnt/shared/adarsh/results/emotionbench/emotionbench_llama_results0.6.json")

# PANAS-X emotions as they appear in the benchmark prompts
# This mapping is based on the prompt text seen in your command history.
PANAS_X_ITEMS = [
    "Hostile", "Active", "Interested", "Excited", "Inspired", "Alert",
    "Jittery", "Distressed", "Guilty", "Proud", "Strong", "Afraid",
    "Attentive", "Determined", "Enthusiastic", "Scared", "Irritable",
    "Upset", "Ashamed", "Nervous"
]

# Standard PANAS-X scales
POSITIVE_AFFECT_EMOTIONS = {"Active", "Interested", "Excited", "Inspired", "Alert", "Proud", "Strong", "Attentive", "Determined", "Enthusiastic"}
NEGATIVE_AFFECT_EMOTIONS = {"Hostile", "Jittery", "Distressed", "Guilty", "Afraid", "Scared", "Irritable", "Upset", "Ashamed", "Nervous"}

# Get indices for each scale
positive_indices = [i for i, emotion in enumerate(PANAS_X_ITEMS) if emotion in POSITIVE_AFFECT_EMOTIONS]
negative_indices = [i for i, emotion in enumerate(PANAS_X_ITEMS) if emotion in NEGATIVE_AFFECT_EMOTIONS]

with open(RESULTS_PATH, "r") as f:
    data = json.load(f)

# Aggregate statistics
results_by_type = defaultdict(lambda: {"positive": [], "negative": [], "count": 0})
all_scores = []

for entry in data:
    scores = entry.get("scores")
    entry_type = entry.get("Type", "Unknown")

    if not scores or len(scores) != 20:
        print(f"Skipping entry with invalid scores: {entry.get('Scenario', 'No Scenario')}")
        continue

    scores_arr = np.array(scores)
    
    # Calculate scores for each scale
    positive_score = scores_arr[positive_indices].sum()
    negative_score = scores_arr[negative_indices].sum()

    results_by_type[entry_type]["positive"].append(positive_score)
    results_by_type[entry_type]["negative"].append(negative_score)
    results_by_type[entry_type]["count"] += 1
    all_scores.append(scores_arr)

print("EmotionBench Analysis Results")
print("="*30)
print(f"Total entries processed: {len(data)}")
print("\n--- Analysis by Type ---")

for type, stats in results_by_type.items():
    count = stats['count']
    avg_pos = np.mean(stats['positive'])
    avg_neg = np.mean(stats['negative'])
    std_pos = np.std(stats['positive'])
    std_neg = np.std(stats['negative'])
    
    print(f"\nType: {type} ({count} entries)")
    print(f"  - Average Positive Affect Score: {avg_pos:.2f} (SD: {std_pos:.2f})")
    print(f"  - Average Negative Affect Score: {avg_neg:.2f} (SD: {std_neg:.2f})")

# Overall analysis
if all_scores:
    all_scores_arr = np.array(all_scores)
    avg_scores_per_item = np.mean(all_scores_arr, axis=0)
    
    print("\n--- Overall Item Analysis ---")
    print("Average score for each emotion item (1-5 scale):")
    for i, emotion in enumerate(PANAS_X_ITEMS):
        print(f"  - {emotion:<15}: {avg_scores_per_item[i]:.2f}")

print("\nAnalysis complete.")
