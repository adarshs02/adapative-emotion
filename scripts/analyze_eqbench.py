import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# CONFIG
RESULTS_PATH = Path("/mnt/shared/adarsh/results/eqbench/eqbench_llama_results.json")
CSV_OUT = Path("/mnt/shared/adarsh/results/eqbench/eqbench_llama_analysis.csv")
N_WORST = 5

# Helper to parse scores from model output (robust to format)
def parse_scores(model_output, emotions):
    # Try to find the section after 'First pass scores:' or 'Revised scores:'
    scores = {e: np.nan for e in emotions}
    # 1. Try revised scores first
    match = re.search(r"Revised scores:(.*?)(?:\n\s*\n|\[End|$)", model_output, re.DOTALL|re.IGNORECASE)
    if not match:
        # 2. Try first pass
        match = re.search(r"First pass scores:(.*?)(?:\n\s*\n|Critique:|$)", model_output, re.DOTALL|re.IGNORECASE)
    if match:
        block = match.group(1)
        for emo in emotions:
            emo_match = re.search(rf"{re.escape(emo)}\s*:\s*(-?\d+(?:\.\d+)?)", block, re.IGNORECASE)
            if emo_match:
                scores[emo] = float(emo_match.group(1))
    return scores

def main():
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    rows = []
    all_emotions = set()
    # First pass: gather all unique emotion labels
    for entry in data:
        gt = entry["ground_truth"]
        for k in gt:
            if k.endswith("_score"): continue
            all_emotions.add(gt[k])
    all_emotions = sorted(list(all_emotions))

    # Second pass: parse, compare, collect rows
    for entry in data:
        gt = entry["ground_truth"]
        qid = entry["qid"]
        prompt = entry["prompt"]
        model_output = entry["model_output"]
        # Build emotion label -> score ground truth
        gt_scores = {gt[f"emotion{i+1}"]: gt[f"emotion{i+1}_score"] for i in range(4) if f"emotion{i+1}" in gt}
        # Parse model scores
        pred_scores = parse_scores(model_output, list(gt_scores.keys()))
        row = {"qid": qid, **{f"gt_{k}": v for k, v in gt_scores.items()}, **{f"pred_{k}": pred_scores[k] for k in gt_scores}}
        row["prompt"] = prompt
        row["model_output"] = model_output
        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute per-emotion MAE/RMSE
    metrics = {}
    for emo in all_emotions:
        gt_col = f"gt_{emo}"
        pred_col = f"pred_{emo}"
        if gt_col in df and pred_col in df:
            gt_vals = df[gt_col].values
            pred_vals = df[pred_col].values
            mask = ~np.isnan(pred_vals)
            mae = np.mean(np.abs(gt_vals[mask] - pred_vals[mask])) if np.any(mask) else np.nan
            rmse = np.sqrt(np.mean((gt_vals[mask] - pred_vals[mask]) ** 2)) if np.any(mask) else np.nan
            metrics[emo] = {"MAE": mae, "RMSE": rmse}

    # Overall metrics
    all_gt = []
    all_pred = []
    for emo in all_emotions:
        gt_col = f"gt_{emo}"
        pred_col = f"pred_{emo}"
        if gt_col in df and pred_col in df:
            mask = ~np.isnan(df[pred_col].values)
            all_gt.extend(df[gt_col].values[mask])
            all_pred.extend(df[pred_col].values[mask])
    overall_mae = np.mean(np.abs(np.array(all_gt) - np.array(all_pred))) if all_gt else np.nan
    overall_rmse = np.sqrt(np.mean((np.array(all_gt) - np.array(all_pred)) ** 2)) if all_gt else np.nan

    # Print summary
    print("Emotion    | MAE   | RMSE")
    print("-----------------------------")
    for emo in all_emotions:
        m = metrics.get(emo, {})
        print(f"{emo:<11} | {m.get('MAE', float('nan')):.3f} | {m.get('RMSE', float('nan')):.3f}")
    print(f"Overall    | {overall_mae:.3f} | {overall_rmse:.3f}")

    # Save CSV
    df.to_csv(CSV_OUT, index=False)
    print(f"\nCSV written to {CSV_OUT}")

    # Find N worst errors
    df["abs_error_sum"] = 0
    for emo in all_emotions:
        gt_col = f"gt_{emo}"
        pred_col = f"pred_{emo}"
        if gt_col in df and pred_col in df:
            df["abs_error_sum"] += np.abs(df[gt_col] - df[pred_col])
    df_sorted = df.sort_values("abs_error_sum", ascending=False)
    print(f"\nWorst {N_WORST} predictions:")
    for i, row in df_sorted.head(N_WORST).iterrows():
        gt_str = ', '.join(['{}: {}'.format(emo, row[f'gt_{emo}']) for emo in all_emotions if f'gt_{emo}' in row])
        pred_str = ', '.join(['{}: {}'.format(emo, row[f'pred_{emo}']) for emo in all_emotions if f'pred_{emo}' in row])
        print(f"QID {row['qid']}:\n  GT = {{ {gt_str} }}\n  Pred = {{ {pred_str} }}\n  abs_error_sum = {row['abs_error_sum']:.2f}\n  Model Output: {row['model_output'][:400]}...\n")

if __name__ == "__main__":
    main()
