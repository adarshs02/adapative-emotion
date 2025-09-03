#!/usr/bin/env python3
"""
Aggregate EmoPatient judge results.

Reads a results JSON (either a flat list or nested under scenarios[].qa[]),
then computes:
- Win% for RECAP and baseline (over all items and over decided only)
- Average score_baseline, average score_recap
- Average margin

Usage:
  python scripts/aggregate_emopatient_results.py \
    --input /mnt/shared/adarsh/datasets/EmoPatient/results/oss20b_judged.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Union


def _load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Case 1: already a flat list
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    # Case 2: object with one list value
    if isinstance(data, dict):
        list_keys = [k for k, v in data.items() if isinstance(v, list)]
        if len(list_keys) == 1:
            key = list_keys[0]
            lst = data[key]
            # Special: nested scenarios[].qa[] (reconstructed output keeps original shape)
            if key.lower() == "scenarios" and all(isinstance(x, dict) for x in lst):
                flat: List[Dict[str, Any]] = []
                for s in lst:
                    qa_list = s.get("qa")
                    if isinstance(qa_list, list):
                        for qa in qa_list:
                            if isinstance(qa, dict):
                                flat.append(qa)
                return flat
            # Fallback: single list value
            return [x for x in lst if isinstance(x, dict)]
    raise ValueError("Unsupported input format: expected a list or an object with one list value (optionally scenarios[].qa[]).")


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def aggregate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = 0
    n_recap = 0
    n_baseline = 0
    n_tie = 0

    sum_base = 0.0
    sum_recap = 0.0
    sum_margin = 0.0
    cnt_base = 0
    cnt_recap = 0
    cnt_margin = 0

    for row in items:
        n_total += 1
        winner = str(row.get("winner", "")).strip().lower()
        if winner == "recap":
            n_recap += 1
        elif winner == "baseline":
            n_baseline += 1
        elif winner == "tie":
            n_tie += 1
        # scores
        if "score_baseline" in row:
            v = _to_float(row.get("score_baseline"))
            if v == v:  # not NaN
                sum_base += v
                cnt_base += 1
        if "score_recap" in row:
            v = _to_float(row.get("score_recap"))
            if v == v:
                sum_recap += v
                cnt_recap += 1
        if "margin" in row:
            v = _to_float(row.get("margin"))
            if v == v:
                sum_margin += v
                cnt_margin += 1

    decided = n_recap + n_baseline
    pct_recap_all = (100.0 * n_recap / n_total) if n_total else 0.0
    pct_base_all = (100.0 * n_baseline / n_total) if n_total else 0.0
    pct_recap_decided = (100.0 * n_recap / decided) if decided else 0.0
    pct_base_decided = (100.0 * n_baseline / decided) if decided else 0.0

    avg_base = (sum_base / cnt_base) if cnt_base else float("nan")
    avg_recap = (sum_recap / cnt_recap) if cnt_recap else float("nan")
    avg_margin = (sum_margin / cnt_margin) if cnt_margin else float("nan")

    return {
        "n_total": n_total,
        "n_recap": n_recap,
        "n_baseline": n_baseline,
        "n_tie": n_tie,
        "pct_recap_all": pct_recap_all,
        "pct_baseline_all": pct_base_all,
        "pct_recap_decided": pct_recap_decided,
        "pct_baseline_decided": pct_base_decided,
        "avg_score_baseline": avg_base,
        "avg_score_recap": avg_recap,
        "avg_margin": avg_margin,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate EmoPatient judge results (win% and averages)")
    ap.add_argument(
        "--input",
        default="/mnt/shared/adarsh/datasets/EmoPatient/results/oss20b_judged.json",
        help="Path to judged results JSON",
    )
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text summary")
    args = ap.parse_args()

    items = _load_items(args.input)
    stats = aggregate(items)

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    # Human-readable summary
    def fmt(x: float) -> str:
        return f"{x:.2f}" if isinstance(x, float) else str(x)

    print("EmoPatient Results Summary")
    print("---------------------------")
    print(f"Total items:           {stats['n_total']}")
    print(f"RECAP wins:            {stats['n_recap']}  ({fmt(stats['pct_recap_all'])}% of all, {fmt(stats['pct_recap_decided'])}% of decided)")
    print(f"Baseline wins:         {stats['n_baseline']}  ({fmt(stats['pct_baseline_all'])}% of all, {fmt(stats['pct_baseline_decided'])}% of decided)")
    print(f"Ties:                  {stats['n_tie']}")
    print()
    print(f"Average score (RECAP):   {fmt(stats['avg_score_recap'])}")
    print(f"Average score (Baseline):{fmt(stats['avg_score_baseline'])}")
    print(f"Average margin:          {fmt(stats['avg_margin'])}")


if __name__ == "__main__":
    main()
