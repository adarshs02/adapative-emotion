#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$REPO_ROOT")"

echo "Starting Full Evaluation Pipeline..."

# 1. Run Baseline Evaluation
echo "------------------------------------------------"
echo "Running Baseline Evaluation..."
python3 "$REPO_ROOT/EmoBIRDv2/eval_scripts/run_emopatient_multiturn_basemodel.py" "$@"

# 2. Run EmoBIRDv2 Evaluation
echo "------------------------------------------------"
echo "Running EmoBIRDv2 Evaluation..."
python3 "$REPO_ROOT/EmoBIRDv2/eval_scripts/run_emopatient_multiturn_emobirdv2.py" "$@"

# 3. Merge Outputs
echo "------------------------------------------------"
echo "Merging Outputs..."
python3 "$REPO_ROOT/EmoBIRDv2/scripts/merge_multiturn_outputs.py"

echo "------------------------------------------------"
echo "Pipeline Completed Successfully."
echo "Merged output: EmoBIRDv2/eval_results/emopatient_multiturn/emopatient_multiturn_merged_cleaned.json"
