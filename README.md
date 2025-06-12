# Emotional Intelligence Benchmarks

This repository contains various implementations and evaluations for emotion-related benchmarks for language models, including:

- **EmotionBench**: Tests a model's ability to recognize emotions in different scenarios
- **EmoBench**: Emotional intelligence benchmark for language models
- **EQ-Bench**: Emotional intelligence quotient benchmark for language models
- **SECEU**: Situational Evaluation of Complex Emotional Understanding

## Structure

- `scripts/`: Contains evaluation scripts for the different benchmarks
- `lib/`: Utility functions used across different scripts
- `data/`: Data files used by the benchmarks
- `results/`: Output directory for evaluation results
- `EmotionBench/`, `EmoBench/`, `EQ-Bench/`: Source benchmark implementations

## Dependencies

See `requirements.txt` for the required Python packages.

## Usage

Each benchmark can be run via its respective evaluation script in the `scripts/` directory:

```bash
# For EmotionBench
python scripts/emotionbench_eval.py

# For EmoBench
python scripts/emobench_eval.py

# For EQ-Bench
python scripts/eqbench_eval.py

# For Security and Emotional Understanding evaluation
python scripts/seceu_eval.py
```

## Notes

- The `ModelInitializer` class in `scripts/scripts_utils.py` handles model loading and inference
- Results contain both `full_raw_output` (including the prompt) and `completion` (only the model's response)
