import pandas as pd
import requests
from collections import defaultdict
import re

# ---------------------- Config ----------------------
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "../data/EMO-KNOW-v1.pkl"  # Make sure this path is correct
TEMPERATURE = 0.0
MAX_TOKENS = 100
NUM_SAMPLES = 10000 # number of samples, dataset has 700,000 samples
# ----------------------------------------------------

def normalize(text):
    return re.sub(r'[^\w\s]', '', text.strip().lower())

# Emotion groups
emotion_grouping = {
    # Positive
    "joy": "positive",
    "love": "positive",
    "gratitude": "positive",
    "relief": "positive",
    "hope": "positive",
    "pride": "positive",
    "happy": "positive",
    "grateful": "positive",
    "lucky": "positive",
    "proud": "positive",
    "excited": "positive",

    # Negative
    "anger": "negative",
    "sadness": "negative",
    "fear": "negative",
    "guilt": "negative",
    "shame": "negative",
    "envy": "negative",
    "disgust": "negative",
    "upset": "negative",
    "nervous": "negative",
    "lonely": "negative",
    "jealous": "negative",
    "depressed": "negative",

    # Ambiguous
    "surprise": "ambiguous",
    "confused": "ambiguous",
    "shocked": "ambiguous"
}


# Load dataset
try:
    df = pd.read_pickle(DATA_PATH).head(NUM_SAMPLES)
except FileNotFoundError:
    print(f"❌ File not found at path: {DATA_PATH}")
    exit()

# Setup
total = len(df)
correct_emotion = 0
correct_cause = 0
emotion_stats = defaultdict(lambda: {"correct": 0, "total": 0})
cause_stats = defaultdict(lambda: {"correct": 0, "total": 0})
group_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# Query vLLM
def query_vllm(prompt):
    response = requests.post(
        VLLM_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        }
    )
    return response.json()["choices"][0]["message"]["content"].strip()

# Evaluation loop
for idx, row in df.iterrows():
    tweet = row["tweet"]
    gold_emotion = normalize(row["emotion"])
    gold_cause = normalize(str(row["cause"]))

    grouped_emotions = ", ".join(sorted(emotion_grouping.keys()))

    prompt = f"""
Given the following tweet, identify the emotion being expressed and its cause.

Tweet: "{tweet}"

First, state the emotion using only one word from the following list: {grouped_emotions}

Second, briefly state the cause of this emotion in a phrase.

Format your response as:
Emotion: <emotion>
Cause: <cause>
""".strip()

    try:
        response = query_vllm(prompt)
        lines = response.lower().splitlines()
        pred_emotion = normalize(next((l.split(":", 1)[1] for l in lines if "emotion" in l), ""))
        pred_cause = normalize(next((l.split(":", 1)[1] for l in lines if "cause" in l), ""))

        gold_group = emotion_grouping.get(gold_emotion, "unknown")
        pred_group = emotion_grouping.get(pred_emotion, "unknown")
        is_emotion_correct = gold_group == pred_group
        is_cause_correct = gold_cause in pred_cause or pred_cause in gold_cause

    except Exception as e:
        print(f"[{idx+1}/{total}] ❌ ERROR: {e}")
        pred_emotion = "error"
        pred_cause = "error"
        gold_group = pred_group = "unknown"
        is_emotion_correct = False
        is_cause_correct = False

    emotion_stats[gold_emotion]["total"] += 1
    cause_stats[gold_emotion]["total"] += 1
    group_stats[gold_group]["total"] += 1

    if is_emotion_correct:
        emotion_stats[gold_emotion]["correct"] += 1
        correct_emotion += 1
        group_stats[gold_group]["correct"] += 1
    if is_cause_correct:
        cause_stats[gold_emotion]["correct"] += 1
        correct_cause += 1

    print(f"[{idx+1}/{total}] Emotion: {pred_emotion} ({pred_group}) | Gold: {gold_emotion} ({gold_group}) | "
          f"Cause: {pred_cause} | {'✅' if is_emotion_correct else '❌'} | {'✅' if is_cause_correct else '❌'}")

# Summary
emotion_accuracy = correct_emotion / total * 100
cause_accuracy = correct_cause / total * 100

print(f"\n✅ Emotion Accuracy (by group): {emotion_accuracy:.2f}%")
print(f"✅ Cause Accuracy: {cause_accuracy:.2f}%")

# Print per-emotion accuracy
print("\n📊 Per-Emotion Accuracy (Emotion | Cause):")
for emotion in sorted(emotion_stats.keys()):
    e_stats = emotion_stats[emotion]
    c_stats = cause_stats[emotion]
    e_acc = (e_stats["correct"] / e_stats["total"] * 100) if e_stats["total"] > 0 else 0
    c_acc = (c_stats["correct"] / c_stats["total"] * 100) if c_stats["total"] > 0 else 0
    print(f"{emotion:<12}: {e_stats['correct']}/{e_stats['total']} ({e_acc:.2f}%) | "
          f"{c_stats['correct']}/{c_stats['total']} ({c_acc:.2f}%)")

# Print per-group accuracy
print("\n📊 Per-Group Accuracy:")
for group, stats in group_stats.items():
    acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    print(f"{group:<10}: {stats['correct']}/{stats['total']} correct ({acc:.2f}%)")

# Save CSV
emotion_df = pd.DataFrame.from_dict(emotion_stats, orient="index")
emotion_df["emotion_accuracy"] = (emotion_df["correct"] / emotion_df["total"] * 100).round(2)
cause_df = pd.DataFrame.from_dict(cause_stats, orient="index")
cause_df["cause_accuracy"] = (cause_df["correct"] / cause_df["total"] * 100).round(2)
results_df = emotion_df.join(cause_df["cause_accuracy"])
results_df.reset_index(inplace=True)
results_df.rename(columns={"index": "emotion"}, inplace=True)
results_df.to_csv("../data/emo_know_grouped_results.csv", index=False)

print("\n📁 Results saved to ../data/emo_know_grouped_results.csv")



# #VLLM code to start the server
# python3 -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --max-model-len 4096 \
#   --gpu-memory-utilization 0.85 \
#   --trust-remote-code \
#   --port 8000 