import pandas as pd
# import requests # No longer needed
import torch # For device checking
from collections import defaultdict
from scripts_utils import ModelInitializer, print_gpu_info
import re

# ---------------------- Config ----------------------
# VLLM_URL = "http://localhost:8000/v1/chat/completions" # No longer needed
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "../data/EMO-KNOW-v1.pkl"
TEMPERATURE = 0.0
MAX_TOKENS = 100
NUM_SAMPLES = 100000
FILTER_EMOTION = "tired"  # Change this to filter by a different emotion
# ----------------------------------------------------

# ---------------------- Functions ----------------------

# Normalize text to lowercase, remove punctuation, and map synonyms
def normalize(text):
    text = re.sub(r'[^\w\s]', '', text.strip().lower())
    synonyms = {
        "angry": "anger",
        "sad": "sadness",
        "happy": "joy",
        "fearful": "fear",
        "disgusted": "disgust",
        "surprised": "surprise",
        "concentrated": "concentration",
        "grateful": "gratitude",
    }
    return synonyms.get(text, text)

# ---- GPU Info & Model Init ----
print_gpu_info()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

client = ModelInitializer(
    model_name=MODEL_NAME,
    device=DEVICE,
    default_max_new_tokens=MAX_TOKENS,
    default_temperature=TEMPERATURE,
    default_do_sample=(TEMPERATURE > 0.0) # Sample if temperature > 0
)

# Load and filter dataset
try:
    df = pd.read_pickle(DATA_PATH)
except FileNotFoundError:
    print(f"❌ File not found at path: {DATA_PATH}")
    exit()

df = df[df["emotion"].str.lower() == FILTER_EMOTION.lower()].head(NUM_SAMPLES)
if df.empty:
    print(f"❌ No entries found for emotion: {FILTER_EMOTION}")
    exit()

total = len(df)
correct_emotion = 0
correct_cause = 0

emotion_stats = defaultdict(lambda: {"correct": 0, "total": 0})
cause_stats = defaultdict(lambda: {"correct": 0, "total": 0})
detailed_results = []

# query_vllm function removed, using ModelInitializer instead.

# Evaluation loop
for idx, row in df.iterrows():
    tweet = row["tweet"]
    gold_emotion = normalize(row["emotion"])
    gold_cause = normalize(str(row["cause"]))

    prompt = f"""
Given the following tweet, identify the emotion being expressed and its cause.

Tweet: "{tweet}"

First, state the emotion in one word.

Second, briefly state the cause of this emotion in a phrase.

Format your response as:
Emotion: <emotion>
Cause: <cause>
""".strip()

    try:
        response_dict = client.gen_response(prompt)
        response_text = response_dict['completion']
        lines = response_text.lower().splitlines()
        pred_emotion = normalize(next((l.split(":", 1)[1] for l in lines if "emotion" in l), ""))
        pred_cause = normalize(next((l.split(":", 1)[1] for l in lines if "cause" in l), ""))

        is_emotion_correct = gold_emotion == pred_emotion
        is_cause_correct = gold_cause in pred_cause or pred_cause in gold_cause

    except Exception as e:
        print(f"[{idx+1}/{total}] ❌ ERROR: {e}")
        pred_emotion = "error"
        pred_cause = "error"
        is_emotion_correct = False
        is_cause_correct = False

    emotion_stats[gold_emotion]["total"] += 1
    cause_stats[gold_emotion]["total"] += 1

    if is_emotion_correct:
        emotion_stats[gold_emotion]["correct"] += 1
        correct_emotion += 1
    if is_cause_correct:
        cause_stats[gold_emotion]["correct"] += 1
        correct_cause += 1

    print(f"\n[{idx+1}/{total}] Tweet: {tweet}")
    print(f"Pred → Emotion: {pred_emotion} | Cause: {pred_cause}")
    print(f"Gold → Emotion: {gold_emotion} | Cause: {gold_cause}")
    print(f"Result → Emotion: {'✅' if is_emotion_correct else '❌'} | Cause: {'✅' if is_cause_correct else '❌'}")

    # Save per-tweet result
    detailed_results.append({
        "Tweet": tweet,
        "Gold Emotion": gold_emotion,
        "Predicted Emotion": pred_emotion,
        "Emotion Correct": is_emotion_correct,
        "Gold Cause": gold_cause,
        "Predicted Cause": pred_cause,
        "Cause Correct": is_cause_correct
    })

# Accuracy Summary
emotion_accuracy = correct_emotion / total * 100
cause_accuracy = correct_cause / total * 100
print(f"\n✅ Emotion Accuracy: {emotion_accuracy:.2f}%")
print(f"✅ Cause Accuracy: {cause_accuracy:.2f}%")

# Save per-emotion accuracy
results_df = pd.DataFrame({
    "Emotion": list(emotion_stats.keys()),
    "Correct": [v["correct"] for v in emotion_stats.values()],
    "Total": [v["total"] for v in emotion_stats.values()]
})
results_df["Accuracy"] = (results_df["Correct"] / results_df["Total"]) * 100
results_df.to_csv("emotion_accuracy.csv", index=False)
print("✅ Emotion results saved to emotion_accuracy.csv")

# Save cause accuracy
cause_results_df = pd.DataFrame({
    "Emotion": list(cause_stats.keys()),
    "Correct": [v["correct"] for v in cause_stats.values()],
    "Total": [v["total"] for v in cause_stats.values()]
})
cause_results_df["Accuracy"] = (cause_results_df["Correct"] / cause_results_df["Total"]) * 100
cause_results_df.to_csv("cause_accuracy.csv", index=False)
print("✅ Cause results saved to cause_accuracy.csv")

# Save detailed predictions
detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv("detailed_predictions.csv", index=False)
print("✅ Detailed tweet-level predictions saved to detailed_predictions.csv")
