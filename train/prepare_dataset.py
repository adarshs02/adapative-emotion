import pandas as pd
import json
import os
from emotions import EMOTION_LIST

# --- Configuration ---
DATA_PATH = "/mnt/shared/adarsh/data/emoknow/EMO-KNOW.pkl"
TRAIN_FILE = "./train_dataset.json"
EVAL_FILE = "./eval_dataset.json"


def format_training_sample(sample):
    """Formats a single sample for SFTTrainer with a simplified prompt."""
    # Create the target JSON object for the response
    emotion_distribution = {emotion: 0.0 for emotion in EMOTION_LIST}
    if sample['emotion'] in emotion_distribution:
        emotion_distribution[sample['emotion']] = 1.0

    # Simplified prompt and response structure
    prompt = f"PROMPT: Given the tweet, generate a JSON object with the probability for each emotion.\nTweet: {sample['tweet']}\nRESPONSE:"
    response = json.dumps(emotion_distribution)

    # The SFTTrainer expects a single 'text' field containing both prompt and response.
    return f"{prompt} {response}"


def main():
    """Loads, splits, and formats the EmoKnow dataset."""
    print("--- Starting Dataset Preparation ---")

    if not os.path.exists(DATA_PATH):
        print(f"\u274c EMO-KNOW.pkl not found at {DATA_PATH}.")
        return

    print(f"Loading raw data from {DATA_PATH}...")
    df = pd.read_pickle(DATA_PATH)

    print("Shuffling and splitting data...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df.head(10000)
    eval_df = df.iloc[10000:10500]

    print(f"- {len(train_df)} samples for training")
    print(f"- {len(eval_df)} samples for evaluation")

    print(f"Formatting and saving training set to {TRAIN_FILE}...")
    with open(TRAIN_FILE, 'w') as f:
        for _, row in train_df.iterrows():
            if row['emotion'] in EMOTION_LIST:
                formatted_text = format_training_sample(row)
                # The trainer expects a JSON object with a 'text' key
                f.write(json.dumps({"text": formatted_text}) + "\n")

    print(f"Saving evaluation set to {EVAL_FILE}...")
    # For evaluation, we save the raw data. The eval script will format it.
    eval_df.to_json(EVAL_FILE, orient='records', lines=True)

    print("\n\u2728 Dataset preparation complete! \u2728")
    print(f"Training data: {TRAIN_FILE}")
    print(f"Evaluation data: {EVAL_FILE}")
    print("-------------------------------------")

if __name__ == "__main__":
    main()
