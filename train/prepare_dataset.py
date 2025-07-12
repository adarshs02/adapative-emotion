import pandas as pd
from datasets import Dataset
import os

# --- Configuration ---
DATA_PATH = "/mnt/shared/adarsh/data/emoknow/EMO-KNOW.pkl"
TRAIN_FILE = "./train_dataset.json"
EVAL_FILE = "./eval_dataset.json"

def format_prompt(sample):
    """Formats a sample from the EmoKnow dataset into a training prompt."""
    return f"<s>[INST] Given the tweet: '{sample['tweet']}', identify the emotion. [/INST] Emotion: {sample['emotion']}</s>"

def main():
    """Loads, splits, and formats the EmoKnow dataset for training and evaluation."""
    print("--- Starting Dataset Preparation ---")
    
    # --- 1. Load Raw Data ---
    if not os.path.exists(DATA_PATH):
        print(f"\u274c EMO-KNOW.pkl not found at {DATA_PATH}. Please check the path.")
        return

    print(f"Loading raw data from {DATA_PATH}...")
    df = pd.read_pickle(DATA_PATH)

    # --- 2. Shuffle and Split ---
    print("Shuffling and splitting data...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df.head(10000)
    eval_df = df.iloc[10000:12000]

    print(f"- {len(train_df)} samples for training")
    print(f"- {len(eval_df)} samples for evaluation")

    # --- 3. Format and Save Training Set ---
    print(f"Formatting and saving training set to {TRAIN_FILE}...")
    train_dataset = Dataset.from_pandas(train_df)
    # We need to format the prompt for the SFTTrainer's text field
    formatted_train_dataset = train_dataset.map(lambda sample: {"text": format_prompt(sample)})
    formatted_train_dataset.to_json(TRAIN_FILE, orient='records', lines=True)

    # --- 4. Save Evaluation Set ---
    print(f"Saving evaluation set to {EVAL_FILE}...")
    # The evaluation script will handle its own formatting, so we save the raw records
    eval_df.to_json(EVAL_FILE, orient='records', lines=True)

    print("\n\u2728 Dataset preparation complete! \u2728")
    print(f"Training data saved to: {TRAIN_FILE}")
    print(f"Evaluation data saved to: {EVAL_FILE}")
    print("-------------------------------------")

if __name__ == "__main__":
    main()
