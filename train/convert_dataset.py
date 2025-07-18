import json

def convert_dataset():
    """Convert manually edited train_dataset2.json to SFTTrainer format."""
    
    # Load the manually edited dataset
    with open('./train_dataset2.json', 'r') as f:
        data = json.load(f)
    
    # Convert to SFTTrainer format
    converted_data = []
    
    for item in data:
        tweet = item['tweet']
        emotions = item['emotions']
        
        # Create the prompt and response in the format SFTTrainer expects
        prompt = f"PROMPT: Given the tweet, generate a JSON object with the probability for each emotion.\nTweet: {tweet}\nRESPONSE:"
        response = json.dumps(emotions)
        
        # Combine into single text field
        text = f"{prompt} {response}"
        
        converted_data.append({"text": text})
    
    # Save the converted dataset
    with open('./train_dataset2_converted.json', 'w') as f:
        for item in converted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Converted {len(converted_data)} samples")
    print("✅ Saved to train_dataset2_converted.json")
    print("✅ Ready for SFTTrainer!")

if __name__ == "__main__":
    convert_dataset()
