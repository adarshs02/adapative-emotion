import pandas as pd
import json

# Path to the pickle file
pickle_path = '/mnt/shared/adarsh/data/emoknow/EMO-KNOW.pkl'

# Load the DataFrame from the pickle file
df = pd.read_pickle(pickle_path)

# Get the unique emotions from the 'emotion' column
unique_emotions = sorted(df['emotion'].unique().tolist())

# Print the list as a JSON array for easy parsing
print(json.dumps(unique_emotions))
