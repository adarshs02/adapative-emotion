import pickle
import pandas as pd

file_path = '/mnt/shared/adarsh/data/emoknow/EMO-KNOW.pkl'

print(f"Loading pickle file from: {file_path}")
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"Data type: {type(data)}")

if isinstance(data, list) and data:
    print(f"Number of items: {len(data)}")
    print("First item:")
    print(data[0])
elif isinstance(data, dict) and data:
    print(f"Keys: {list(data.keys())}")
    # Print the first key-value pair to inspect structure
    first_key = list(data.keys())[0]
    print(f"First item (key: '{first_key}'):")
    print(data[first_key])
