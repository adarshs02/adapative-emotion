import pandas as pd
import os

def combine_panas_results(file_paths, model_names, output_path):
    """Combines PANAS scores from multiple model CSV files."""
    if len(file_paths) != len(model_names):
        print("Error: The number of file paths must match the number of model names.")
        return

    all_data = []
    common_emotions = None

    for i, file_path in enumerate(file_paths):
        model_name = model_names[i]
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return
        
        # Filter for 'Evoked' type scenarios if 'Type' column exists
        if 'Type' in df.columns:
            df = df[df['Type'] == 'Evoked'].copy()
        
        # Identify emotion columns (assuming they are not 'Type' or 'Scenario')
        potential_emotion_cols = [col for col in df.columns if col not in ['Type', 'Scenario']]
        
        # Use the first file's emotion columns as the standard
        if common_emotions is None:
            common_emotions = potential_emotion_cols
        elif set(potential_emotion_cols) != set(common_emotions):
            print(f"Warning: Emotion columns differ in {file_path}. Using emotions from the first file: {common_emotions}")
            # Select only common emotions, or handle as per specific requirement
            df = df[['Scenario'] + [col for col in common_emotions if col in df.columns]]

        # Rename emotion columns to include model name
        renamed_columns = {'Scenario': 'Scenario'}
        for col in common_emotions:
            if col in df.columns:
                renamed_columns[col] = f"{col}_{model_name}"
            else:
                print(f"Warning: Emotion column '{col}' not found in {file_path}. It will be missing for {model_name}.")
        df = df.rename(columns=renamed_columns)
        
        # Select only 'Scenario' and the renamed emotion columns
        columns_to_keep = ['Scenario'] + [f"{emo}_{model_name}" for emo in common_emotions if f"{emo}_{model_name}" in df.columns]
        all_data.append(df[columns_to_keep])

    if not all_data:
        print("No data to combine.")
        return

    # Merge all dataframes on 'Scenario'
    combined_df = all_data[0]
    for i in range(1, len(all_data)):
        combined_df = pd.merge(combined_df, all_data[i], on='Scenario', how='outer')

    # Reorder columns: Scenario, then grouped by emotion
    if common_emotions:
        ordered_columns = ['Scenario']
        for emotion in common_emotions:
            for model_name in model_names:
                col_name = f"{emotion}_{model_name}"
                if col_name in combined_df.columns:
                    ordered_columns.append(col_name)
        
        # Add any columns that might have been missed (e.g. if an emotion was only in later files)
        for col in combined_df.columns:
            if col not in ordered_columns:
                ordered_columns.append(col)
        combined_df = combined_df[ordered_columns]

    try:
        combined_df.to_csv(output_path, index=False)
        print(f"Successfully combined PANAS scores and saved to {output_path}")
    except IOError as e:
        print(f"Error writing to {output_path}: {e}")

def main():
    base_dir = "/mnt/shared/adarsh/results/emotionbench/"
    
    files = [
        os.path.join(base_dir, "llama3.1-8b-PANAS-testing.csv"),
        os.path.join(base_dir, "qwen-PANAS-testing.csv"),
        os.path.join(base_dir, "mistral-PANAS-testing.csv")
    ]
    
    # Short model names for column suffixes
    models = ["Llama", "Qwen", "Mistral"]
    
    output_file = os.path.join(base_dir, "combined_PANAS_model_scores.csv")
    
    combine_panas_results(files, models, output_file)

if __name__ == "__main__":
    main()
