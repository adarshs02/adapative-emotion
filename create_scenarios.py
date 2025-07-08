import csv
import json

def create_scenarios_from_csv(csv_file_path, json_file_path):
    """
    Reads a CSV file with situations, generates scenarios with unique IDs,
    and writes them to a JSON file.

    Args:
        csv_file_path (str): The path to the input CSV file.
        json_file_path (str): The path to the output JSON file.
    """
    scenarios = []
    with open(csv_file_path, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    if len(rows) < 3:
        print("CSV file has fewer than 3 rows, not enough data to process.")
        return

    headers = rows[0]
    # Scenarios start from the 3rd row (index 2)
    scenario_descriptions = rows[2:]

    for col_idx, header in enumerate(headers):
        for row_idx, row in enumerate(scenario_descriptions):
            if col_idx < len(row) and row[col_idx]:
                description = row[col_idx]
                # A more descriptive ID
                scenario_id = f"{header.replace('-', '_').lower()}_scenario_{row_idx + 1}"
                scenarios.append({
                    "id": scenario_id,
                    "description": description.strip()
                })

    output_data = {"scenarios": scenarios}

    with open(json_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(output_data, outfile, indent=2)

    print(f"Successfully created {json_file_path} with {len(scenarios)} scenarios.")

if __name__ == '__main__':
    # Using the paths from the user's request
    csv_path = '/mnt/shared/adarsh/EmotionBench/situations/situations.csv'
    json_path = '/mnt/shared/adarsh/EmotionBench/situations/scenarios-emotionbench.json'
    create_scenarios_from_csv(csv_path, json_path)
