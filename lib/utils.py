import os
import re
import yaml
import json
from tabulate import tabulate
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import string

def load_yaml(file_path):
    data = {}
    with open(file_path) as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    return data


def parse_json_response(res):
    try:
        if res is None:
            raise ValueError("No response found")
        elif "```json" in res:
            regex = r"```json\s*([\s\S]*?)```"
            res = re.search(regex, res, re.DOTALL)
            res = res.group(0).replace("```json", "").replace("```", "")
        res = json.loads(res)
        return res
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return


def get_response_format(task, lang, use_cot):
    res_config = load_yaml("src/configs/response.yaml")
    res_format = """
{statement}
```json
    {{
    {conditions}
    }}
``` 
    """
    statement = res_config["cot"][lang] if use_cot else res_config["base"][lang]
    conditions = res_config[task][lang]
    if use_cot:
        conditions = res_config["reasoning"][lang] + ",\n" + conditions

    return res_format.format(statement=statement, conditions=conditions)


def save_gen_results(data, task, model_name, run_index=1):
    file_dir = f"results/{task}"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = f"{file_dir}/{model_name}_run{run_index}.jsonl"

    with open(file_name, "a+") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def save_eval_results(results, task, lang, model_name, run_index=1):
    file_name = f"results/{task}/leaderboard_run{run_index}.json"

    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            json.dump({model_name: {lang: results}}, f, ensure_ascii=False, indent=2)
    else:
        data = json.load(open(file_name, "r"))
        data[model_name] = data.get(model_name, {})
        data[model_name][lang] = results
        with open(file_name, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        "\n",
        tabulate(
            results.items(), headers=[f"{task} Category", "Accuracy"], tablefmt="grid"
        ),
        "\n",
    )


def normalize_text(text: str) -> str:
    """
    Normalizes a string by:
    1. Converting to lowercase.
    2. Removing punctuation (keeps apostrophes for contractions).
    3. Stripping leading/trailing whitespace.
    4. Replacing multiple spaces with a single space.
    """
    if not isinstance(text, str):
        return ""  # Or raise TypeError, or return text as is
    
    text = text.lower()
    
    # Create a translation table that removes all punctuation except '
    translator = str.maketrans('', '', ''.join(c for c in string.punctuation if c != "'"))
    text = text.translate(translator)
    
    text = text.strip()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text
