import json
import os
from tqdm import tqdm

def calculate_average_word_count(file_path, field_name='predict'):
    """
    Calculate average word count for the specified field from a JSON or JSONL file.

    Args:
        file_path (str): Path to input file (.json or .jsonl).
        field_name (str): Target field name to calculate word count.

    Returns:
        float: Average word count.
    """
    total_word_counts = 0
    valid_entries = 0

    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None

    try:
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing lines"):
                    try:
                        data = json.loads(line.strip())
                        if field_name in data and isinstance(data[field_name], str):
                            words = [w for w in data[field_name].split() if w]
                            total_word_counts += len(words)
                            valid_entries += 1
                    except (json.JSONDecodeError, TypeError):
                        continue
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                for data in tqdm(data_list, desc="Processing entries"):
                    if isinstance(data, dict) and field_name in data and isinstance(data[field_name], str):
                        words = [w for w in data[field_name].split() if w]
                        total_word_counts += len(words)
                        valid_entries += 1
        else:
            print("Unsupported file format. Please use .json or .jsonl.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return total_word_counts / valid_entries if valid_entries > 0 else 0.0

# Example usage
if __name__ == "__main__":
    input_path = "/valleblob/v-wenxichen/data/s2s/json/voice_assistant.json"
    field = "output"  # change this to your target field name  llama_qa, trivia_qa, web_qa
    avg_wc = calculate_average_word_count(input_path, field)

    if avg_wc is not None:
        print(f"Average word count in '{field}' field: {avg_wc:.2f}")
