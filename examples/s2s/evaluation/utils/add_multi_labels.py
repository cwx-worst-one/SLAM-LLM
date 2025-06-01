import json
from datasets import load_dataset
from tqdm import tqdm

def add_labels_to_jsonl(input_path, output_path, dataset_name, split, cache_dir):
    """
    Adds label field to each line in a jsonl file using questions from a HuggingFace dataset.

    Args:
        input_path (str): Path to the input .jsonl file.
        output_path (str): Path to the output .jsonl file.
        dataset_name (str): Name of the HuggingFace dataset.
        split (str): Which split to use (e.g., 'test').
        cache_dir (str): Cache directory for dataset loading.
    """
    # Load the reference dataset
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for i, line in enumerate(tqdm(infile, desc="Processing lines")):
            data = json.loads(line)

            # Get the corresponding list of questions from the dataset
            questions = dataset[i]["answers"]
            if isinstance(questions, list):
                label_str = "|||".join(questions)
            else:
                label_str = str(questions)

            # Add the new label to the data
            data["label"] = label_str

            # Write the updated line to the output file
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Example usage
    add_labels_to_jsonl(
        input_path="/home/wenxi/mydisk/exp/standard_qa_eval/web_qa/qwen2.5-3b-instruct-new/web_qa_predictions.jsonl",
        output_path="/home/wenxi/mydisk/exp/standard_qa_eval/web_qa/qwen2.5-3b-instruct-new/web_qa_predictions_with_labels.jsonl",
        dataset_name="TwinkStart/speech-web-questions",
        split="test",
        cache_dir="/home/wenxi/mydisk/data/standard_qa_eval/web_qa"
    )
