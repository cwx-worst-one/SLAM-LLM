import json
import os
import logging
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def initialize_asr(model_directory: str):
    """
    Initialize the ASR model pipeline using the Whisper model.

    Args:
        model_directory: Path to the pre-trained ASR model.
    
    Returns:
        A Hugging Face ASR pipeline.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load the model and processor with low CPU memory usage optimization
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=model_directory,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_directory)

    # Create and return the ASR pipeline
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return asr_pipeline


def batch_asr_processing(asr_pipeline, output_path, dataset_path, dataset_name):
    ds = load_dataset(dataset_name, cache_dir=dataset_path)
    test_data = ds["test"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_data = []

    for sample in tqdm(test_data, desc="Running ASR"):
        audio_input = {
            "array": sample["audio"]["array"],
            "sampling_rate": sample["audio"]["sampling_rate"]
        }
        try:
            result = asr_pipeline(audio_input)
            text = result["text"].strip()
        except Exception as e:
            logging.error(f"Failed to transcribe sample: {e}")
            text = ""

        output_record = {
            "instruction": text,
            "input": "",
            "output": ""
        }
        output_data.append(output_record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def main():
    logging.basicConfig(level=logging.INFO)

    model_cache_dir = "/home/wenxi/mydisk/models/whisper/models--openai--whisper-large-v3"

    dataset_name = "TwinkStart/speech-web-questions"         # llama-questions, speech-triavia-qa speech-web-questions
    dataset_cache_dir = "/home/wenxi/mydisk/data/standard_qa_eval/web_qa" # llama_qa trivia_qa web_qa
    output_path = "/home/wenxi/mydisk/data/standard_qa_eval/json/web_qa_asr_results.json" # llama_qa_asr_results.json trivia_qa_asr_results.json web_qa_asr_results.json

    asr_pipeline = initialize_asr(model_cache_dir)
    batch_asr_processing(asr_pipeline, output_path, dataset_cache_dir, dataset_name)


if __name__ == "__main__":
    main()


#  python /home/wenxi/SLAM-LLM/examples/s2s/evaluation/utils/asr_question.py