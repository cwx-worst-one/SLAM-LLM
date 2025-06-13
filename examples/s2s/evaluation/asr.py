import os
import torch
import logging
from tqdm import tqdm
from argparse import ArgumentParser
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


def transcribe_audio_files(asr_pipe, input_directory: str, output_file: str):
    """
    Process all audio files in the input_directory using the ASR pipeline and write transcriptions to the output file.

    Args:
        asr_pipe: The initialized ASR pipeline.
        input_directory: Path to the directory containing audio files.
        output_file: Path to the output file to store transcriptions.
    """
    audio_files = os.listdir(input_directory)
    num_audio = len(audio_files)
    logging.info(f"Number of audio files to process: {num_audio}")

    with open(output_file, mode="w", encoding="utf-8") as f:
        for file_name in tqdm(audio_files, desc="Transcribing audio files"):
            audio_path = os.path.join(input_directory, file_name)
            if os.path.exists(audio_path):
                result = asr_pipe([audio_path], batch_size=1)
                logging.info(f"{file_name} transcribed: {result}")
                transcription = result[0].get("text", "").strip()
                f.write(f"{file_name}\t{transcription}\n")
            else:
                # Write an empty transcription if the audio file does not exist
                f.write(f"{file_name}\t\n")


def setup_logging():
    """
    Configures logging with a standard format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = ArgumentParser(description="ASR Transcription Script using Whisper")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing audio (.wav) files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the transcription file")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory of the pre-trained Whisper model (e.g., whisper-large-v3)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate the input directory
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory '{args.input_dir}' does not exist.")

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file = os.path.join(args.output_dir, "pred_audio_asr_text")

    setup_logging()
    logging.info("ASR transcription process started.")

    asr_pipe = initialize_asr(args.model_dir)
    transcribe_audio_files(asr_pipe, args.input_dir, output_file)

    logging.info("ASR transcription process completed.")


if __name__ == "__main__":
    main()
