#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# DECODE_DIR=/home/wenxi/mydisk/exp/standard_qa_eval/web_qa/gpu4-btz1-lr1e-4-interleave_text12_audio36-Qwen2.5-3b-gradient_accumulation2-lora-audio_embed_only-lora_rank512-alpha1024
# INPUT_DIR="$DECODE_DIR/pred_audio/prompt_6"
DECODE_DIR="/home/wenxi/mydisk/exp/standard_qa_eval/web_qa/LLaMA-Omni2-3B/"   # llama_qa trivia_qa web_qa
INPUT_DIR="$DECODE_DIR/wav"
OUTPUT_DIR="$DECODE_DIR"
MODEL_DIR="/home/wenxi/mydisk/models/whisper/whisper-large-v3"

# Run the ASR transcription Python script
python examples/s2s/evaluation/asr.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_dir "$MODEL_DIR"

# bash ./examples/s2s/scripts/eval/asr.sh