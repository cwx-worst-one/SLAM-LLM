#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAME="qwen2.5-7b-instruct"
MODEL_PATH="/home/wenxi/mydisk/models/qwen/$MODEL_NAME"
DATASET="llama_qa_asr"  # llama_qa_asr trivia_qa_asr web_qa_asr
SAVE_PATH=/home/wenxi/mydisk/exp/standard_qa_eval/${DATASET}/$MODEL_NAME
mkdir -p "$SAVE_PATH"

python scripts/vllm_infer.py \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --save_name $SAVE_PATH/${DATASET}_predictions.jsonl \

# bash examples/wenxi/infer.sh