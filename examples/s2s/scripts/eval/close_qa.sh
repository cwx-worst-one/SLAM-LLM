#!/bin/bash

DATASET=web_qa # llama_qa trivia_qa web_qa
DECODE_DIR=/home/wenxi/mydisk/exp/standard_qa_eval/${DATASET}/gpu4-btz1-lr1e-4-Qwen2.5-7b-Instruct-lora_r32-audio_embed_only-s2t-whisper_large-v3-s2t_1.00-t2t_0.50
# DECODE_DIR=/home/wenxi/mydisk/exp/standard_qa_eval/web_qa_asr/qwen2.5-7b-instruct-VA_sft-new_llamafactory_cli/generated_predictions_with_labels.jsonl
FORMAT=tsv    # tsv jsonl


if [[ "$FORMAT" == "tsv" ]]; then
    PRED_FILE="$DECODE_DIR/pred_text"
    # PRED_FILE="$DECODE_DIR/pred_audio_asr_text"
    GT_FILE="$DECODE_DIR/gt_text"
else
    PRED_FILE="$DECODE_DIR"
    GT_FILE=""
fi

# -m debugpy --listen 5678 --wait-for-client
python ./examples/s2s/evaluation/close_qa.py \
    --pred "$PRED_FILE" \
    --gt "$GT_FILE" \
    --exist \
    --format "$FORMAT" \
    --show-mixmatch

# bash ./examples/s2s/scripts/eval/close_qa.sh