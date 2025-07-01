#!/bin/bash

DATASET=web_qa # llama_qa trivia_qa web_qa
DECODE_DIR=/home/wenxi/mydisk/exp/standard_qa_eval/${DATASET}/Qwen2.5-7b-Instruct-lora_r32_alpha64-audio_embed_only-s2t-whisper_large-v3-qwen2.5-7b-instruct_enrich_new
# DECODE_DIR=/home/wenxi/mydisk/exp/standard_qa_eval/web_qa/qwen2.5-7b-instruct-VA_qwen2.5-7b-instruct_refined_new_sft-llamafactory_cli/generated_predictions_with_labels.jsonl
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