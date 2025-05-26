#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/v-wenxichen/anaconda3/envs/slam/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1


code_dir=examples/s2s

whisper_size=large-v3                  # tiny base small medium large-v3
speech_encoder_path="/valleblob/v-wenxichen/models/whisper/${whisper_size}.pt"   # replace this with your own whisper model path (different whisper size)
llm_path="/valleblob/v-wenxichen/models/qwen/qwen2.5-7b-instruct"
codec_decoder_path="/valleblob/v-wenxichen/models/CosyVoice/CosyVoice-300M-SFT" # replace this with your own CosyVoice model path
llm_name=Qwen2.5-7b-Instruct

encoder_dim=1280                     # 384 512 768 896 1024 1280 
mel_size=128                         # 80 128 (128 for whisper-large only, 80 for others)
llm_dim=3584                         # 896 1536 2048 3584  -> 0.5B 1.5B 3B 7B

task_type=s2t

# vocabulary settings
code_layer=0                        # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_audio_vocabsize=4160          # the vocab size of the codec token
llm_vocabsize=152000                # the vocab size of the LLM model (Qwen2 here)
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))

# code settings
code_type=CosyVoice                 # CosyVoice or SNAC
codec_decoder_type=CosyVoice
num_latency_tokens=0                # number of latency tokens (same as the number in training)
do_layershift=false                 # if false, tokens in each layers use the same codebook, otherwise, use different codebooks


ckpt_path=/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz1-lr1e-4-interleave_text12_audio36-qwen2.5-7b-instruct-s2t-freeze_llm-gradient_accumulation_steps2-total_steps150000-qwen2.5-7b-instruct_prediction_answer/s2s_epoch_3_step_56390

# PEFT settings
use_peft=false
lora_r=32
lora_alpha=$((lora_r * 2))

# jsonl dataset
# manifest_format=jsonl
# val_data_path=/home/v-wenxichen/SLAM-LLM/examples/s2s/demo/data/${split}.jsonl

# huggingface dataset
manifest_format=parquet
val_data_path=TwinkStart/speech-web-questions        # llama-questions speech-triavia-qa speech-web-questions
load_from_cache_file=true
DATASET_NAME=web_qa # llama_qa trivia_qa web_qa
cache_dir=/home/wenxi/mydisk/data/standard_qa_eval/$DATASET_NAME

# decode config
modeling_paradigm=interleaved
interleaved_text_token_num=12
interleaved_audio_token_num=36
text_repetition_penalty=1.2
audio_repetition_penalty=1.2        # default 1.0, set to 1.2 for reduce silence
max_new_tokens=3000                 # 500 for SNAC, 3000 for CosyVoice-single
do_sample=false
top_p=1.0
top_k=0
temperature=1.0
decode_text_only=false
if [[ "$task_type" == "s2t" ]]; then
        codec_decode=false                  # since the model outputs text only, set to false
else
        codec_decode=true
fi

output_text_only=true
speech_sample_rate=22050            # 22050 for CosyVoice, 24000 for SNAC
inference_online=false
# audio_prompt_path=./examples/s2s/audio_prompt/zh/prompt_6.wav      # replace this with your own audio prompt path or our provided audio prompt path
audio_prompt_path=./examples/s2s/audio_prompt/en/prompt_6.wav      # replace this with your own audio prompt path or our provided audio prompt path

decode_log=/home/wenxi/mydisk/exp/standard_qa_eval/${DATASET_NAME}/gpu4-btz1-lr1e-4-interleave_text12_audio36-${llm_name}-lora-audio_embed_only-freeze_llm-s2t-whisper_${whisper_size}-qwen2.5-7b-instruct-prediction_answer

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_s2s.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=$llm_name \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=linear \
        ++model_config.codec_decoder_path=$codec_decoder_path \
        ++model_config.codec_decode=$codec_decode \
        ++model_config.vocab_config.code_layer=$code_layer \
        ++model_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++model_config.code_type=$code_type \
        ++model_config.codec_decoder_type=$codec_decoder_type \
        ++dataset_config.dataset=speech_dataset_s2s \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.train_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=$mel_size \
        ++dataset_config.inference_mode=true \
        ++dataset_config.manifest_format=$manifest_format \
        ++dataset_config.load_from_cache_file=$load_from_cache_file \
        ++dataset_config.task_type=$task_type \
        ++dataset_config.vocab_config.code_layer=$code_layer \
        ++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++dataset_config.code_type=$code_type \
        ++dataset_config.num_latency_tokens=$num_latency_tokens \
        ++dataset_config.do_layershift=$do_layershift \
        ++dataset_config.modeling_paradigm=$modeling_paradigm \
        ++dataset_config.interleaved_text_token_num=$interleaved_text_token_num \
        ++dataset_config.interleaved_audio_token_num=$interleaved_audio_token_num \
        ++dataset_config.cache_dir=$cache_dir \
        ++train_config.model_name=s2s \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.task_type=$task_type \
        ++train_config.modeling_paradigm=$modeling_paradigm \
        ++train_config.interleaved_text_token_num=$interleaved_text_token_num \
        ++train_config.interleaved_audio_token_num=$interleaved_audio_token_num \
        ++train_config.use_peft=$use_peft \
        ++train_config.peft_config.lora_alpha=$lora_alpha \
        ++train_config.peft_config.r=$lora_r \
        ++decode_config.text_repetition_penalty=$text_repetition_penalty \
        ++decode_config.audio_repetition_penalty=$audio_repetition_penalty \
        ++decode_config.max_new_tokens=$max_new_tokens \
        ++decode_config.task_type=$task_type \
        ++decode_config.do_sample=$do_sample \
        ++decode_config.top_p=$top_p \
        ++decode_config.top_k=$top_k \
        ++decode_config.temperature=$temperature \
        ++decode_config.decode_text_only=$decode_text_only \
        ++decode_config.do_layershift=$do_layershift \
        ++decode_log=$decode_log \
        ++decode_config.num_latency_tokens=$num_latency_tokens \
        ++ckpt_path=$ckpt_path/model.pt \
        ++output_text_only=$output_text_only \
        ++inference_online=$inference_online \
        ++speech_sample_rate=$speech_sample_rate \
        ++audio_prompt_path=$audio_prompt_path

# bash ./examples/s2s/scripts/inference/inference_s2s_batch_interleave.sh