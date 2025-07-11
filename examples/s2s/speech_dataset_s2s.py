import json
import copy

import numpy as np

import torch
import whisper
from utils.snac_utils import layershift, get_snac_answer_token, simple_shift
from utils.codec_utils import get_single_layer_answer_token, get_group_answer_token
from utils.dataset_utils import get_first_existing_value
import librosa
import random
import logging
import os

logger = logging.getLogger(__name__)
class SpeechDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        
        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        self.multitask_prompt_path = dataset_config.get("multitask_prompt_path", None)
        self.multitask_prompt_list = {}
        if self.multitask_prompt_path is not None:
            with open(dataset_config.multitask_prompt_path) as f_prompt:
                for line in f_prompt:
                    item = json.loads(line.strip())
                    if item["task"] in self.multitask_prompt_list:
                        self.multitask_prompt_list[item["task"]].append(item["prompt"])
                    else:
                        self.multitask_prompt_list[item["task"]] = [item["prompt"]]
            if split == "train" and int(os.environ.get("RANK", "0")) == 0:
                logger.info("[Multitask Prompt]")
                for task, prompts in self.multitask_prompt_list.items():
                    logger.info(f"  {task}:")
                    for p in prompts:
                        logger.info(f"    - {p}")
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        self.prompt_template = "<SYSTEM>: {}\n "
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        self.manifest_format = dataset_config.get("manifest_format", "parquet")
        self.seed = dataset_config.get("seed", 42)
        self.split_size = dataset_config.get("split_size", 0.1)
        self.modeling_paradigm = dataset_config.get("modeling_paradigm", "parallel")
        self.interleaved_text_token_num = dataset_config.get("interleaved_text_token_num", 12)
        self.interleaved_audio_token_num = dataset_config.get("interleaved_audio_token_num", 36)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 
        assert self.manifest_format in ["parquet", "jsonl"], "manifest_format must be one of [parquet, jsonl]"
        assert self.modeling_paradigm in ["parallel", "interleaved"], "modeling_paradigm must be one of [parallel, interleaved]"

        # vocab config
        self.vocab_config = dataset_config.get("vocab_config", None)
        self.text_vocabsize = self.vocab_config.text_vocabsize
        self.text_specialtokens = self.vocab_config.text_specialtokens
        self.audio_vocabsize = self.vocab_config.audio_vocabsize
        self.audio_specialtokens = self.vocab_config.audio_specialtokens
        self.padded_text_vocabsize = self.vocab_config.padded_text_vocabsize
        self.padded_audio_vocabsize = self.vocab_config.padded_audio_vocabsize
        self.total_vocabsize = self.vocab_config.total_vocabsize
        self._eot = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.vocab_config.eot
        self._pad_t = self.vocab_config.pad_t
        self._input_t = self.vocab_config.input_t
        self._answer_t = self.vocab_config.answer_t
        self._asr = self.vocab_config.asr
        self._eoa = self.vocab_config.eoa
        self._pad_a = self.vocab_config.pad_a
        self._input_a = self.vocab_config.input_a
        self._answer_a = self.vocab_config.answer_a
        self._split = self.vocab_config.split
        self.code_layer = self.vocab_config.code_layer

        self.special_token_a = self._answer_a
        self.special_token_t = self._answer_t

        # task config 
        self.task_type = dataset_config.get("task_type", "s2s")

        # upsample config 
        self.upsample_text_tokens = dataset_config.get("upsample_text_tokens", False)
        self.upsampling_factor = dataset_config.get("upsampling_factor", 1)
        self.upsample_method = dataset_config.get("upsample_method", "repeat")

        # code type config
        self.code_type = dataset_config.get("code_type", "SNAC")
        if self.code_type != "SNAC" and self.code_type != "CosyVoice":
            raise ValueError("code_type must be one of [SNAC, CosyVoice]")
        
        # number of tokens for latency
        self.num_latency_tokens = dataset_config.get("num_latency_tokens", 1)

        # layershift config
        self.do_layershift = dataset_config.get("do_layershift", True)
        if self.do_layershift:
            self.layershift = layershift
        else:
            self.layershift = simple_shift
        

        self.data_list = []

        # TODO: design a better way to load data
        if self.manifest_format == "parquet":
            if dataset_config.train_data_path is not None and dataset_config.val_data_path is not None and dataset_config.train_data_path != dataset_config.val_data_path:
                if split == "train": 
                    ds_train = self._load_dataset(dataset_config.train_data_path, dataset_config.load_from_cache_file, dataset_config.cache_dir_train)
                    self.data_list = ds_train['train'] if 'train' in ds_train else ds_train
                elif split == "val":
                    ds_val = self._load_dataset(dataset_config.val_data_path, dataset_config.load_from_cache_file, dataset_config.cache_dir_val)
                    self.data_list = ds_val['train'] if 'train' in ds_val else ds_val

            elif split == "train" or split == "val":
                ds = self._load_dataset(dataset_config.train_data_path, dataset_config.load_from_cache_file, dataset_config.cache_dir)
                ds_train = ds['train'] if 'train' in ds else ds
                train_val_split = ds_train.train_test_split(test_size=self.split_size, seed=self.seed, shuffle=False)
                
                if split == "train":
                    self.data_list = train_val_split['train']
                else:
                    self.data_list = train_val_split['test']

            elif split == "test":
                ds_test = self._load_dataset(dataset_config.train_data_path, dataset_config.load_from_cache_file, dataset_config.cache_dir)
                self.data_list = ds_test['test']

        elif self.manifest_format == "jsonl":
            if split == "train":
                with open(dataset_config.train_data_path, encoding='utf-8') as fin:
                    for line in fin:
                        data_dict = json.loads(line.strip())
                        self.data_list.append(data_dict)
            else:
                with open(dataset_config.val_data_path, encoding='utf-8') as fin:
                    for line in fin:
                        data_dict = json.loads(line.strip())
                        self.data_list.append(data_dict)
        else:
            raise ValueError("manifest_format must be one of [parquet, jsonl]")

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)

    def _load_dataset(self, path, use_cache, cache_dir):
        from datasets import load_dataset, load_from_disk
        if path is None:
            return None
        return load_dataset(path, cache_dir=cache_dir) if use_cache else load_from_disk(path)

    def extract_audio_feature(self, audio_path):
        # audio path is a dictionary, resample the audio to 16kHz
        if self.manifest_format == "parquet" and isinstance(audio_path, dict):
            audio_raw = audio_path['array']
            audio_raw_sr = audio_path['sampling_rate']
            if not isinstance(audio_raw, np.ndarray):
                audio_raw = np.array(audio_raw)
            audio_raw = librosa.resample(audio_raw, orig_sr=audio_raw_sr, target_sr=16000).astype(np.float32)
        elif (self.manifest_format == "parquet" and (isinstance(audio_path, str) or isinstance(audio_path, list))) or (self.manifest_format == "jsonl" and isinstance(audio_path, list)):
            if self.code_type == "SNAC":
                audio_res, audio_length = get_snac_answer_token(audio_path)
            elif self.code_type == "CosyVoice":
                audio_tokens = audio_path
                if self.code_layer <= 1:
                    audio_res, audio_length = get_single_layer_answer_token(audio_tokens, self.num_latency_tokens, self._pad_a, self._eoa)
                else:
                    audio_res, audio_length = get_group_answer_token(audio_tokens, self.num_latency_tokens, self._pad_a, self._eoa, self.code_layer)
            return audio_res, audio_length
        else:
            audio_raw = whisper.load_audio(audio_path)
            
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320
            audio_length = audio_length // 5
            audio_res = audio_raw
        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2
            audio_length = audio_length // 5
            audio_res = audio_mel

        return audio_res, audio_length

    def get_input_ids(self, length, special_token_a, special_token_t, task_type=None):
        input_ids = []
        if self.code_layer == 0:
            input_ids_item = []
            if task_type is not None and task_type in ["tts", "t2s", "t2t"]:
                input_ids_item.append(self._input_t)
                input_ids_item += [self._pad_t] * length
                input_ids_item += [self._eot, special_token_t]
                input_ids = torch.tensor(input_ids_item).unsqueeze(0).unsqueeze(0)
            else:
                input_ids_item.append(self.layershift(self._input_a, 0))
                input_ids_item += [self.layershift(self._pad_a, 0)] * length
                input_ids_item += [(self.layershift(self._eoa, 0)), self.layershift(special_token_a, 0)]
                input_ids = torch.tensor(input_ids_item).unsqueeze(0).unsqueeze(0)
            return input_ids

        for i in range(self.code_layer):
            input_ids_item = []
            input_ids_item.append(self.layershift(self._input_a, i))
            input_ids_item += [self.layershift(self._pad_a, i)] * length
            input_ids_item += [(self.layershift(self._eoa, i)), self.layershift(special_token_a, i)]
            input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
        input_id_T = torch.tensor([self._input_t] + [self._pad_t] * length + [self._eot, special_token_t])
        input_ids.append(input_id_T.unsqueeze(0))
        return input_ids

    def get_padded_input(self, text_input_idx, text_index_length):
        padded_input = []
        for i in range(self.code_layer):
            padded_input_item = [self.layershift(self._pad_a, i)] * text_index_length
            padded_input.append(torch.tensor(padded_input_item).unsqueeze(0))
        
        final_layer_input = torch.tensor(text_input_idx)
        padded_input.append(final_layer_input.unsqueeze(0))
        return padded_input

    def get_answer_ids(self, length):
        answer_ids = []
        for i in range(self.code_layer):
            answer_ids_item = []
            answer_ids_item += [self.layershift(self._pad_a, i)] * length
            answer_ids.append(torch.tensor(answer_ids_item).unsqueeze(0))
        answer_id_T = torch.tensor([self._pad_t] * length)
        answer_ids.append(answer_id_T.unsqueeze(0))
        return answer_ids

    def pad_interleaved_chunks(self, answer_text_ids, target_audio):
        audio_chunk_num = (len(target_audio) + self.interleaved_audio_token_num - 1) // self.interleaved_audio_token_num
        text_chunk_num = (len(answer_text_ids) + self.interleaved_text_token_num - 1) // self.interleaved_text_token_num

        padding_needed_text = self.interleaved_text_token_num * text_chunk_num - len(answer_text_ids)
        if padding_needed_text > 0:
            pad_tensor_text = torch.full((padding_needed_text,), self._pad_t, dtype=answer_text_ids.dtype)
            answer_text_ids = torch.cat([answer_text_ids, pad_tensor_text])

        padding_needed_audio = self.interleaved_audio_token_num * audio_chunk_num - len(target_audio)
        if padding_needed_audio > 0:
            pad_tensor_audio = torch.full((padding_needed_audio,), self._pad_a, dtype=target_audio.dtype)
            target_audio = torch.cat([target_audio, pad_tensor_audio])

        if audio_chunk_num >= text_chunk_num:
            chunk_diff = audio_chunk_num - text_chunk_num
            pad_tensor_text = torch.full((self.interleaved_text_token_num * chunk_diff,), self._pad_t, dtype=answer_text_ids.dtype)
            answer_text_ids = torch.cat([answer_text_ids, pad_tensor_text])
        else:
            chunk_diff = text_chunk_num - audio_chunk_num
            pad_tensor_audio = torch.full((self.interleaved_audio_token_num * chunk_diff,), self._pad_a, dtype=target_audio.dtype)
            target_audio = torch.cat([target_audio, pad_tensor_audio])

        return answer_text_ids, target_audio

    def interleave_chunks(self, answer_text_ids, target_audio):
        interleaved_tokens = []
        text_chunk_size = self.interleaved_text_token_num
        audio_chunk_size = self.interleaved_audio_token_num

        num_chunks = max(
            len(answer_text_ids) // text_chunk_size, 
            len(target_audio) // audio_chunk_size
        )

        for i in range(num_chunks):
            text_chunk = answer_text_ids[i * text_chunk_size:(i + 1) * text_chunk_size]
            audio_chunk = target_audio[i * audio_chunk_size:(i + 1) * audio_chunk_size]

            interleaved_tokens.extend(text_chunk)
            interleaved_tokens.extend(audio_chunk)

        return interleaved_tokens

    def upsample_tokens(self, tokens, upsampling_factor, method="repeat"):
        """
        Upsample the input tokens based on the given method and factor.
        
        Args:
            tokens: List[int], a list of token IDs to be upsampled.
            upsampling_factor: int, the factor by which the tokens will be upsampled.
            method: str, the upsampling method, either "repeat" or "blank".

        Returns:
            List[int], the upsampled token IDs.
        """
        if upsampling_factor <= 1:
            return tokens

        upsampled_tokens = []
        blank_token = self.tokenizer.pad_token_id  # Use pad token as the blank token

        if method == "repeat":
            # Repeat each token 'upsampling_factor' times
            for token in tokens:
                upsampled_tokens.extend([token] * upsampling_factor)

        elif method == "blank":
            # Add (upsampling_factor - 1) blank tokens after each token
            for token in tokens:
                upsampled_tokens.append(token)
                upsampled_tokens.extend([blank_token] * (upsampling_factor - 1))

        else:
            raise ValueError("Unsupported upsampling method. Choose 'repeat' or 'blank'.")

        return upsampled_tokens
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        task_type = self.task_type
        audio_mel = None
        example_ids = None
        key = None
        audio_length = 0
        target_audio_length = 0

        if self.manifest_format == "parquet":
            source_audio = get_first_existing_value(data_dict, ["question_audio", "question_wav", "source_wav", "audio"])
            if self.code_type == "SNAC":
                target_audio = data_dict.get("answer_snac", None)
            elif self.code_type == "CosyVoice":
                target_audio = data_dict.get("answer_cosyvoice_speech_token", None)
            source_text = get_first_existing_value(data_dict, ["question", "question_text", "Questions"])
            target_text = get_first_existing_value(data_dict, ["answer", "answer_text", "Answer", "answers", "Answers"])
            if source_audio is not None and type(source_audio) == dict:
                key = source_audio.get("path", None)
            task_type = data_dict.get("task_type", task_type)
        elif self.manifest_format == "jsonl":
            source_audio = data_dict.get("source_wav", None)
            target_audio = data_dict.get("target_token", None)
            source_text = data_dict.get("source_text", None)
            target_text = data_dict.get("target_text", None)
            key = data_dict.get("key", None)
            task_type = data_dict.get("task_type", task_type)
        else:
            raise ValueError("manifest_format must be one of [parquet, jsonl]")

        if task_type in ["s2s", "asr", "s2t"]:
            audio_mel, audio_length = self.extract_audio_feature(source_audio)
        
        if task_type in ["s2s", "t2s", "tts"] and target_audio is not None:
            target_audio, target_audio_length = self.extract_audio_feature(target_audio)
        elif task_type in ["asr", "s2t", "t2t"]:
            target_audio, target_audio_length = None, 0

        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio

        if "system prompt" in data_dict:
            prompt = data_dict["system prompt"]
        elif self.multitask_prompt_path is not None and task_type in self.multitask_prompt_list:
            prompt = random.choice(self.multitask_prompt_list[task_type])
        else:
            prompt = self.prompt
        prompt = self.prompt_template.format(prompt)

        if source_text and any(tag in source_text for tag in ["<USER>:", "<OBSERVATION>:"]) and task_type in {"s2s", "t2s"}:
            for tag in ["<USER>:", "<OBSERVATION>:"]:
                if tag in source_text:
                    history_chat = source_text.rsplit(tag, 1)[0].strip()
                    break

            if history_chat:  
                prompt += history_chat + "\n "

        prompt_ids = self.tokenizer.encode(prompt)
        prompt_ids = [self._input_t] + prompt_ids + [self._eot]
        prompt_length = len(prompt_ids)
        prompt_ids = self.get_padded_input(prompt_ids, prompt_length)

        if task_type in ["s2s", "asr", "s2t"]:
            example_ids = self.get_input_ids(audio_length, self.special_token_a, self.special_token_t)
            example_ids = [torch.cat((prompt_ids[i], example_ids[i]), dim = 1) for i in range(self.code_layer + 1)] # 1 for text layer
        elif task_type in ["tts", "t2s", "t2t"]:
            text_input = target_text if self.task_type == "tts" else source_text
            if task_type == "t2s" and any(tag in text_input for tag in ["<USER>:", "<OBSERVATION>:"]):
                for tag in ["<USER>:", "<OBSERVATION>:"]:
                    if tag in text_input:
                        text_input = text_input.rsplit(tag, 1)[-1].strip()
                        break
            text_input_ids = self.tokenizer.encode(text_input)
            text_input_length = len(text_input_ids)
            text_input_ids = torch.tensor(text_input_ids, dtype=torch.int64)
            example_ids = self.get_input_ids(text_input_length, self.special_token_a, self.special_token_t, task_type) # <prompt> <bos> <text> <eos> <task>
            text_layer = example_ids[self.code_layer]
            text_layer = torch.cat((text_layer[:,:1], text_input_ids.unsqueeze(0), text_layer[:,-2:]), dim=1)
            example_ids[self.code_layer] = text_layer
            example_ids = [torch.cat((prompt_ids[i], example_ids[i]), dim = 1) for i in range(self.code_layer + 1)]
        else:
            raise ValueError(f"task_type {task_type} is not supported")

        input_length = audio_length
        if task_type in ["tts", "t2s", "t2t"]:
            input_length = text_input_length

        if task_type == "asr":
            if "<USER>:" in source_text:
                source_text = source_text.split("<USER>:")[-1].strip()
            target_text = source_text

        if self.inference_mode:
            example_mask = example_ids[0][0].ge(-1)  # [True,True]
            example_ids = torch.stack(example_ids).squeeze() if self.modeling_paradigm == "parallel" else torch.stack(example_ids).squeeze(0)

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio_mel": audio_mel,
                "input_length": input_length,
                "audio_length": audio_length,
                "target_audio": target_audio,
                "target_audio_length": target_audio_length,
                "key": key,
                "source_text": source_text,
                "target_text": target_text,
                "prompt_length": prompt_length,
                "task_type": task_type,
            }

        answer_text = self.answer_template.format(target_text)
        answer_text_ids = self.tokenizer.encode(answer_text)  # [answer]
        
        if self.upsample_text_tokens:
            answer_text_ids = self.upsample_tokens(answer_text_ids, 
                                                upsampling_factor=self.upsampling_factor, 
                                                method=self.upsample_method)

        answer_text_ids.append(self._eot) # [answer,eos]
        answer_text_ids = torch.tensor(answer_text_ids, dtype=torch.int64)

        if self.modeling_paradigm == "parallel":
            answer_length = max(len(answer_text_ids), target_audio_length)
            answer_ids = self.get_answer_ids(answer_length)                 # NOTE: somtimes answer_text_ids is longer than target_audio_length 
            answer_ids[self.code_layer] = torch.cat((answer_text_ids.unsqueeze(0), answer_ids[self.code_layer][:,len(answer_text_ids):]),dim=1)     # [answer_text,eos]
            text_padding_length = target_audio_length - len(answer_text_ids)

            labels_ids = copy.deepcopy(answer_ids)
            ori_example_ids = copy.deepcopy(example_ids)
            
            if target_audio is not None:    
                for i in range(self.code_layer):
                    labels_ids[i] = torch.cat((target_audio[i].unsqueeze(0), answer_ids[i][:,target_audio_length:]), dim=1)
                    answer_ids[i] = torch.cat((self.layershift(target_audio[i], i).unsqueeze(0), labels_ids[i][:,target_audio_length:]), dim=1)

            for i in range(self.code_layer + 1):
                example_ids[i] = torch.cat((ori_example_ids[i], answer_ids[i]), dim=1)  # [prompt,audio,answer,eos]
                labels_ids[i] = torch.cat((ori_example_ids[i], labels_ids[i]), dim=1)

            example_ids = torch.stack(example_ids).squeeze()
            labels_ids = torch.stack(labels_ids).squeeze()
            labels_ids[:,:input_length + prompt_length + 3] = -1  # [-1,-1,answer,eos]; NOTE: here 3 include <bos> <eos> <ans_t>

            if text_padding_length > 0:
                labels_ids[self.code_layer,-text_padding_length:] = -1   # [-1,-1,answer_text,eos,-1]
            else:
                audio_padding_length = -text_padding_length
                labels_ids[:self.code_layer,-audio_padding_length:] = -1  # [-1,-1,answer_text,eos,-1]

        elif self.modeling_paradigm == "interleaved":
            example_ids = example_ids[0]
            if target_audio is not None:
                target_audio = target_audio.squeeze(0)
                answer_text_ids, target_audio = self.pad_interleaved_chunks(answer_text_ids, target_audio)
                target_audio_labels = self.layershift(target_audio, 0)
                interleaved_sequence = self.interleave_chunks(answer_text_ids, target_audio_labels)
                interleaved_sequence = torch.tensor(interleaved_sequence)
            else: 
                interleaved_sequence = answer_text_ids

            example_ids = torch.cat((example_ids, interleaved_sequence.unsqueeze(0)), dim=1)
            labels_ids = example_ids.clone()
            labels_ids[:,:input_length + prompt_length + 3] = -1  # [-1,-1,answer,eos]; NOTE: here 3 include <bos> <eos> <ans_t>

            # NOTE: here padding token loss is calculated 
        
        example_mask = example_ids[0].ge(-1)  # [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio_mel": audio_mel,
            "input_length": input_length,
            "audio_length": audio_length,
            "target_audio": target_audio,
            "target_audio_length": target_audio_length,
            "key": key,
            "source_text": source_text,
            "target_text": target_text,
            "prompt_length": prompt_length,
            "task_type": task_type,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    # sequence = torch.nn.functional.pad(sequence, (0, padding_length)) FIXME: this is wrong before in SLAM-LLM
                    padding_tensor = torch.full((sequence.size(0), padding_length), padding_idx, dtype=sequence.dtype)
                    if padding_side == "left":
                        sequence = torch.cat((padding_tensor, sequence), dim=1)
                    else:
                        sequence = torch.cat((sequence, padding_tensor), dim=1)
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 
        input_prompt_lengths = [s["input_length"] + s['prompt_length'] + 3 for s in samples] #[321, 321, 321, 321]
        input_answer_lengths = [len(s["input_ids"][0]) - s["input_length"] - s['prompt_length'] - 3 for s in samples]  #[264, 99, 206, 141]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        # NOTE: left padding for prompt and right padding for answer 
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])

        input_length = torch.tensor([s["input_length"] for s in samples])
        audio_length = torch.tensor([s["audio_length"] for s in samples])
        audio_raw = None
        audio_mask = None
        audio_mel = None
        audio_mel_post_mask = None

        task_type = [s['task_type'] for s in samples]
        if all(t == task_type[0] for t in task_type):
            batch_task_type = task_type[0]
        else:
            raise ValueError("All samples must have the same task type for collator.")

        if self.input_type == "raw" and batch_task_type in ["s2s", "asr", "s2t"]:
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel" and batch_task_type in ["s2s", "asr", "s2t"]:
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
    
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index] + 1 + samples[index]['prompt_length'] # +1 for <bos>
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            target_text = [s['target_text'] for s in samples]
            source_text = [s['source_text'] for s in samples]
            target_audio = [s['target_audio'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw,
                "audio_mask": audio_mask,
                "input_length": input_length,
                "audio_length": audio_length,
                "audio_mel": audio_mel,
                "audio_mel_post_mask": audio_mel_post_mask,
                "modality_mask": modality_mask,
                "keys": keys,
                "target_texts": target_text,
                "source_texts": source_text,
                "target_audio": target_audio,
                "task_types": task_type
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw,
            "audio_mask": audio_mask,
            "input_length": input_length,
            "audio_length": audio_length,
            "audio_mel": audio_mel,
            "audio_mel_post_mask": audio_mel_post_mask,
            "modality_mask": modality_mask,
            "task_types": task_type
        }


def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)
    return dataset
