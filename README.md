<div align="center">
    <h1>
    SLAM-LLM
    </h1>
    <p>
    <b>SLAM-LLM</b> is a deep learning toolkit that allows researchers and
developers to train custom multimodal large language model (MLLM), focusing on <b>S</b>peech, <b>L</b>anguage, <b>A</b>udio, <b>M</b>usic processing. We provide detailed recipes for training and high-performance checkpoints for inference. <br>
    </p>
    <p>
    <img src="docs/logo.jpg" alt="SLAM-LLM Logo" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/Cuda-11.8+-orange" alt="version"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/PyTorch-2.01+-brightgreen" alt="python"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit"></a>
</div>

# Table of Contents
1. [News](#news)
2. [Installation](#installation)
3. [Usage](#usage)
    - [List of Recipes](#list-of-recipes)
    - [Configuration Priority](#configuration-priority)
4. [Features](#features)
5. [Acknowledge](#acknowledge)
6. [Citation](#citation)

# News
- [Update Apr. 24, 2025] We have supported [large-scale industrial training](examples/aispeech_asr/README.md), suitable for datasets on the order of 100,000 hours. Its main features include:
  - **Support for multi-task training:** Designed to support tasks such as ASR and ST through a unified data format. 
  - **Dynamic prompt selection:** Supports random selection from multiple prompts. 
  - **Iterative dataset:** Uses an iterative dataset format to reduce startup time for large datasets. 
  - **Deepspeed training:** Supports DeepSpeed training to significantly reduce memory usage.
  - **Multi-machine multi-GPU inference:** Supports distributed inference across multiple machines and GPUs to reduce evaluation time.
  - **Dynamic frame batching:** Dynamically combines frames based on audio size rather than using a fixed batch size, significantly reducing training and evaluation time (reduces training time by 3/4 for 100,000 hours of data).
- [Update Apr. 24, 2025] We have supported the Deepspeed, checkout the instruction #Fine-tuning using Deepspeed at [here](examples/asr_librispeech/README.md).
- [Update Jan. 22, 2025] 🔥🔥🔥 Full reproduction (including all data preparation, model training, and inference) for [SLAM-Omni](examples/s2s/README.md) has been supported.  
![](docs/slam-omni-model.png)
  - SLAM-Omni is a **timbre-controllable** voice interaction system that requires only **single-stage training** and minimal resources to achieve high-quality, end-to-end speech dialogue, supporting multi-turn conversations in both Chinese and English. ([paper](https://arxiv.org/abs/2412.15649), [demo](https://slam-omni.github.io))
  - We have fully reproduced the **training and inference** processes of SLAM-Omni and open-sourced all related training datasets. The provided code framework theoretically supports all codec-based spoken dialogue models. Additionally, we offer the reproduction code for [Mini-Omni](https://github.com/gpt-omni/mini-omni).

<table class="center">
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/73597edb-0d66-453b-b10c-8cf8dd3cae18" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/7a797491-0509-4da8-8662-f2107bd8856a" muted="false"></video>
    </td>
</tr>
</table>

- [Update Nov. 17, 2024] Recipes for [LLM-Based Contextual ASR](examples/contextual_asr/README.md) have been supported. 
- [Update Nov. 5, 2024] Recipes for [speech emotion captioning (SEC)](examples/sec_emotioncaps/README.md) with [emotion2vec](https://github.com/ddlBoJack/emotion2vec) as the encoder has been supported.
- [Update Oct. 12, 2024] Recipes for [SLAM-AAC](examples/slam_aac/README.md) with [EAT](https://github.com/cwx-worst-one/EAT) as the encoder have been supported. 
- [Update Sep. 28, 2024] Recipes for [CoT-ST](examples/st_covost2/README.md) have been supported. 
- [Update Sep. 25, 2024] Recipes for [DRCap](examples/drcap_zeroshot_aac/README.md) have been supported. 
- [Update Jun. 12, 2024] Recipes for [MaLa-ASR](examples/mala_asr_slidespeech/README.md) have been supported. 
- **[CALL FOR EXAMPLE]** We sincerely invite developers and researchers to develop new applications, conduct academic research based on SLAM-LLM, and pull request your examples! We also acknowledge engineering PR (such as improving and speeding up multi-node training). 
- [Update May. 22, 2024] Please join [slack](https://join.slack.com/t/slam-llm/shared_invite/zt-2mc0pkhhs-5jjOi8Cwc8R1Xc8IQmykDA) or [WeChat group](./docs/Wechat.jpg). We will sync our updates and Q&A here. 
- [Update May. 21, 2024] Recipes for [Spatial Audio Understanding](examples/seld_spatialsoundqa/README.md) have been supported. 
- [Update May. 20, 2024] Recipes for [music caption (MC)](examples/mc_musiccaps/README.md) have been supported. 
- [Update May. 8, 2024] Recipes for [visual speech recognition (VSR)](examples/vsr_LRS3/README.md) have been supported. 
- [Update May. 4, 2024] Recipes for [zero-shot text-to-speech (TTS)](examples/vallex/README.md) have been supported. 
- [Update Apr. 28, 2024] Recipes for [automated audio captioning (AAC)](examples/aac_audiocaps/README.md) have been supported. 
- [Update Mar. 31, 2024] Recipes for [automatic speech recognition (ASR)](examples/asr_librispeech/README.md) have been supported. 

# Installation
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout tags/v4.35.2
pip install -e .
cd ..
git clone https://github.com/huggingface/peft.git
cd peft
git checkout tags/v0.6.0
pip install -e .
cd ..
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/cwx-worst-one/SLAM-LLM.git
cd SLAM-LLM
pip install  -e .
```

For some examples, you may need to use `fairseq`, the command line is as follows:
```
# you need to install fairseq before SLAM-LLM
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
We also provide a docker image for convenience:
```shell
# build docker image
docker build -t slam-llm:latest .

# run docker image with gpu
docker run -it --gpus all --name slam --shm-size=256g slam-llm:latest /bin/bash
```
# Usage
## List of Recipes
We provide reference implementations of various LLM-based speech, audio, and music tasks: 
- **Speech Task**
    - Automatic Speech Recognition (ASR)
        - [SLAM-ASR](examples/asr_librispeech/README.md)
    
    - Contextual Automatic Speech Recognition (CASR)
        - [ Mala-ASR](examples/mala_asr_slidespeech/README.md)
        - [LLM-Based Contextual ASR](examples/contextual_asr/README.md)
    
    - [Visual Speech Recognition (VSR)](examples/vsr_LRS3/README.md) 
    - Speech-to-Text Translation (S2TT)
        - [CoT-ST](examples/st_covost2/README.md)
    
    - Text-to-Speech (TTS)
        - [VALL-E-X](examples/vallex/README.md)
    - [Speech Emotion Captioning (SEC)](examples/sec_emotioncaps/README.md)
    - Voice Interaction System
        - [SLAM-Omni](examples/s2s/README.md)
    
- **Audio Task**
    - [Automated Audio Captioning (AAC)](examples/aac_audiocaps/README.md)
      - [SLAM-AAC](examples/slam_aac/README.md)
      - [DRCap](examples/drcap_zeroshot_aac/README.md)
  
    - Spatial Audio Understanding
      - [BAT](examples/seld_spatialsoundqa/README.md)
    
- **Music Task**
    - [Music Caption (MC)](examples/mc_musiccaps/README.md)

## Configuration Priority
We provide hierarchical configuration inheritance relationships as follows:
```
command-line (shell file) > Hydra configuration (yaml file) > dataclass configuration (Python file)
```

# Features
- Easily extend to new models and tasks.
- Detailed recipes for training and high-performance checkpoints for inference.
- Mixed precision training which trains faster with less GPU memory on NVIDIA tensor cores. 
- Multi-GPU training with data and model parallel, supporting [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [deepspeed](https://github.com/microsoft/DeepSpeed) (still need to be improved).  
- Flexible configuration based on [Hydra](https://github.com/facebookresearch/hydra) and [dataclass](https://docs.python.org/3/library/dataclasses.html) allowing a combination of code, command-line and file based configuration. 

# Acknowledge
- We borrow code from [Llama-Recipes](https://github.com/meta-llama/llama-recipes) for the training process. 
- We borrow code from [Fairseq](https://github.com/facebookresearch/fairseq) for deepspeed configuration. 
- We thank the contributors for providing diverse recipes. 

# Citation

## Speech Task

SLAM-ASR:
```
@article{ma2024embarrassingly,
  title={An Embarrassingly Simple Approach for LLM with Strong ASR Capacity},
  author={Ma, Ziyang and Yang, Guanrou and Yang, Yifan and Gao, Zhifu and Wang, Jiaming and Du, Zhihao and Yu, Fan and Chen, Qian and Zheng, Siqi and Zhang, Shiliang and others},
  journal={arXiv preprint arXiv:2402.08846},
  year={2024}
}
```
Mala-ASR:
```
@article{yang2024mala,
  title={MaLa-ASR: Multimedia-Assisted LLM-Based ASR},
  author={Yang, Guanrou and Ma, Ziyang and Yu, Fan and Gao, Zhifu and Zhang, Shiliang and Chen, Xie},
  journal={Proc. INTERSPEECH},
  year={2024}
}
```
LLM-Based Contextual ASR:
```
@article{yang2024ctc,
  title={CTC-Assisted LLM-Based Contextual ASR},
  author={Yang, Guanrou and Ma, Ziyang and Gao, Zhifu and Zhang, Shiliang and Chen, Xie},
  journal={Proc. SLT},
  year={2024}
}
```
CoT-ST:
```
@article{du2024cot,
  title={CoT-ST: Enhancing LLM-based Speech Translation with Multimodal Chain-of-Thought},
  author={Du, Yexing and Ma, Ziyang and Yang, Yifan and Deng, Keqi and Chen, Xie and Yang, Bo and Xiang, Yang and Liu, Ming and Qin, Bing},
  journal={arXiv preprint arXiv:2409.19510},
  year={2024}
}
```

SLAM-Omni:
```
@article{chen2024slam,
  title={SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training},
  author={Chen, Wenxi and Ma, Ziyang and Yan, Ruiqi and Liang, Yuzhe and Li, Xiquan and Xu, Ruiyang and Niu, Zhikang and Zhu, Yanqiao and Yang, Yifan and Liu, Zhanxun and others},
  journal={arXiv preprint arXiv:2412.15649},
  year={2024}
}
```

## Audio Task
SLAM-AAC:
```
@article{chen2025slam,
  title={SLAM-AAC: Enhancing Audio Captioning with Paraphrasing Augmentation and CLAP-Refine through LLMs},
  author={Chen, Wenxi and Ma, Ziyang and Li, Xiquan and Xu, Xuenan and Liang, Yuzhe and Zheng, Zhisheng and Yu, Kai and Chen, Xie},
  journal={Proc. ICASSP},
  year={2025}
}
```
DRCap:
```
@article{li2025drcap,
  title={DRCap: Decoding CLAP Latents with Retrieval-augmented Generation for Zero-shot Audio Captioning},
  author={Li, Xiquan and Chen, Wenxi and Ma, Ziyang and Xu, Xuenan and Liang, Yuzhe and Zheng, Zhisheng and Kong, Qiuqiang and Chen, Xie},
  journal={Proc. ICASSP},
  year={2025}
}
```
BAT:
```
@article{zheng2024bat,
  title={BAT: Learning to Reason about Spatial Sounds with Large Language Models},
  author={Zheng, Zhisheng and Peng, Puyuan and Ma, Ziyang and Chen, Xie and Choi, Eunsol and Harwath, David},
  journal={Proc. ICML},
  year={2024}
}
```
