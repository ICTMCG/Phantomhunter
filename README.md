# PhantomHunter: AI-Generated Text Detection with Multi-Task MoE Framework

Official implementation of "PhantomHunter: A Multi-Task Framework with Mixture of Experts for Generalized Generated Text Detection".

## Overview

<p align="center">
  <img src="pic/head.png" alt="PhantomHunter" width="70%">
</p>

PhantomHunter is a unified framework for detecting AI-generated text that leverages Mixture of Experts (MoE) architecture, Contrastive Learning (CL), and Low-Rank Adaptation (LoRA) to achieve state-of-the-art performance across multiple AI model families.



## Architecture

![PhantomHunter Architecture](pic/method.png)

**PhantomHunter** and the training process. Given a text sample $\mathbf{x}$, it **1)** extracts the probability feature from $M$ base models and encode them with CNN and transformer blocks; **2)** predicts the family of $\mathbf{x}$ to determine the family gating weights; and **3)** feeds the representation $\mathbf{R}_{F}$ to a mixture-of-experts network controlled by the gating weights from Step 2 for final prediction of $\mathbf{x}$ being LLM-generated. During training, contrastive learning is applied in each mini-batch to better model family relationships. The red terms are loss functions.

## Data
We simulate two common LLM usage scenarios: **writing** (69,297 arXiv paper abstracts) and **question-answering** (3,062 Q&A pairs from ELI5, finance, and medicine domains). We select four open-source models (LLaMA-2-7B-Chat, Gemma-7B-it, Mistral-7B-Instruct-v0.1, Qwen2.5-7B-Instruct) and fine-tune each with full-parameter and LoRA methods on domain-specific corpora, resulting in 48 derivative models for evaluation. 

```
Some test data can be available at ./data/
```

## Quick Start

### Installation
```
pip install -r requirements.txt

```

### Genfeature through four white-box model

1. loading models
```python
# cd ./genfeatures/
# you can modify you own model path in ./genfeatures/backend_api.py
python backend_api.py --port 6009 --timeout 30000 --debug --model=llama --gpu=0
python backend_api.py --port 6010 --timeout 30000 --debug --model=gemma --gpu=1
python backend_api.py --port 6011 --timeout 30000 --debug --model=mistral --gpu=2
python backend_api.py --port 6012 --timeout 30000 --debug --model=qwen2.5 --gpu=4
```
2. genfeatures
```python
# you should modify the en_input_files and en_outfiles path in ./genfeatures/gen_features.py
python ./genfeatures/gen_features.py --get_en_features_multithreading
```

### Train 
```bash
python main.py \
    --cuda  \
    --seed 2024 \
    --exp-name moe+logits+cl_arxiv-lora_5e-4 \
    --train-path /feature/arxiv_new/lora/train.jsonl \
    --val-path /feature//arxiv_new/lora/val.jsonl \
    --test-path /feature/arxiv_new/lora/test_ood.jsonl \
    --batch-size 64 \
    --lr 5e-4 \
    --train
```

### Evaluation

```bash
python main.py \
    --cuda  \
    --seed 2024 \
    --exp-name moe+logits+cl_arxiv-lora_5e-4 \
    --train-path /feature/arxiv_new/lora/train.jsonl \
    --val-path /feature//arxiv_new/lora/val.jsonl \
    --test-path /feature/arxiv_new/lora/test_ood.jsonl \
    --batch-size 64 \
    --lr 5e-4 \
    --test
```



## License

MIT License
