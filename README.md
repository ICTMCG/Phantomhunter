# PhantomHunter: AI-Generated Text Detection with Multi-Task MoE Framework

Official implementation of "PhantomHunter: A Multi-Task Framework with Mixture of Experts for Generalized Generated Text Detection" (SIGIR 2026).

## Overview

<p align="center">
  <img src="pic/head.png" alt="PhantomHunter" width="70%">
</p>

PhantomHunter is a unified framework for detecting AI-generated text that leverages Mixture of Experts (MoE) architecture, Contrastive Learning (CL), and Low-Rank Adaptation (LoRA) to achieve state-of-the-art performance across multiple AI model families.



## Architecture

![PhantomHunter Architecture](pic/method.png)

**PhantomHunter** and the training process. Given a text sample $\mathbf{x}$, it **1)** extracts the probability feature from $M$ base models and encode them with CNN and transformer blocks; **2)** predicts the family of $\mathbf{x}$ to determine the family gating weights; and **3)** feeds the representation $\mathbf{R}_{F}$ to a mixture-of-experts network controlled by the gating weights from Step 2 for final prediction of $\mathbf{x}$ being LLM-generated. During training, contrastive learning is applied in each mini-batch to better model family relationships. The red terms are loss functions.

## Key Features

- **Multi-Task Learning**: Simultaneously performs binary detection (human vs. AI) and multi-class classification (identifying specific AI models)
- **Mixture of Experts**: Dynamic routing mechanism adapts to different AI generation patterns
- **Contrastive Learning**: Enhances feature discrimination across model families
- **LoRA Adaptation**: Efficient fine-tuning with model-specific expert adapters

## Supported AI Models

- **Human** (baseline)
- **Llama** (Meta)
- **Mistral** (Mistral AI)
- **Gemma** (Google)
- **Qwen2.5** (Alibaba)

## Quick Start

### Installation

```bash
pip install torch transformers scikit-learn
```

### Training

```bash
python main.py \
    --cuda \
    --exp-name phantom_hunter \
    --train-path feature/arxiv_new/lora/train.jsonl \
    --val-path feature/arxiv_new/lora/val.jsonl \
    --test-path feature/arxiv_new/lora/test_ood.jsonl \
    --batch-size 64 \
    --lr 5e-4 \
    --is-cl \
    --train
```

### Evaluation

```bash
python main.py \
    --cuda \
    --test \
    --exp-name phantom_hunter \
    --test-path feature/arxiv_new/lora/test_ood.jsonl
```



## License

MIT License
