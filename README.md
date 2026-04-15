# Vietnamese GPT-2: Stage-1 Pretraining + Health Data Bootstrap

A clean and reproducible GPT-2 pretraining pipeline for Vietnamese, currently focused on:

- **Stage 1**: Pretrain GPT-2 from random initialization on mixed Vietnamese corpora.
- **Health bootstrap**: Crawl and structure health-domain data for a future stage-2 adaptation.

This repository includes data preparation, tokenizer training, stage-1 pretraining, text generation, and a first health-domain crawling step.

---

## Repository Structure

```text
vietnamese-gpt2/
├── src/
│   ├── config.py               # Central configuration: paths, datasets, hyperparameters
│   ├── utils.py                # Shared helpers for normalization, callbacks, generation
│   ├── train_tokenizer.py      # Train tokenizer from Vietnamese corpora
│   ├── train_1.py              # Stage 1 pretraining from scratch
│   ├── generate_base.py        # Generate text with the stage-1 model
│   └── __init__.py
├── data_prep/
│   ├── news/download_datasets.py
│   ├── wiki/crawl_vi_wiki.py
│   ├── wiki/process_vi_wiki.py
│   ├── health/crawl_disease_index.py
│   └── deduplicate.py
├── scripts/
│   └── train_1.sh
├── artifacts/                  # Tokenizer, checkpoints, logs, final models
└── data/                       # Raw and processed datasets
```


## Training Overview

### Stage 1: Base Language Pretraining

Train GPT-2 from random initialization on mixed Vietnamese corpora such as news and Wikipedia.

### Health Bootstrap: Disease Index Crawling

Bootstrap a health-domain dataset by collecting disease detail links from Tam Anh's A-Z disease index page.

## Requirements

* Python **3.11+**
* CUDA-compatible GPU
* `flash-attn` (optional; requires a compatible CUDA toolchain)
* [uv](https://github.com/astral-sh/uv) for environment and package management

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/duongtruongbinh/vietnamese-gpt2
cd vietnamese-gpt2
uv sync
uv pip install -e .
```

Run all commands from the repository root.


## Pipeline

### 1. Prepare raw corpora

```bash
uv run python data_prep/news/download_datasets.py
uv run python data_prep/wiki/crawl_vi_wiki.py
uv run python data_prep/wiki/process_vi_wiki.py
```

### 2. Train the tokenizer

```bash
uv run python src/train_tokenizer.py
```

### 3. Deduplicate the pretraining data

```bash
uv run python data_prep/deduplicate.py
```

### 4. Run stage 1 pretraining

```bash
bash scripts/train_1.sh
```

### 5. Bootstrap health-domain disease links

```bash
uv run python data_prep/health/crawl_disease_index.py --save-html
```

## Text Generation

Generate text with the base model:

```bash
uv run python src/generate_base.py
```

## Configuration

All important paths and hyperparameters are managed in:

```text
src/config.py
```

This includes:

* Dataset paths
* Tokenizer directory
* Checkpoint directory
* Sequence length
* Batch size
* Learning rate
* Training budget
* Logging and runtime settings


## Outputs

Training artifacts are stored under:

```text
artifacts/
```

Typical outputs include:

* Trained tokenizer
* Intermediate checkpoints
* Final stage-1 model
* Training logs

---

## Notes

* Stage 1 is intended for **general Vietnamese language modeling**.
* The current health crawler only handles the first step: fetching and saving disease detail links.
* For best results, ensure corpus quality and deduplication are completed before training.
* A GPU is strongly recommended for both tokenizer experimentation and model training.

---

## Project Goal

This project aims to provide a simple, practical, and extensible foundation for training Vietnamese GPT-2 models from scratch and extending them toward domain-specific corpora such as health content.
# gpt2-disease
