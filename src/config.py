#!/usr/bin/env python3
"""Centralized configuration for the Vietnamese GPT-2 pretraining project."""

# ── Dataset ──────────────────────────────────────────────────────────────────
RAW_DATASETS = [
    "data/train/bkai_train.parquet",
    "data/train/vi_wiki_articles_clean.parquet",
]

DEDUP_DIR = "data/train/deduped"

# Deduped sources for training. weight = how many times the source is repeated
# in the training mixture (the deduped parquets on disk stay unique).
DATASETS = [
    {"path": "data/train/deduped/bkai_train.parquet", "weight": 1},
    {"path": "data/train/deduped/vi_wiki_articles_clean.parquet", "weight": 3},
]

# ── Tokenizer training ──────────────────────────────────────────────────────
VOCAB_SIZE = 50257
MIN_FREQUENCY = 2
SPECIAL_TOKEN = "<|endoftext|>"

# ── Model ────────────────────────────────────────────────────────────────────
BASE_MODEL = "gpt2"
MAX_LENGTH = 1024

# ── Training hyperparameters ─────────────────────────────────────────────────
SEED = 42
TOKEN_BUDGET = 2_480_000_000
EVAL_SPLIT_RATIO = 0.01
PREPROCESSING_NUM_WORKERS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 64
WARMUP_RATIO = 0.1
BF16 = True
GRADIENT_CHECKPOINTING = True
DATALOADER_NUM_WORKERS = 10
WANDB_RUN_NAME_STAGE_1 = "rand-init"

# ── Paths ────────────────────────────────────────────────────────────────────
TOKENIZER_DIR = "./artifacts/tokenizer"
CHECKPOINT_DIR = f"./artifacts/checkpoints/{WANDB_RUN_NAME_STAGE_1}"
MODEL_DIR = f"./artifacts/checkpoints/{WANDB_RUN_NAME_STAGE_1}/final"

# ── Inference defaults ───────────────────────────────────────────────────────
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.2


# ── Stage 2 continued pretraining (health corpus) ───────────────────────────
HEALTH_RAW_DIR = "data/raws"

WANDB_RUN_NAME_HEALTH = "continued-pretrain-health"
HEALTH_AZ_URL = "https://tamanhhospital.vn/benh-hoc-a-z/"
HEALTH_DATA_PATH = "data/train/health_disease_clean.jsonl"
HEALTH_RAW_JSONL = f"{HEALTH_RAW_DIR}/health_disease_content.jsonl"
HEALTH_RAW_HTML_DIR = f"{HEALTH_RAW_DIR}/health_html"
HEALTH_METADATA_CSV = f"{HEALTH_RAW_DIR}/health_disease_links.csv"
HEALTH_CHECKPOINT_DIR = f"./artifacts/checkpoints/{WANDB_RUN_NAME_HEALTH}"
HEALTH_MODEL_DIR = f"./artifacts/checkpoints/{WANDB_RUN_NAME_HEALTH}/final"
HEALTH_INDEX_HTML = "data/raws/health_disease_index.html"
HEALTH_PREFIX = ""
HEALTH_EPOCHS = 20
HEALTH_BATCH_SIZE = 32
HEALTH_LEARNING_RATE = 5e-5
HEALTH_WEIGHT_DECAY = 0.1
HEALTH_MAX_LENGTH = 512