#!/usr/bin/env python3
from loguru import logger
import os

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast

from src.config import (
    RAW_DATASETS, VOCAB_SIZE, MIN_FREQUENCY,
    SPECIAL_TOKEN, TOKENIZER_DIR, MAX_LENGTH,
)
from src.utils import normalize_text

def get_training_corpus(datasets_list, batch_size=10000):
    for dataset in datasets_list:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            yield [normalize_text(text) for text in batch["text"]]

def train_tokenizer():

    all_datasets = []
    for path in RAW_DATASETS:
        logger.info("Loading: {}", path)
        ds = load_dataset("parquet", data_files=path, split="train")
        ds = ds.select_columns(["text"])
        logger.info("  {} samples", f"{len(ds):,}")
        all_datasets.append(ds)

    tokenizer = ByteLevelBPETokenizer()
    logger.info(
        "Training tokenizer: vocab_size={}, min_freq={}",
        f"{VOCAB_SIZE:,}", MIN_FREQUENCY,
    )
    tokenizer.train_from_iterator(
        get_training_corpus(all_datasets),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=[SPECIAL_TOKEN],
        show_progress=True,
    )

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save(os.path.join(TOKENIZER_DIR, "tokenizer.json"))

    gpt2_tokenizer = GPT2TokenizerFast(
        tokenizer_file=os.path.join(TOKENIZER_DIR, "tokenizer.json"),
        bos_token=SPECIAL_TOKEN,
        eos_token=SPECIAL_TOKEN,
        unk_token=SPECIAL_TOKEN,
        pad_token=SPECIAL_TOKEN,
        model_max_length=MAX_LENGTH,
    )
    gpt2_tokenizer.save_pretrained(TOKENIZER_DIR)

    test_text = "Xin chào Việt Nam! Đây là bài kiểm tra tokenizer."
    encoded = gpt2_tokenizer.encode(normalize_text(test_text))
    decoded = gpt2_tokenizer.decode(encoded)
    logger.info("Vocab size: {}", f"{len(gpt2_tokenizer):,}")
    logger.info("Test: {} → {} tokens → {}", test_text, len(encoded), decoded)
    logger.info("Saved to: {}", TOKENIZER_DIR)

    return gpt2_tokenizer

if __name__ == "__main__":
    train_tokenizer()
