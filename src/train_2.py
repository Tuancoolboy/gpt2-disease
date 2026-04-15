#!/usr/bin/env python3
"""Continued pretraining on health corpus (stage 2)."""

from loguru import logger
import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

from src.config import (
    MODEL_DIR, BF16, WARMUP_RATIO,
    HEALTH_DATA_PATH, HEALTH_CHECKPOINT_DIR,
    HEALTH_EPOCHS, HEALTH_BATCH_SIZE, HEALTH_LEARNING_RATE,
    HEALTH_WEIGHT_DECAY, HEALTH_MAX_LENGTH,
    DATALOADER_NUM_WORKERS, WANDB_RUN_NAME_HEALTH,
    SEED, PREPROCESSING_NUM_WORKERS,
)
from src.utils import load_gpt2, normalize_text, PerplexityCallback, perplexity


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}", device)

    dtype = torch.bfloat16 if BF16 else torch.float32
    model, tokenizer, _ = load_gpt2(
        MODEL_DIR,
        torch_dtype=dtype,
        tie_weights=True,
        pad_token_to_eos=True,
        eval_mode=False,
    )
    logger.info(
        "Loaded base model from {} — {:.1f}M params",
        MODEL_DIR,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    logger.info("Loading data from {}", HEALTH_DATA_PATH)
    ds = load_dataset("json", data_files=HEALTH_DATA_PATH, split="train")
    logger.info("Samples: {}", f"{len(ds):,}")

    eos = tokenizer.eos_token

    def tokenize(batch):
        texts = [normalize_text(t) + eos for t in batch["text"]]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=HEALTH_MAX_LENGTH,
            padding="max_length",
        )

        # For plain causal LM continued pretraining:
        # labels are input_ids, except padding positions.
        enc["labels"] = [
            [
                tok if enc["attention_mask"][j][i] == 1 else -100
                for i, tok in enumerate(ids)
            ]
            for j, ids in enumerate(enc["input_ids"])
        ]
        return enc

    ds = ds.map(
        tokenize,
        batched=True,
        num_proc=PREPROCESSING_NUM_WORKERS,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )
    ds = ds.train_test_split(test_size=0.1, seed=SEED)
    logger.info("Train: {} | Eval: {}", len(ds["train"]), len(ds["test"]))

    args = TrainingArguments(
        output_dir=HEALTH_CHECKPOINT_DIR,
        num_train_epochs=HEALTH_EPOCHS,
        per_device_train_batch_size=HEALTH_BATCH_SIZE,
        per_device_eval_batch_size=HEALTH_BATCH_SIZE,
        learning_rate=HEALTH_LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=HEALTH_WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=BF16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to=["wandb"],
        run_name=WANDB_RUN_NAME_HEALTH,
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        callbacks=[PerplexityCallback()],
    )

    logger.info(
        "Training: epochs={}, batch={}, lr={}",
        HEALTH_EPOCHS, HEALTH_BATCH_SIZE, HEALTH_LEARNING_RATE,
    )
    trainer.train()

    final_dir = os.path.join(HEALTH_CHECKPOINT_DIR, "final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)

    result = trainer.evaluate()
    loss = result["eval_loss"]
    ppl = perplexity(loss)
    logger.info("Done. Eval loss={:.4f}, perplexity={:.2f}", loss, ppl)
    logger.info("Saved to {}", final_dir)

    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(os.path.join(HEALTH_CHECKPOINT_DIR, "log_history.csv"), index=False)


if __name__ == "__main__":
    main()