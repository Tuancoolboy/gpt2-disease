#!/usr/bin/env python3
"""Shared utility functions."""

import math
import unicodedata
from pathlib import Path
import wandb
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainerCallback


def perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")

class PerplexityCallback(TrainerCallback):
    """Add train/eval perplexity into Trainer logs."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        if "loss" in logs:
            logs["train_perplexity"] = perplexity(logs["loss"])

        if "eval_loss" in logs:
            logs["eval_perplexity"] = perplexity(logs["eval_loss"])

def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization, returning empty string for None."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", text)

def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string (B/KB/MB/GB/TB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

def load_gpt2(model_dir, *, torch_dtype=None, tie_weights=False,
              pad_token_to_eos=False, eval_mode=True):
    """Load GPT2LMHeadModel + tokenizer. Returns (model, tokenizer, device)."""
    model_dir = str(Path(model_dir).resolve())
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    if pad_token_to_eos and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kw = {"torch_dtype": torch_dtype} if torch_dtype else {}
    model = GPT2LMHeadModel.from_pretrained(model_dir, **kw)
    if tie_weights:
        model.tie_weights()
    if pad_token_to_eos:
        model.config.pad_token_id = tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if eval_mode:
        model.eval()

    return model, tokenizer, device

def generate_texts(model, tokenizer, device, prompt, **kwargs):
    """Run model.generate on NFC-normalized prompt, return decoded strings."""
    prompt = normalize_text(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )

    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
