"""Load model, tokenizer, dataset and build benchmark prompts."""
import random

import torch
from datasets import load_dataset as hf_load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import CFG


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def load_model(attn_implementation: str = None):
    """Return (model, tokenizer) loaded to DEVICE in bfloat16.

    Args:
        attn_implementation: passed to from_pretrained. Use "eager" when
            output_attentions=True is needed (e.g. attention router). Defaults
            to None (transformers chooses, typically "sdpa").
    """
    print(f"Loading tokenizer: {CFG.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {CFG.model_name}")
    kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto" if DEVICE.type != "mps" else None,
    )
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(CFG.model_name, **kwargs)
    if DEVICE.type == "mps":
        model = model.to(DEVICE)
    model.eval()

    print(f"  Vocab size:  {model.config.vocab_size:,}")
    print(f"  Hidden dim:  {model.config.hidden_size}")
    print(f"  Layers:      {model.config.num_hidden_layers}")
    print(f"  dtype:       {model.dtype}")
    print(f"  Device:      {DEVICE}")
    return model, tokenizer


def load_raw_dataset():
    """Return the raw HuggingFace DatasetDict for WikiText-2."""
    ds = hf_load_dataset(CFG.dataset_name, CFG.dataset_config)
    print(ds)
    return ds


def build_bench_prompts(tokenizer, raw_dataset) -> list[torch.Tensor]:
    """
    Sample N_BENCH_PROMPTS fixed-length prompts from the test split.
    Each prompt is a LongTensor of shape [PROMPT_LEN].
    """
    random.seed(CFG.seed)
    lines = [l for l in raw_dataset["test"]["text"] if len(l.strip()) > 20]
    parts = [tokenizer.encode(l, return_tensors="pt")[0] for l in lines]
    all_tokens = torch.cat(parts)

    prompts = []
    step = len(all_tokens) // (CFG.n_bench_prompts + 1)
    for i in range(CFG.n_bench_prompts):
        start = i * step
        chunk = all_tokens[start : start + CFG.prompt_len]
        if len(chunk) == CFG.prompt_len:
            prompts.append(chunk)

    print(f"Benchmark prompts: {len(prompts)} x {CFG.prompt_len} tokens")
    return prompts


def lm_head_cpu_fp32(model) -> torch.Tensor:
    """Return lm_head weight matrix as float32 on CPU: [V, d]."""
    return model.lm_head.weight.detach().float().cpu()
