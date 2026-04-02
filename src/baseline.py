"""Baseline profiling: full-vocabulary decode timing and perplexity."""
import math
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.config import CFG
from src.model_utils import DEVICE


def _sync(device=None):
    """Synchronise the given device (or DEVICE if not specified)."""
    dev = device if device is not None else DEVICE
    if str(dev).startswith("mps"):
        torch.mps.synchronize()
    elif str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)


def timed_lm_head(hidden: torch.Tensor, weight: torch.Tensor):
    """Run lm_head projection; return (logits, elapsed_seconds).

    hidden and weight may live on different devices when the model is sharded
    across multiple GPUs (device_map='auto'). We move hidden to weight's device
    before the matmul, so the operation is always local.
    """
    lm_dev = weight.device
    hidden_lm = hidden.to(lm_dev)
    _sync(lm_dev)
    t0 = time.perf_counter()
    logits = hidden_lm @ weight.T
    _sync(lm_dev)
    return logits, time.perf_counter() - t0


def baseline_decode_with_profile(model, input_ids: torch.Tensor,
                                  max_new_tokens: int = CFG.bench_max_new_tokens) -> Dict:
    """Greedy decode with full vocab; track per-step timings."""
    input_ids = input_ids.unsqueeze(0).to(DEVICE)
    past_key_values = None
    total_times, lmhead_times = [], []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            t_start = time.perf_counter()
            out = model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1][:, -1, :]
            past_key_values = out.past_key_values

            logits, lmhead_t = timed_lm_head(hidden, model.lm_head.weight)
            next_token = logits[0].argmax(dim=-1, keepdim=True).unsqueeze(0).to(DEVICE)
            input_ids = next_token

            _sync(DEVICE)
            total_times.append(time.perf_counter() - t_start)
            lmhead_times.append(lmhead_t)

    return {"total_times": total_times, "lmhead_times": lmhead_times}


def run_baseline_profiling(model, bench_prompts: List[torch.Tensor]) -> Dict:
    """Warm-up then profile all bench prompts. Returns summary dict."""
    print("Warming up…")
    _ = baseline_decode_with_profile(model, bench_prompts[0], max_new_tokens=5)

    results = []
    for prompt in tqdm(bench_prompts, desc="Baseline profiling"):
        results.append(baseline_decode_with_profile(model, prompt))

    all_total = np.array([t for r in results for t in r["total_times"]])
    all_lmhead = np.array([t for r in results for t in r["lmhead_times"]])

    summary = {
        "tokens_per_sec": float(1.0 / all_total.mean()),
        "mean_total_ms": float(all_total.mean() * 1000),
        "mean_lmhead_ms": float(all_lmhead.mean() * 1000),
        "lmhead_frac": float(all_lmhead.mean() / all_total.mean()),
    }

    print(f"\n=== Baseline Results ===")
    print(f"  Tokens/sec:        {summary['tokens_per_sec']:.2f}")
    print(f"  Mean total/token:  {summary['mean_total_ms']:.2f} ms")
    print(f"  Mean lm_head:      {summary['mean_lmhead_ms']:.2f} ms")
    print(f"  lm_head fraction:  {summary['lmhead_frac']*100:.1f}%")
    return summary


def full_vocab_ppl(model, tokenizer, raw_dataset,
                   max_tokens: int = CFG.ppl_max_tokens) -> float:
    """Compute perplexity on WikiText-2 test split with full vocabulary."""
    lines = [l for l in raw_dataset["test"]["text"] if len(l.strip()) > 20]
    tokens = torch.cat([tokenizer.encode(l, return_tensors="pt")[0] for l in lines])[:max_tokens]
    input_ids = tokens.unsqueeze(0).to(DEVICE)
    stride, seq_len, nlls = 512, input_ids.size(1), []

    with torch.no_grad():
        for begin in tqdm(range(0, seq_len - 1, stride), desc="PPL baseline", leave=False):
            end = min(begin + stride, seq_len - 1)
            chunk = input_ids[:, begin : end + 1]
            out = model(input_ids=chunk)
            logits = out.logits[0, :-1, :]
            targets = chunk[0, 1:]
            nll = F.cross_entropy(logits, targets, reduction="none")
            nlls.extend(nll.float().cpu().tolist())

    ppl = math.exp(np.mean(nlls))
    print(f"  Baseline perplexity: {ppl:.3f}")
    return ppl


def load_baseline_summary() -> Dict:
    """Read saved baseline_summary.csv; raise if missing."""
    if not CFG.baseline_csv.exists():
        raise FileNotFoundError(
            f"{CFG.baseline_csv} not found. Run run_baseline.py first."
        )
    row = pd.read_csv(CFG.baseline_csv).iloc[0].to_dict()
    return row
