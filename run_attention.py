#!/usr/bin/env python3
"""Section 6: Attention Graph router sweep."""
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from src.config import CFG
from src.model_utils import load_model, load_raw_dataset, build_bench_prompts, lm_head_cpu_fp32, DEVICE
from src.baseline import load_baseline_summary
from src.routers import get_attention_shortlist
from src.evaluate import evaluate_attention_router, compute_perplexity_attention
from src.logging_utils import setup_logging

# Attention router is ~4-5x slower than other routers (output_attentions=True +
# eager attention).  Use fewer prompts so the full K sweep finishes in ~1 hour.
ATTN_N_PROMPTS = 10


def main():
    setup_logging(CFG.results_dir / "run_attention.log")

    # Load existing results so we can resume a partial run
    existing = []
    done_ks = set()
    if CFG.attn_csv.exists():
        existing = pd.read_csv(CFG.attn_csv).to_dict("records")
        done_ks = {r["k"] for r in existing}
        remaining = [k for k in CFG.k_values if k not in done_ks]
        if not remaining:
            print("Attention graph results already complete — skipping.")
            return
        print(f"Resuming attention router: done {sorted(done_ks)}, remaining {remaining}")

    model, tokenizer = load_model(attn_implementation="eager")
    raw_dataset = load_raw_dataset()
    bench_prompts = build_bench_prompts(tokenizer, raw_dataset)
    baseline = load_baseline_summary()
    baseline_ppl = baseline["baseline_ppl"]

    lm_head_weight = lm_head_cpu_fp32(model)
    lm_head_norm_dev = F.normalize(
        model.lm_head.weight.float(), dim=-1
    )  # [V, d] float32 on DEVICE

    def attn_fn(h, k, seq_ids, attn_weights):
        return get_attention_shortlist(h, k, seq_ids, attn_weights, lm_head_norm_dev)

    print(f"\n=== Attention Graph Router  (n_prompts={ATTN_N_PROMPTS}) ===")
    attn_results = list(existing)
    for k in CFG.k_values:
        if k in done_ks:
            print(f"  K={k} already done — skipping.")
            continue
        print(f"\n--- Attention K={k} ---")
        r = evaluate_attention_router(
            model, bench_prompts, attn_shortlist_fn=attn_fn, k=k,
            n_prompts=ATTN_N_PROMPTS,
        )
        r["perplexity"] = compute_perplexity_attention(
            model, tokenizer, raw_dataset, attn_shortlist_fn=attn_fn, k=k,
        )
        r["delta_ppl"] = r["perplexity"] - baseline_ppl
        attn_results.append(r)
        print(f"  Acc={r['acceptance_rate']:.3f}  PPL={r['perplexity']:.3f}  "
              f"ΔPPL={r['delta_ppl']:+.3f}  TPS={r['tokens_per_sec']:.2f}")

        # Save incrementally so a timeout doesn't lose completed K values
        pd.DataFrame(attn_results).to_csv(CFG.attn_csv, index=False)
        print(f"  Saved → {CFG.attn_csv}")

    print(f"\nSaved → {CFG.attn_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
