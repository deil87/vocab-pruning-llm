#!/usr/bin/env python3
"""Section 3: Static Top-K and Cosine router sweep."""
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from src.config import CFG
from src.model_utils import load_model, load_raw_dataset, build_bench_prompts, lm_head_cpu_fp32, DEVICE
from src.baseline import load_baseline_summary
from src.routers import build_static_index, get_static_shortlist, get_cosine_shortlist
from src.evaluate import evaluate_router, compute_perplexity, compute_perplexity_cosine
from src.logging_utils import setup_logging


def main():
    setup_logging(CFG.results_dir / "run_static.log")
    if CFG.static_csv.exists() and CFG.cosine_csv.exists():
        print("Static and Cosine results already exist — skipping.")
        return

    model, tokenizer = load_model()
    raw_dataset = load_raw_dataset()
    bench_prompts = build_bench_prompts(tokenizer, raw_dataset)
    baseline = load_baseline_summary()
    baseline_ppl = baseline["baseline_ppl"]

    lm_head_weight = lm_head_cpu_fp32(model)           # [V, d] float32 CPU
    static_sorted_idx = build_static_index(lm_head_weight)

    # Precompute normalised lm_head on cuda:0 (float32) — used by cosine router.
    # Force .to("cuda:0") so the matmul against hidden states (always on cuda:0)
    # works correctly under device_map="auto" where lm_head may land on cuda:1.
    _lm_head_f32 = model.lm_head.weight.float()
    _lm_head_dev0 = _lm_head_f32.to(next(model.parameters()).device)  # first param device = cuda:0
    lm_head_norm_dev = F.normalize(_lm_head_dev0, dim=-1)  # [V, d] float32 on cuda:0

    # ── Static router ──────────────────────────────────────────────────────────
    if not CFG.static_csv.exists():
        print("\n=== Static Top-K Router ===")
        static_results = []
        for k in CFG.k_values:
            print(f"\n--- K={k} ---")
            shortlist_cached = get_static_shortlist(static_sorted_idx, k)
            r = evaluate_router(model, bench_prompts,
                                 router_fn=lambda h, k, sl=shortlist_cached: sl, k=k)
            r["perplexity"] = compute_perplexity(
                model, tokenizer, raw_dataset,
                router_fn=lambda h, k, sl=shortlist_cached: sl, k=k,
            )
            r["delta_ppl"] = r["perplexity"] - baseline_ppl
            static_results.append(r)
            print(f"  Acc={r['acceptance_rate']:.3f}  PPL={r['perplexity']:.3f}  "
                  f"ΔPPL={r['delta_ppl']:+.3f}  TPS={r['tokens_per_sec']:.2f}")

        pd.DataFrame(static_results).to_csv(CFG.static_csv, index=False)
        print(f"Saved → {CFG.static_csv}")
    else:
        print("Static results already exist — skipping.")

    # ── Cosine router ──────────────────────────────────────────────────────────
    if not CFG.cosine_csv.exists():
        print("\n=== Cosine Router ===")
        cosine_results = []
        for k in CFG.k_values:
            print(f"\n--- K={k} ---")
            router_fn = lambda h, k, n=lm_head_norm_dev: get_cosine_shortlist(h, n, k)
            r = evaluate_router(model, bench_prompts, router_fn=router_fn, k=k)
            r["perplexity"] = compute_perplexity_cosine(
                model, tokenizer, raw_dataset, lm_head_norm_dev=lm_head_norm_dev, k=k,
            )
            r["delta_ppl"] = r["perplexity"] - baseline_ppl
            cosine_results.append(r)
            print(f"  Acc={r['acceptance_rate']:.3f}  PPL={r['perplexity']:.3f}  "
                  f"ΔPPL={r['delta_ppl']:+.3f}  TPS={r['tokens_per_sec']:.2f}")

        pd.DataFrame(cosine_results).to_csv(CFG.cosine_csv, index=False)
        print(f"Saved → {CFG.cosine_csv}")
    else:
        print("Cosine results already exist — skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
