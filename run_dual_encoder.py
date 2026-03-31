#!/usr/bin/env python3
"""Section 5: Dual-Encoder router — train, step-level sweep, prefetch+refresh."""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import CFG
from src.model_utils import load_model, load_raw_dataset, build_bench_prompts, lm_head_cpu_fp32, DEVICE
from src.baseline import load_baseline_summary
from src.dual_encoder import (
    build_completion_index, train_router, load_router, project_completion_index,
)
from src.routers import get_dual_encoder_shortlist, prefetch_shortlist, refresh_shortlist
from src.evaluate import evaluate_router, compute_perplexity, evaluate_prefetch_router, compute_perplexity_prefetch
from src.logging_utils import setup_logging


def main():
    setup_logging(CFG.results_dir / "run_dual_encoder.log")
    both_done = CFG.dual_enc_csv.exists() and CFG.prefetch_csv.exists()
    if both_done:
        print("Dual-encoder results already exist — skipping.")
        return

    model, tokenizer = load_model()
    raw_dataset = load_raw_dataset()
    bench_prompts = build_bench_prompts(tokenizer, raw_dataset)
    baseline = load_baseline_summary()
    baseline_ppl = baseline["baseline_ppl"]

    lm_head_weight = lm_head_cpu_fp32(model)
    lm_head_norm_dev = F.normalize(
        model.lm_head.weight.float(), dim=-1
    )  # [V, d] float32 on DEVICE

    # ── Build or load completion corpus ───────────────────────────────────────
    print("\n=== Building completion index ===")
    completion_token_lists, completion_index, train_tokens = build_completion_index(
        model, tokenizer, raw_dataset
    )

    # ── Train or load router ───────────────────────────────────────────────────
    if CFG.router_checkpoint.exists():
        print("\nLoading existing router checkpoint…")
        router, key_projector = load_router(model)
    else:
        print("\n=== Training RouterMLP ===")
        router, key_projector = train_router(model, train_tokens)

    # ── Project completion index → query space ─────────────────────────────────
    print("\nProjecting completion index…")
    completion_index_proj = project_completion_index(key_projector, completion_index, device=DEVICE)
    print(f"  Projected index shape: {completion_index_proj.shape}, device: {completion_index_proj.device}")

    # Move router to DEVICE for GPU-native inference (eliminates CPU round-trip)
    router = router.to(DEVICE)

    # Closure helpers
    def de_shortlist(h, k):
        return get_dual_encoder_shortlist(
            h, router, completion_index_proj, completion_token_lists, lm_head_norm_dev, k
        )

    def _prefetch(prompt_ids):
        return prefetch_shortlist(
            model, prompt_ids, router, completion_index_proj, completion_token_lists
        )

    def _refresh(shortlist, h_T):
        return refresh_shortlist(
            shortlist, h_T, router, completion_index_proj, completion_token_lists
        )

    # ── Step-level evaluation ──────────────────────────────────────────────────
    if not CFG.dual_enc_csv.exists():
        print("\n=== Dual-Encoder (step-level) ===")
        dual_enc_results = []
        for k in CFG.k_values:
            print(f"\n--- DE step-level K={k} ---")
            r = evaluate_router(model, bench_prompts, router_fn=de_shortlist, k=k)
            r["perplexity"] = compute_perplexity(
                model, tokenizer, raw_dataset, router_fn=de_shortlist, k=k,
            )
            r["delta_ppl"] = r["perplexity"] - baseline_ppl
            dual_enc_results.append(r)
            print(f"  Acc={r['acceptance_rate']:.3f}  PPL={r['perplexity']:.3f}  "
                  f"ΔPPL={r['delta_ppl']:+.3f}  TPS={r['tokens_per_sec']:.2f}")

        pd.DataFrame(dual_enc_results).to_csv(CFG.dual_enc_csv, index=False)
        print(f"Saved → {CFG.dual_enc_csv}")
    else:
        print("Step-level DE results already exist — skipping.")

    # ── Prefetch + refresh evaluation ──────────────────────────────────────────
    if not CFG.prefetch_csv.exists():
        print("\n=== Dual-Encoder (prefetch + refresh) ===")
        result = evaluate_prefetch_router(
            model, bench_prompts,
            prefetch_fn=_prefetch,
            refresh_fn=_refresh,
        )
        ppl = compute_perplexity_prefetch(
            model, tokenizer, raw_dataset,
            prefetch_fn=_prefetch,
            refresh_fn=_refresh,
        )
        shortlist_sizes = result.pop("shortlist_sizes")
        result["perplexity"] = ppl
        result["delta_ppl"] = ppl - baseline_ppl

        print(f"  Mean shortlist size: {result['mean_shortlist_size']:.0f}")
        print(f"  Acc={result['acceptance_rate']:.3f}  PPL={ppl:.3f}  "
              f"ΔPPL={result['delta_ppl']:+.3f}  TPS={result['tokens_per_sec']:.2f}")

        pd.DataFrame([result]).to_csv(CFG.prefetch_csv, index=False)
        np.save(CFG.shortlist_sizes_npy, np.array(shortlist_sizes))
        print(f"Saved → {CFG.prefetch_csv}, {CFG.shortlist_sizes_npy}")
    else:
        print("Prefetch+refresh results already exist — skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
