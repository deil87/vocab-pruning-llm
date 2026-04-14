#!/usr/bin/env python3
"""Hybrid Attention + MLP Graph router sweep.

Design:
    At each decode step, live attention weights from the last transformer layer
    identify the most-attended past token positions.  Those token IDs are used as
    anchors for a 1-hop walk in the prebuilt MLP transition graph, yielding a
    shortlist without a full-vocabulary cosine scan.

    Compared with the original Attention Graph router (run_attention.py):
      - No per-attended-token full-vocab cosine scan  →  O(n_pos * M) instead of
        O(n_pos * V * d) for the expansion step.
      - output_attentions=True is still passed to the full model forward (current
        HuggingFace transformers forces eager mode on all layers when this flag is
        set).  The captured_attn hook only *reads* the last layer's weights;
        future work can make this truly single-layer by patching the last attention
        module's forward directly.

    Compared with the MLP Graph router (run_graph.py):
      - Anchor tokens come from live attention weights instead of cosine similarity
        to h_T  →  prompt-aware anchors that can surface rare tokens being copied
        or referred back to.
      - No full-vocab cosine scan for anchor selection.

Results are saved incrementally after each K so a timeout loses at most one K.
"""
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from src.config import CFG
from src.model_utils import load_model, load_raw_dataset, build_bench_prompts, lm_head_cpu_fp32, DEVICE
from src.baseline import load_baseline_summary
from src.mlp_graph import load_or_build_graph
from src.routers import get_hybrid_shortlist
from src.evaluate import evaluate_hybrid_router, compute_perplexity_hybrid
from src.logging_utils import setup_logging

# output_attentions=True forces eager mode on all layers in current HF transformers.
# Use fewer prompts than the default so the full K sweep finishes in reasonable time.
# This is still more than the attention router (10 prompts) because the graph
# expansion is cheaper than the per-attended-token cosine scan.
HYBRID_N_PROMPTS = 25


def main():
    setup_logging(CFG.results_dir / "run_hybrid.log")

    # Resume support: load any previously completed K values
    existing = []
    done_ks: set = set()
    if CFG.hybrid_csv.exists():
        existing = pd.read_csv(CFG.hybrid_csv).to_dict("records")
        done_ks = {r["k"] for r in existing}
        remaining = [k for k in CFG.k_values if k not in done_ks]
        if not remaining:
            print("Hybrid router results already complete — skipping.")
            return
        print(f"Resuming hybrid router: done {sorted(done_ks)}, remaining {remaining}")

    # Load model in eager mode (required for output_attentions=True).
    # Future optimisation: patch only the last attention layer to eager and load
    # the rest with attn_implementation=None (SDPA/Flash).
    model, tokenizer = load_model(attn_implementation="eager")
    raw_dataset = load_raw_dataset()
    bench_prompts = build_bench_prompts(tokenizer, raw_dataset)
    baseline = load_baseline_summary()
    baseline_ppl = baseline["baseline_ppl"]

    lm_head_weight = lm_head_cpu_fp32(model)
    lm_head_norm_dev = F.normalize(
        model.lm_head.weight.float(), dim=-1
    )  # [V, d] float32 on DEVICE

    print("\n=== Hybrid Attention + MLP Graph Router ===")
    print(f"  attn_top_positions : {CFG.hybrid_attn_top_positions}")
    print(f"  graph_hops         : {CFG.hybrid_graph_hops}  (1-hop walk)")
    print(f"  n_prompts          : {HYBRID_N_PROMPTS}")

    graph_edges, _ = load_or_build_graph(model, lm_head_weight)
    # graph_edges: [V, M] int32 on CPU

    def hybrid_fn(hidden, k, seq_ids, captured_attn):
        return get_hybrid_shortlist(
            hidden, k, seq_ids, captured_attn, graph_edges, lm_head_norm_dev
        )

    hybrid_results = list(existing)
    for k in CFG.k_values:
        if k in done_ks:
            print(f"  K={k} already done — skipping.")
            continue

        print(f"\n--- Hybrid K={k} ---")
        r = evaluate_hybrid_router(
            model, bench_prompts,
            hybrid_shortlist_fn=hybrid_fn,
            k=k,
            n_prompts=HYBRID_N_PROMPTS,
        )
        r["perplexity"] = compute_perplexity_hybrid(
            model, tokenizer, raw_dataset,
            hybrid_shortlist_fn=hybrid_fn,
            k=k,
        )
        r["delta_ppl"] = r["perplexity"] - baseline_ppl
        hybrid_results.append(r)
        print(
            f"  Acc={r['acceptance_rate']:.3f}  PPL={r['perplexity']:.3f}  "
            f"ΔPPL={r['delta_ppl']:+.3f}  TPS={r['tokens_per_sec']:.2f}  "
            f"lmhead_frac={r['lmhead_frac']:.3f}"
        )

        # Incremental save — safe against timeout
        pd.DataFrame(hybrid_results).to_csv(CFG.hybrid_csv, index=False)
        print(f"  Saved → {CFG.hybrid_csv}")

    print(f"\nDone.  Results → {CFG.hybrid_csv}")


if __name__ == "__main__":
    main()
