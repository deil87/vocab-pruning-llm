#!/usr/bin/env python3
"""Section 7: MLP transition graph router sweep."""
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from src.config import CFG
from src.model_utils import load_model, load_raw_dataset, build_bench_prompts, lm_head_cpu_fp32, DEVICE
from src.baseline import load_baseline_summary
from src.mlp_graph import load_or_build_graph
from src.routers import get_graph_shortlist
from src.evaluate import evaluate_router, compute_perplexity
from src.logging_utils import setup_logging


def main():
    setup_logging(CFG.results_dir / "run_graph.log")
    if CFG.graph_csv.exists():
        print("MLP graph results already exist — skipping.")
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

    print("\n=== MLP Transition Graph Router ===")
    graph_edges, graph_weights = load_or_build_graph(model, lm_head_weight)

    def graph_fn(h, k):
        return get_graph_shortlist(h, k, graph_edges, lm_head_norm_dev)

    graph_results = []
    for k in CFG.k_values:
        print(f"\n--- MLP Graph K={k} ---")
        r = evaluate_router(model, bench_prompts, router_fn=graph_fn, k=k)
        r["perplexity"] = compute_perplexity(
            model, tokenizer, raw_dataset, router_fn=graph_fn, k=k,
        )
        r["delta_ppl"] = r["perplexity"] - baseline_ppl
        graph_results.append(r)
        print(f"  Acc={r['acceptance_rate']:.3f}  PPL={r['perplexity']:.3f}  "
              f"ΔPPL={r['delta_ppl']:+.3f}  TPS={r['tokens_per_sec']:.2f}")

    pd.DataFrame(graph_results).to_csv(CFG.graph_csv, index=False)
    print(f"Saved → {CFG.graph_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
