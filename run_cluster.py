#!/usr/bin/env python3
"""Section 4: k-means cluster router sweep."""
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from src.config import CFG
from src.model_utils import load_model, load_raw_dataset, build_bench_prompts, lm_head_cpu_fp32
from src.baseline import load_baseline_summary
from src.routers import get_cluster_shortlist
from src.evaluate import evaluate_router, compute_perplexity
from src.logging_utils import setup_logging


def main():
    setup_logging(CFG.results_dir / "run_cluster.log")
    if CFG.cluster_csv.exists():
        print("Cluster results already exist — skipping.")
        return

    model, tokenizer = load_model()
    raw_dataset = load_raw_dataset()
    bench_prompts = build_bench_prompts(tokenizer, raw_dataset)
    baseline = load_baseline_summary()
    baseline_ppl = baseline["baseline_ppl"]

    lm_head_weight = lm_head_cpu_fp32(model)
    lm_head_norm = F.normalize(lm_head_weight, dim=-1)

    print(f"\n=== Cluster Router (k-means, {CFG.n_kmeans_clusters} clusters) ===")
    print("Fitting k-means…")
    kmeans = MiniBatchKMeans(
        n_clusters=CFG.n_kmeans_clusters,
        random_state=CFG.seed,
        batch_size=4096,
        n_init=3,
    )
    kmeans.fit(lm_head_weight.numpy())
    cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    cluster_centers_norm = F.normalize(cluster_centers, dim=-1).to(model.device)

    cluster_to_tokens = {
        c: (cluster_labels == c).nonzero(as_tuple=True)[0]
        for c in range(CFG.n_kmeans_clusters)
    }
    sizes = [len(cluster_to_tokens[c]) for c in range(CFG.n_kmeans_clusters)]
    print(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.0f}")

    cluster_results = []
    for k in CFG.k_values:
        print(f"\n--- Cluster K={k} ---")
        router_fn = lambda h, k, cn=cluster_centers_norm, ct=cluster_to_tokens: \
            get_cluster_shortlist(h, cn, ct, k)
        r = evaluate_router(model, bench_prompts, router_fn=router_fn, k=k)
        r["perplexity"] = compute_perplexity(
            model, tokenizer, raw_dataset, router_fn=router_fn, k=k,
        )
        r["delta_ppl"] = r["perplexity"] - baseline_ppl
        cluster_results.append(r)
        print(f"  Acc={r['acceptance_rate']:.3f}  PPL={r['perplexity']:.3f}  "
              f"ΔPPL={r['delta_ppl']:+.3f}  TPS={r['tokens_per_sec']:.2f}")

    pd.DataFrame(cluster_results).to_csv(CFG.cluster_csv, index=False)
    print(f"Saved → {CFG.cluster_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
