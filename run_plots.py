#!/usr/bin/env python3
"""Section 9: Assemble all results into all_results.csv and generate all plots."""
import sys

import numpy as np
import pandas as pd

from src.config import CFG
from src.baseline import load_baseline_summary
from src.plots import (
    plot_baseline_breakdown,
    plot_acceptance_rate,
    plot_ppl_degradation,
    plot_throughput,
    plot_pareto,
    plot_shortlist_growth,
    plot_speculative_sweep,
)
from src.logging_utils import setup_logging


def load_csv_or_warn(path, name):
    if path.exists():
        return pd.read_csv(path)
    print(f"  WARNING: {name} results not found ({path}) — skipping.")
    return pd.DataFrame()


def main():
    setup_logging(CFG.results_dir / "run_plots.log")
    print("=== Assembling results ===")
    baseline = load_baseline_summary()
    baseline_tps = baseline["tokens_per_sec"]
    baseline_ppl = baseline["baseline_ppl"]

    def label(df, name):
        df = df.copy()
        df["router"] = name
        return df

    parts = []
    for path, name in [
        (CFG.static_csv,   "Static Top-K"),
        (CFG.cosine_csv,   "Cosine"),
        (CFG.cluster_csv,  "Cluster"),
        (CFG.dual_enc_csv, "Dual-Encoder (step)"),
        (CFG.prefetch_csv, "Dual-Encoder (prefetch+refresh)"),
        (CFG.attn_csv,     "Attention Graph"),
        (CFG.graph_csv,    "MLP Graph"),
    ]:
        df = load_csv_or_warn(path, name)
        if not df.empty:
            parts.append(label(df, name))

    baseline_row = pd.DataFrame([{
        "k": 128_256,
        "acceptance_rate": 1.0,
        "tokens_per_sec": baseline_tps,
        "lmhead_frac": baseline["lmhead_frac"],
        "perplexity": baseline_ppl,
        "delta_ppl": 0.0,
        "router": "Baseline (full vocab)",
    }])
    df_all = pd.concat([baseline_row] + parts, ignore_index=True)
    df_all.to_csv(CFG.all_results_csv, index=False)
    print(f"Saved → {CFG.all_results_csv}")

    # ── Summary table ──────────────────────────────────────────────────────────
    cols = ["router", "k", "acceptance_rate", "perplexity", "delta_ppl", "tokens_per_sec"]
    available = [c for c in cols if c in df_all.columns]
    print("\n=== Final Summary ===")
    print(f"Baseline perplexity:  {baseline_ppl:.3f}")
    print(f"Baseline tokens/sec:  {baseline_tps:.2f}")
    print(f"lm_head time frac:    {baseline['lmhead_frac']*100:.1f}%\n")
    print(df_all[available].to_string(index=False))

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n=== Generating plots ===")
    plot_baseline_breakdown(baseline)
    if not df_all.empty:
        plot_acceptance_rate(df_all)
        plot_ppl_degradation(df_all)
        plot_throughput(df_all, baseline_tps)
        plot_pareto(df_all, baseline_tps)

    if CFG.shortlist_sizes_npy.exists():
        sizes = np.load(CFG.shortlist_sizes_npy).tolist()
        plot_shortlist_growth(
            sizes,
            steps_per_prompt=CFG.bench_max_new_tokens,
            n_prompts=CFG.n_bench_prompts,
        )

    if CFG.spec_sweep_csv.exists():
        df_spec = pd.read_csv(CFG.spec_sweep_csv)
        plot_speculative_sweep(df_spec, baseline_tps)

    print("\nAll plots saved to results/")
    print("Done.")


if __name__ == "__main__":
    main()
