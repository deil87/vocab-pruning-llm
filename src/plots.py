"""Generate all result plots and save to results/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CFG

sns.set_theme(style="darkgrid", palette="muted")


def _save(fig, name: str):
    path = CFG.results_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_baseline_breakdown(baseline_summary: dict):
    """Bar chart: lm_head vs rest of forward-pass time."""
    lm_ms = baseline_summary["mean_lmhead_ms"]
    total_ms = baseline_summary["mean_total_ms"]
    rest_ms = total_ms - lm_ms

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["lm_head", "rest"], [lm_ms, rest_ms], color=["#2196F3", "#90CAF9"])
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Mean time per token (ms)")
    for i, v in enumerate([lm_ms, rest_ms]):
        axes[0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)

    frac = baseline_summary["lmhead_frac"]
    axes[1].pie([frac, 1 - frac], labels=["lm_head", "rest"],
                autopct="%1.1f%%", colors=["#2196F3", "#90CAF9"])
    axes[1].set_title("lm_head fraction of total step time")

    fig.suptitle(f"Baseline: {baseline_summary['tokens_per_sec']:.1f} tok/s", fontsize=13)
    plt.tight_layout()
    _save(fig, "baseline_profile.png")


def plot_acceptance_rate(df_all: pd.DataFrame):
    routers = [
        "Static Top-K", "Cosine", "Cluster",
        "Dual-Encoder (step)", "Dual-Encoder (prefetch+refresh)",
        "Attention Graph", "MLP Graph",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in routers:
        sub = df_all[df_all["router"] == name].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["acceptance_rate"], marker="o", label=name)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Baseline")
    ax.set_xlabel("Shortlist size K")
    ax.set_ylabel("Token acceptance rate")
    ax.set_title("Acceptance Rate vs. Shortlist Size K")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "acceptance_rate_vs_k.png")


def plot_ppl_degradation(df_all: pd.DataFrame):
    routers = [
        "Static Top-K", "Cosine", "Cluster",
        "Dual-Encoder (step)", "Dual-Encoder (prefetch+refresh)",
        "Attention Graph", "MLP Graph",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in routers:
        sub = df_all[df_all["router"] == name].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["delta_ppl"], marker="o", label=name)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="Baseline (ΔPPL=0)")
    ax.set_xlabel("Shortlist size K")
    ax.set_ylabel("ΔPPL (pruned − full vocab)")
    ax.set_title("Perplexity Degradation vs. Shortlist Size K")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "ppl_degradation_vs_k.png")


def plot_throughput(df_all: pd.DataFrame, baseline_tps: float):
    routers = [
        "Static Top-K", "Cosine", "Cluster",
        "Dual-Encoder (step)", "Dual-Encoder (prefetch+refresh)",
        "Attention Graph", "MLP Graph",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in routers:
        sub = df_all[df_all["router"] == name].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["tokens_per_sec"], marker="o", label=name)
    ax.axhline(baseline_tps, color="black", linestyle="--", linewidth=1, label="Baseline")
    ax.set_xlabel("Shortlist size K")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput vs. Shortlist Size K")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "throughput_vs_k.png")


def plot_pareto(df_all: pd.DataFrame, baseline_tps: float):
    routers = [
        "Static Top-K", "Cosine", "Cluster",
        "Dual-Encoder (step)", "Dual-Encoder (prefetch+refresh)",
        "Attention Graph", "MLP Graph",
    ]
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, name in enumerate(routers):
        sub = df_all[df_all["router"] == name].sort_values("k")
        if sub.empty:
            continue
        ax.scatter(sub["tokens_per_sec"], sub["delta_ppl"],
                   c=colours[i % len(colours)], label=name, s=80, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(
                f"K={int(row['k'])/1000:.0f}k",
                (row["tokens_per_sec"], row["delta_ppl"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, color=colours[i % len(colours)],
            )

    ax.scatter(baseline_tps, 0.0, marker="*", s=200, color="black", zorder=5, label="Baseline")
    ax.set_xlabel("Tokens/sec  (higher = better)")
    ax.set_ylabel("ΔPPL vs. baseline  (lower = better)")
    ax.set_title("Pareto Frontier: Speed vs. Accuracy")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    _save(fig, "pareto_frontier.png")


def plot_shortlist_growth(shortlist_sizes: list, steps_per_prompt: int, n_prompts: int):
    arr = np.array(shortlist_sizes).reshape(n_prompts, steps_per_prompt)
    mean_sizes = arr.mean(axis=0)
    p10 = np.percentile(arr, 10, axis=0)
    p90 = np.percentile(arr, 90, axis=0)
    steps = np.arange(1, steps_per_prompt + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, mean_sizes, label="Mean shortlist size")
    ax.fill_between(steps, p10, p90, alpha=0.2, label="10th–90th percentile")
    for r in range(CFG.refresh_every, steps_per_prompt, CFG.refresh_every):
        ax.axvline(r, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(
        CFG.refresh_every, color="gray", linestyle=":", linewidth=0.8,
        label=f"Refresh every {CFG.refresh_every} steps",
    )
    ax.set_xlabel("Decode step")
    ax.set_ylabel("Shortlist size (unique tokens)")
    ax.set_title("Shortlist Growth: Dual-Encoder Prefetch+Refresh")
    ax.legend()
    plt.tight_layout()
    _save(fig, "shortlist_growth.png")

    print(f"  Shortlist size at step 1:  {mean_sizes[0]:.0f}")
    print(f"  Shortlist size at step {steps_per_prompt}: {mean_sizes[-1]:.0f}")
    print(f"  Growth factor: {mean_sizes[-1]/mean_sizes[0]:.2f}x")


def plot_speculative_sweep(df_spec: pd.DataFrame, baseline_tps: float):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for method, grp in df_spec.groupby("method"):
        grp = grp.sort_values("draft_len")
        axes[0].plot(grp["draft_len"], grp["tokens_per_call"], marker="o", label=method)
        axes[1].plot(grp["draft_len"], grp["effective_tps"], marker="o", label=method)
        axes[2].plot(grp["draft_len"], grp["draft_acceptance_rate"], marker="o", label=method)

    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, label="Baseline (1 tok/call)")
    axes[0].set_xlabel("Draft length D")
    axes[0].set_ylabel("Tokens accepted per verifier call")
    axes[0].set_title("Accepted tokens/call vs Draft length")
    axes[0].legend(fontsize=8)

    axes[1].axhline(baseline_tps, color="black", linestyle="--", linewidth=1, label="Baseline TPS")
    axes[1].set_xlabel("Draft length D")
    axes[1].set_ylabel("Effective tokens/sec")
    axes[1].set_title("Effective Throughput vs Draft Length")
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel("Draft length D")
    axes[2].set_ylabel("Draft acceptance rate")
    axes[2].set_title("Draft Acceptance Rate vs Draft Length")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "speculative_decoding_sweep.png")
