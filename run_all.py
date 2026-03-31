#!/usr/bin/env python3
"""
run_all.py — Run the full experiment pipeline end-to-end.

Each step is resume-safe: if its output CSV already exists it will be skipped.
Re-run this script at any time to continue from where you left off.

Steps:
  1. Baseline profiling
  2. Static + Cosine router sweep
  3. Cluster router sweep
  4. Dual-Encoder router (train + step-level + prefetch/refresh)
  5. Attention Graph router sweep
  6. MLP Transition Graph router sweep
  7. Vibe-Aware Speculative Decoding + draft-length sweep
  8. Assemble results + generate all plots
"""
import time

import run_baseline
import run_static
import run_cluster
import run_dual_encoder
import run_attention
import run_graph
import run_speculative
import run_plots


def run_step(name: str, fn):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"  [{name}] finished in {mins}m {secs}s")


def main():
    t_total = time.perf_counter()

    run_step("Baseline profiling",           run_baseline.main)
    run_step("Static + Cosine routers",      run_static.main)
    run_step("Cluster router",               run_cluster.main)
    run_step("Dual-Encoder router",          run_dual_encoder.main)
    run_step("Attention Graph router",       run_attention.main)
    run_step("MLP Transition Graph router",  run_graph.main)
    run_step("Vibe-Aware Speculative",       run_speculative.main)
    run_step("Plots + final summary",        run_plots.main)

    total = time.perf_counter() - t_total
    h, rem = divmod(int(total), 3600)
    m, s = divmod(rem, 60)
    print(f"\n{'='*60}")
    print(f"  ALL DONE in {h}h {m}m {s}s")
    print(f"  Results in: results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
