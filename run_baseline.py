#!/usr/bin/env python3
"""Section 1: baseline profiling — tokens/sec, lm_head fraction, full-vocab PPL."""
import sys

import pandas as pd

from src.config import CFG
from src.model_utils import load_model, load_raw_dataset, build_bench_prompts
from src.baseline import run_baseline_profiling, full_vocab_ppl
from src.plots import plot_baseline_breakdown
from src.logging_utils import setup_logging


def main():
    setup_logging(CFG.results_dir / "run_baseline.log")
    # Always re-run baseline (it's fast and needed by every other script)
    model, tokenizer = load_model()
    raw_dataset = load_raw_dataset()
    bench_prompts = build_bench_prompts(tokenizer, raw_dataset)

    print("\n=== Section 1: Baseline Profiling ===")
    summary = run_baseline_profiling(model, bench_prompts)

    print("\nComputing full-vocab perplexity…")
    baseline_ppl = full_vocab_ppl(model, tokenizer, raw_dataset)
    summary["baseline_ppl"] = baseline_ppl

    pd.DataFrame([summary]).to_csv(CFG.baseline_csv, index=False)
    print(f"Saved → {CFG.baseline_csv}")

    plot_baseline_breakdown(summary)
    print("Done.")


if __name__ == "__main__":
    main()
