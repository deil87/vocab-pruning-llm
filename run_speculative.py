#!/usr/bin/env python3
"""Section 8: Vibe-aware speculative decoding + draft-length sweep."""
import sys

import pandas as pd
import torch
import torch.nn.functional as F

from src.config import CFG
from src.model_utils import load_model, load_raw_dataset, build_bench_prompts, lm_head_cpu_fp32, DEVICE
from src.baseline import load_baseline_summary
from src.mlp_graph import load_or_build_graph
from src.evaluate import evaluate_speculative, evaluate_cosine_speculative
from src.logging_utils import setup_logging

# Speculative decoding requires 2 full forward passes per call + output_attentions=True
# on the first pass.  Reduce prompt count for the sweep to keep runtime manageable.
SWEEP_N_PROMPTS = 10


def main():
    setup_logging(CFG.results_dir / "run_speculative.log")
    if CFG.spec_csv.exists() and CFG.spec_sweep_csv.exists():
        print("Speculative decoding results already exist — skipping.")
        return

    model, tokenizer = load_model(attn_implementation="eager")
    raw_dataset = load_raw_dataset()
    bench_prompts = build_bench_prompts(tokenizer, raw_dataset)
    baseline = load_baseline_summary()
    baseline_tps = baseline["tokens_per_sec"]

    lm_head_weight = lm_head_cpu_fp32(model)
    lm_head_norm_dev = F.normalize(
        model.lm_head.weight.float(), dim=-1
    )  # [V, d] float32 on DEVICE

    print("\n=== MLP Graph (needed for speculative drafts) ===")
    graph_edges, _ = load_or_build_graph(model, lm_head_weight)

    # ── Default draft_len evaluation ───────────────────────────────────────────
    if not CFG.spec_csv.exists():
        print("\n=== Vibe-Aware Speculative Decoding ===")
        spec_result = evaluate_speculative(
            model, bench_prompts, graph_edges, lm_head_norm_dev, lm_head_weight,
            baseline_tps=baseline_tps,
        )
        spec_result["method"] = "Combined (Attn+Graph)"

        cosine_result = evaluate_cosine_speculative(
            model, bench_prompts, lm_head_norm_dev, baseline_tps=baseline_tps,
        )
        cosine_result["method"] = "Cosine baseline"

        df = pd.DataFrame([spec_result, cosine_result])
        df.to_csv(CFG.spec_csv, index=False)
        print(f"Saved → {CFG.spec_csv}")
    else:
        print("Spec decoding results already exist — skipping.")

    # ── Draft-length sweep ─────────────────────────────────────────────────────
    if not CFG.spec_sweep_csv.exists():
        print(f"\n=== Speculative Decoding: Draft-Length Sweep  (n_prompts={SWEEP_N_PROMPTS}) ===")
        # Load existing partial results so we can resume
        existing_sweep = []
        done_Ds_methods: set = set()
        partial_path = CFG.spec_sweep_csv.with_suffix(".partial.csv")
        if partial_path.exists():
            existing_sweep = pd.read_csv(partial_path).to_dict("records")
            done_Ds_methods = {(r["draft_len"], r["method"]) for r in existing_sweep}
            print(f"  Resuming sweep: {len(existing_sweep)} rows already done.")

        sweep = list(existing_sweep)
        for D in CFG.spec_draft_len_sweep:
            print(f"\n--- Draft len D={D} ---")
            if (D, "Combined (Attn+Graph)") not in done_Ds_methods:
                r = evaluate_speculative(
                    model, bench_prompts, graph_edges, lm_head_norm_dev, lm_head_weight,
                    baseline_tps=baseline_tps, draft_len=D, n_prompts=SWEEP_N_PROMPTS,
                )
                r["method"] = "Combined (Attn+Graph)"
                sweep.append(r)
                pd.DataFrame(sweep).to_csv(partial_path, index=False)
            else:
                print(f"  Combined already done — skipping.")

            if (D, "Cosine baseline") not in done_Ds_methods:
                rc = evaluate_cosine_speculative(
                    model, bench_prompts, lm_head_norm_dev,
                    baseline_tps=baseline_tps, draft_len=D, n_prompts=SWEEP_N_PROMPTS,
                )
                rc["method"] = "Cosine baseline"
                sweep.append(rc)
                pd.DataFrame(sweep).to_csv(partial_path, index=False)
            else:
                print(f"  Cosine already done — skipping.")

        pd.DataFrame(sweep).to_csv(CFG.spec_sweep_csv, index=False)
        if partial_path.exists():
            partial_path.unlink()
        print(f"Saved → {CFG.spec_sweep_csv}")
    else:
        print("Spec sweep results already exist — skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
