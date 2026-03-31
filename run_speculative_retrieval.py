#!/usr/bin/env python3
"""
Retrieval-based speculative decoding benchmark.

What this does
--------------
Builds a completion index from WikiText-2 train (50k windows of 8 tokens each,
indexed by the LLM hidden state at the split point). At each decoding step:
  1. One probe forward pass → h_T
  2. Retrieve top-K completions by cosine similarity to h_T
  3. Run one verifier pass per candidate; take the longest accepted prefix
  4. Commit best result; at least 1 token guaranteed

No output_attentions, no eager attention, no router MLP at inference time.
Standard SDPA/FlashAttention is used throughout.

Output files
------------
  results/ret_spec_completion_index.pt   ← reused across runs
  results/ret_spec_results.csv           ← default config (top_k=8, draft_len=8)
  results/ret_spec_sweep.csv             ← top_k × draft_len sweep
"""
import sys

import pandas as pd

from src.config import CFG
from src.model_utils import load_model, load_raw_dataset, build_bench_prompts
from src.baseline import load_baseline_summary
from src.speculative_retrieval import build_retrieval_index, evaluate_retrieval_speculative
from src.logging_utils import setup_logging


def main():
    setup_logging(CFG.results_dir / "run_speculative_retrieval.log")

    # ── Load model (default attn — SDPA, no eager penalty) ──────────────────
    model, tokenizer = load_model()   # no attn_implementation override
    raw_dataset = load_raw_dataset()
    bench_prompts = build_bench_prompts(tokenizer, raw_dataset)
    baseline_tps = load_baseline_summary()["tokens_per_sec"]
    print(f"Baseline TPS: {baseline_tps:.2f}")

    # ── Build / load completion index ────────────────────────────────────────
    token_seqs, index_norm = build_retrieval_index(model, tokenizer, raw_dataset)

    # ── Default config evaluation ────────────────────────────────────────────
    if not CFG.ret_spec_csv.exists():
        print(f"\n=== Retrieval Speculative Decoding "
              f"(top_k={CFG.ret_spec_top_k}, draft_len={CFG.ret_spec_draft_len}) ===")
        result = evaluate_retrieval_speculative(
            model, bench_prompts, index_norm, token_seqs,
            baseline_tps=baseline_tps,
            top_k=CFG.ret_spec_top_k,
            draft_len=CFG.ret_spec_draft_len,
            n_prompts=CFG.ret_spec_n_prompts,
            max_new_tokens=CFG.ret_spec_max_new_tokens,
        )
        pd.DataFrame([result]).to_csv(CFG.ret_spec_csv, index=False)
        print(f"Saved → {CFG.ret_spec_csv}")
    else:
        print(f"Default config results exist — skipping. ({CFG.ret_spec_csv})")

    # ── top_k × draft_len sweep ──────────────────────────────────────────────
    if not CFG.ret_spec_sweep_csv.exists():
        partial_path = CFG.ret_spec_sweep_csv.with_suffix(".partial.csv")
        sweep_rows = []
        done_pairs: set = set()

        if partial_path.exists():
            import pandas as _pd
            existing = _pd.read_csv(partial_path)
            sweep_rows = existing.to_dict("records")
            done_pairs = {(r["top_k"], r["draft_len"]) for r in sweep_rows}
            print(f"Resuming sweep: {len(sweep_rows)} rows already done.")

        total_configs = len(CFG.ret_spec_top_k_sweep) * len(CFG.ret_spec_draft_len_sweep)
        print(f"\n=== Sweep: {total_configs} configs "
              f"(top_k ∈ {CFG.ret_spec_top_k_sweep}, "
              f"draft_len ∈ {CFG.ret_spec_draft_len_sweep}) ===")

        for top_k in CFG.ret_spec_top_k_sweep:
            for draft_len in CFG.ret_spec_draft_len_sweep:
                if (top_k, draft_len) in done_pairs:
                    print(f"  top_k={top_k} draft_len={draft_len} — already done, skipping.")
                    continue

                print(f"\n--- top_k={top_k}  draft_len={draft_len} ---")
                r = evaluate_retrieval_speculative(
                    model, bench_prompts, index_norm, token_seqs,
                    baseline_tps=baseline_tps,
                    top_k=top_k,
                    draft_len=draft_len,
                    n_prompts=CFG.ret_spec_sweep_n_prompts,
                    max_new_tokens=CFG.ret_spec_max_new_tokens,
                )
                sweep_rows.append(r)
                pd.DataFrame(sweep_rows).to_csv(partial_path, index=False)

        pd.DataFrame(sweep_rows).to_csv(CFG.ret_spec_sweep_csv, index=False)
        if partial_path.exists():
            partial_path.unlink()
        print(f"\nSaved sweep → {CFG.ret_spec_sweep_csv}")
    else:
        print(f"Sweep results exist — skipping. ({CFG.ret_spec_sweep_csv})")

    # ── Summary comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Retrieval Speculative Decoding vs baseline")
    print("=" * 70)
    print(f"{'Method':<35} {'TPS':>7} {'Tok/step':>9} {'Draft acc':>10} {'Speedup':>8}")
    print("-" * 70)
    print(f"{'Baseline (full vocab)':<35} {baseline_tps:>7.2f} {'—':>9} {'—':>10} {'1.000x':>8}")

    # old speculative decoding result for comparison
    if CFG.spec_csv.exists():
        import pandas as _pd
        old = _pd.read_csv(CFG.spec_csv)
        for _, row in old.iterrows():
            label = f"Old spec ({row['method']}, D={int(row['draft_len'])})"
            speedup = row["speedup_vs_baseline"]
            print(f"  {label:<33} {row['effective_tps']:>7.2f} {'—':>9} "
                  f"{row['draft_acceptance_rate']:>10.3f} {speedup:>7.3f}x")

    if CFG.ret_spec_csv.exists():
        import pandas as _pd
        df = _pd.read_csv(CFG.ret_spec_csv)
        for _, row in df.iterrows():
            label = f"RetSpec K={int(row['top_k'])} D={int(row['draft_len'])}"
            print(f"  {label:<33} {row['effective_tps']:>7.2f} "
                  f"{row['tokens_per_step']:>9.2f} "
                  f"{row['draft_acceptance_rate']:>10.3f} "
                  f"{row['speedup_vs_baseline']:>7.3f}x")

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
