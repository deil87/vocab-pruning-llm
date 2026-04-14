"""
Multi-dataset robustness sweep for the Dual-Encoder vocabulary-pruning router.

For each dataset in the registry (WikiText-2, Penn Treebank, AG News, CodeParrot):
  1. Load and normalise the dataset via src/dataset_utils
  2. Build a completion index from the train split
  3. Train a fresh dual-encoder router on that dataset
  4. Evaluate full-vocab baseline PPL
  5. Evaluate pruned PPL and token acceptance rate at K ∈ {5k, 10k}
  6. Record ΔPPL = pruned_ppl - baseline_ppl

Results are written to:
  results/multi_dataset_results.csv   — one row per (dataset, K)
  results/multi_dataset_summary.csv   — one row per dataset (best K only)

Usage:
    python run_multi_dataset.py

    # Run a single dataset (useful for debugging or partial reruns):
    DATASET_NAME=ptb python run_multi_dataset.py

    # Fast mode: smaller index + fewer training pairs (~4x faster, good for sanity checks)
    FAST_MODE=1 python run_multi_dataset.py

Environment variables:
    DATASET_NAME          — run only this dataset key (e.g. ptb, ag_news, codeparrot)
    FAST_MODE             — set to 1 to use reduced index/training sizes
    N_INDEX_COMPLETIONS   — override completion index size (default: CFG.n_index_completions)
    N_TRAIN_PAIRS         — override number of training pairs (default: CFG.n_train_pairs)
    K_VALUES              — comma-separated K values to sweep (default: 5000,10000)
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.config import CFG
from src.dataset_utils import DATASET_REGISTRY, list_datasets, load_normalised_dataset
from src.dual_encoder import (
    RouterMLP,
    build_completion_index,
    project_completion_index,
    train_router,
)
from src.model_utils import DEVICE, load_model

# ── Runtime config helpers ─────────────────────────────────────────────────────
# NOTE: deliberately NOT evaluated at module import time so that notebook cells
# can change FAST_MODE / env vars and then call run_sweep() without needing to
# reload the module.

def _resolve_runtime_config(
    fast_mode=None,
    n_index_completions=None,
    n_train_pairs=None,
    k_sweep=None,
):
    """
    Resolve sweep hyper-parameters from explicit args > env vars > defaults.

    Called at the top of run_sweep() so values are read at *call time*, not at
    module import time.  Explicit keyword arguments always win.
    """
    # fast_mode
    if fast_mode is None:
        fast_mode = os.environ.get("FAST_MODE", "0") not in ("0", "", "false", "False")

    # index size
    if n_index_completions is None:
        _env = os.environ.get("N_INDEX_COMPLETIONS", "")
        n_index_completions = int(_env) if _env else (2_000 if fast_mode else CFG.n_index_completions)

    # train pairs
    if n_train_pairs is None:
        _env = os.environ.get("N_TRAIN_PAIRS", "")
        n_train_pairs = int(_env) if _env else (1_000 if fast_mode else CFG.n_train_pairs)

    # K sweep
    if k_sweep is None:
        _k_env = os.environ.get("K_VALUES", "")
        k_sweep = (
            [int(k) for k in _k_env.split(",") if k.strip()]
            if _k_env
            else [5_000, 10_000]
        )

    return fast_mode, n_index_completions, n_train_pairs, k_sweep

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = CFG.results_dir
MULTI_CSV = RESULTS_DIR / "multi_dataset_results.csv"
SUMMARY_CSV = RESULTS_DIR / "multi_dataset_summary.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _texts_to_tokens(texts, tokenizer, max_tokens: int) -> torch.Tensor:
    """Tokenise a list of strings and return the first max_tokens as a 1D LongTensor."""
    parts = [tokenizer.encode(t, return_tensors="pt")[0] for t in texts]
    return torch.cat(parts)[:max_tokens]


def _full_vocab_ppl(model, tokenizer, test_texts, max_tokens=CFG.ppl_max_tokens) -> float:
    """Full-vocabulary perplexity on the given text list."""
    tokens = _texts_to_tokens(test_texts, tokenizer, max_tokens)
    input_ids = tokens.unsqueeze(0).to(DEVICE)
    stride, seq_len, nlls = 512, input_ids.size(1), []

    with torch.no_grad():
        for begin in tqdm(range(0, seq_len - 1, stride), desc="PPL baseline", leave=False):
            end = min(begin + stride, seq_len - 1)
            chunk = input_ids[:, begin : end + 1]
            out = model(input_ids=chunk)
            logits = out.logits[0, :-1, :]
            targets = chunk[0, 1:]
            nll = F.cross_entropy(logits, targets, reduction="none")
            nlls.extend(nll.float().cpu().tolist())

    ppl = math.exp(np.mean(nlls))
    print(f"  Full-vocab PPL: {ppl:.3f}")
    return ppl


def _pruned_ppl_and_acceptance(
    model,
    tokenizer,
    test_texts,
    router: RouterMLP,
    proj_index: torch.Tensor,          # [N, d_r] projected completion embeddings
    completion_token_lists,
    k: int,
    max_tokens: int = CFG.ppl_max_tokens,
):
    """
    Compute (pruned_ppl, acceptance_rate) for the DE (step) router at shortlist size K.

    At each token position:
      - query  = router(h_T)           [d_r]
      - scores = query @ proj_index.T  [N]
      - top-K completions → union of tokens → shortlist
      - NLL computed over shortlist; penalty -log(1e-9) if target missing
    """
    tokens = _texts_to_tokens(test_texts, tokenizer, max_tokens)
    input_ids = tokens.unsqueeze(0).to(DEVICE)
    stride, seq_len, nlls = 512, input_ids.size(1), []
    accepted_count, total_count = 0, 0

    # Project index lives on CPU (consistent with existing evaluate.py pattern)
    proj_cpu = proj_index.cpu()
    router.eval()

    with torch.no_grad():
        for begin in tqdm(range(0, seq_len - 1, stride), desc=f"PPL K={k}", leave=False):
            end = min(begin + stride, seq_len - 1)
            chunk = input_ids[:, begin : end + 1]
            out = model(input_ids=chunk, output_hidden_states=True)
            hiddens = out.hidden_states[-1][:, :-1, :]   # [1, T, d]
            targets = chunk[:, 1:]                        # [1, T]

            for t in range(hiddens.size(1)):
                h = hiddens[0, t].float().cpu()           # [d]
                query = router(h.unsqueeze(0)).squeeze(0) # [d_r]
                scores = query @ proj_cpu.T               # [N]
                top_completions = scores.topk(CFG.n_retrieve).indices  # [n_retrieve]

                # Union of tokens across top completions → shortlist
                token_sets = [completion_token_lists[i] for i in top_completions.tolist()]
                shortlist = torch.cat(token_sets).unique()

                # Trim / pad to exactly K if needed (union may be slightly over/under)
                if len(shortlist) > k:
                    shortlist = shortlist[:k]

                pruned_w = model.lm_head.weight[shortlist].float().cpu()
                logits = h @ pruned_w.T
                target = targets[0, t].item()

                match = (shortlist == target).nonzero(as_tuple=True)[0]
                if match.numel() > 0:
                    local_idx = match[0].item()
                    nll = -F.log_softmax(logits, dim=-1)[local_idx].item()
                    accepted_count += 1
                else:
                    nll = -math.log(1e-9)
                nlls.append(nll)
                total_count += 1

    ppl = math.exp(np.mean(nlls))
    acceptance = accepted_count / total_count if total_count > 0 else 0.0
    return ppl, acceptance


def _build_raw_dataset_for_dual_encoder(normalised: dict, tokenizer):
    """
    The existing dual_encoder.build_completion_index() and train_router() functions
    expect a HuggingFace DatasetDict with a "train" split containing a "text" column.
    We pass in a minimal shim dict that mimics that interface.
    """
    train_tokens = _texts_to_tokens(
        normalised["train"], tokenizer, max_tokens=10_000_000
    )
    return train_tokens


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(
    dataset_keys=None,
    *,
    fast_mode=None,
    n_index_completions=None,
    n_train_pairs=None,
    k_sweep=None,
):
    """
    Run the full multi-dataset robustness sweep.

    Args:
        dataset_keys:        list of dataset keys to run (default: all registered datasets).
                             Can be overridden by DATASET_NAME env var for single-dataset runs.
        fast_mode:           True → use reduced index/training sizes (overrides env var).
        n_index_completions: completion index size (overrides env var + fast_mode default).
        n_train_pairs:       number of router training pairs (overrides env var + fast_mode default).
        k_sweep:             list of K values to sweep (overrides env var default).

    Config resolution order: explicit kwarg > env var > fast_mode default > CFG default.
    Values are resolved here at *call time* so that notebook cells can change FAST_MODE
    without needing to reload the module.
    """
    # Resolve runtime config at call time (not at import time)
    _fast_mode, _n_index_completions, _n_train_pairs, _k_sweep = _resolve_runtime_config(
        fast_mode=fast_mode,
        n_index_completions=n_index_completions,
        n_train_pairs=n_train_pairs,
        k_sweep=k_sweep,
    )

    # Honour DATASET_NAME env var for targeted single-dataset runs
    env_key = os.environ.get("DATASET_NAME", "")
    if env_key and env_key in DATASET_REGISTRY:
        dataset_keys = [env_key]
        print(f"DATASET_NAME={env_key}: running single-dataset sweep.")
    elif dataset_keys is None:
        dataset_keys = list_datasets()

    print(f"\n{'='*60}")
    print(f"Multi-dataset robustness sweep")
    print(f"Model:         {CFG.model_name}")
    print(f"Datasets:      {dataset_keys}")
    print(f"K values:      {_k_sweep}")
    print(f"Fast mode:     {_fast_mode}")
    print(f"Index size:    {_n_index_completions:,}")
    print(f"Train pairs:   {_n_train_pairs:,}")
    print(f"{'='*60}\n")

    # Load model once — shared across all datasets
    model, tokenizer = load_model()

    all_rows = []

    for ds_key in dataset_keys:
        print(f"\n{'─'*60}")
        print(f"Dataset: {ds_key}")
        print(f"{'─'*60}")

        # ── 1. Load + normalise dataset ───────────────────────────────────────
        normalised = load_normalised_dataset(ds_key)
        label = normalised["label"]
        domain = normalised["domain"]

        # ── 2. Tokenise train split for index/router building ─────────────────
        print("Tokenising train split…")
        train_tokens = _texts_to_tokens(
            normalised["train"], tokenizer, max_tokens=10_000_000
        )
        print(f"  Train tokens: {len(train_tokens):,}")

        # ── 3. Build completion index from this dataset's train split ─────────
        # We call build_completion_index via a shim: it expects raw_dataset["train"]["text"]
        # Build a lightweight namespace to satisfy that interface.
        class _DatasetShim:
            """Minimal shim so build_completion_index can call raw_dataset["train"]["text"]."""
            def __init__(self, train_texts):
                self._train = train_texts

            def __getitem__(self, split):
                if split == "train":
                    return {"text": self._train}
                raise KeyError(split)

        raw_shim = _DatasetShim(normalised["train"])
        # Temporarily patch CFG so build_completion_index uses our size override
        _orig_n_index = CFG.n_index_completions
        CFG.n_index_completions = _n_index_completions
        completion_token_lists, completion_index, _ = build_completion_index(
            model, tokenizer, raw_shim
        )
        CFG.n_index_completions = _orig_n_index

        # ── 4. Train a fresh router on this dataset ───────────────────────────
        _orig_n_pairs = CFG.n_train_pairs
        CFG.n_train_pairs = _n_train_pairs
        router, key_projector = train_router(model, train_tokens)
        CFG.n_train_pairs = _orig_n_pairs
        router.eval()
        key_projector.eval()

        # Project completion index with the freshly trained key_projector
        proj_index = project_completion_index(key_projector, completion_index)

        # ── 5. Full-vocab baseline PPL on this dataset's test split ───────────
        baseline_ppl = _full_vocab_ppl(model, tokenizer, normalised["test"])

        # ── 6. Sweep K values ─────────────────────────────────────────────────
        for k in _k_sweep:
            print(f"\n  K={k:,}")
            pruned_ppl, acceptance = _pruned_ppl_and_acceptance(
                model,
                tokenizer,
                normalised["test"],
                router,
                proj_index,
                completion_token_lists,
                k=k,
            )
            delta_ppl = pruned_ppl - baseline_ppl
            print(f"    Pruned PPL:      {pruned_ppl:.3f}")
            print(f"    ΔPPL:            {delta_ppl:+.3f}")
            print(f"    Acceptance rate: {acceptance:.4f} ({acceptance*100:.1f}%)")

            all_rows.append({
                "dataset_key": ds_key,
                "dataset_label": label,
                "domain": domain,
                "model": CFG.model_name,
                "k": k,
                "baseline_ppl": round(baseline_ppl, 4),
                "pruned_ppl": round(pruned_ppl, 4),
                "delta_ppl": round(delta_ppl, 4),
                "acceptance_rate": round(acceptance, 6),
                "n_index_completions": _n_index_completions,
                "n_train_pairs": _n_train_pairs,
                "fast_mode": _fast_mode,
            })

        # Checkpoint: write after each dataset so partial results are preserved
        df = pd.DataFrame(all_rows)
        df.to_csv(MULTI_CSV, index=False)
        print(f"\n  Partial results written → {MULTI_CSV}")

    # ── 7. Write final CSVs ───────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(MULTI_CSV, index=False)

    # Summary: best K per dataset (lowest ΔPPL)
    summary = (
        df.sort_values("delta_ppl")
        .groupby("dataset_key", sort=False)
        .first()
        .reset_index()
        [["dataset_key", "dataset_label", "domain", "model",
          "k", "baseline_ppl", "pruned_ppl", "delta_ppl", "acceptance_rate"]]
    )
    summary.to_csv(SUMMARY_CSV, index=False)

    # ── 8. Print results table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MULTI-DATASET ROBUSTNESS RESULTS")
    print(f"{'='*60}")
    print(
        df.to_string(
            index=False,
            columns=["dataset_label", "domain", "k", "baseline_ppl",
                     "pruned_ppl", "delta_ppl", "acceptance_rate"],
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print(f"\nFull results → {MULTI_CSV}")
    print(f"Summary      → {SUMMARY_CSV}")

    return df


if __name__ == "__main__":
    run_sweep()
