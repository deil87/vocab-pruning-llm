"""
Dataset registry and normalisation layer for multi-dataset robustness experiments.

Each dataset is normalised to a common dict:
    {"train": List[str], "test": List[str]}

This shields the rest of the codebase from differences in HuggingFace split names,
column names, and dataset sizes.
"""
from __future__ import annotations

from typing import Dict, List

from datasets import load_dataset as hf_load_dataset


# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry describes how to fetch and normalise one HuggingFace dataset.
#
# Fields:
#   hf_path      - first positional arg to load_dataset()
#   hf_config    - second positional arg (subset name); None if not needed
#   train_split  - HF split name that contains training / index-building text
#   test_split   - HF split name used for evaluation
#   text_col     - column name that holds the raw text
#   max_train    - cap on number of train examples to load (None = no cap)
#   max_test     - cap on number of test examples to load (None = no cap)
#   label        - human-readable label used in result tables and plots
#   domain       - short domain tag for display

DATASET_REGISTRY: Dict[str, Dict] = {
    "wikitext2": {
        "hf_path": "wikitext",
        "hf_config": "wikitext-2-raw-v1",
        "train_split": "train",
        "test_split": "test",
        "text_col": "text",
        "max_train": None,
        "max_test": None,
        "label": "WikiText-2",
        "domain": "Wikipedia",
    },
    "ptb": {
        "hf_path": "ptb-text-only",
        "hf_config": "penn_treebank",
        "train_split": "train",
        "test_split": "test",
        "text_col": "sentence",
        "max_train": None,
        "max_test": None,
        "label": "Penn Treebank",
        "domain": "News (WSJ)",
    },
    "ag_news": {
        "hf_path": "ag_news",
        "hf_config": None,
        "train_split": "train",
        "test_split": "test",
        "text_col": "text",
        "max_train": 20_000,   # full train is 120k — cap for speed
        "max_test": 2_000,
        "label": "AG News",
        "domain": "News headlines",
    },
    "codeparrot": {
        "hf_path": "codeparrot/codeparrot-clean-valid",
        "hf_config": None,
        "train_split": "train",   # this dataset only has a "train" split
        "test_split": "train",    # we partition manually below
        "text_col": "content",
        "max_train": 5_000,
        "max_test": 500,
        "label": "CodeParrot",
        "domain": "Python code",
    },
}


def list_datasets() -> List[str]:
    """Return all registered dataset keys."""
    return list(DATASET_REGISTRY.keys())


def load_normalised_dataset(key: str) -> Dict[str, List[str]]:
    """
    Load and normalise a dataset from the registry.

    Returns:
        {"train": List[str], "test": List[str], "label": str, "domain": str}

    Each list contains raw text strings, filtered to lines with >20
    non-whitespace characters (consistent with the existing codebase).
    """
    if key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{key}'. Available: {list(DATASET_REGISTRY.keys())}"
        )

    cfg = DATASET_REGISTRY[key]
    print(f"\nLoading dataset: {cfg['label']} ({cfg['domain']})")

    load_kwargs: Dict = {}
    if cfg["hf_config"] is not None:
        ds = hf_load_dataset(cfg["hf_path"], cfg["hf_config"])
    else:
        ds = hf_load_dataset(cfg["hf_path"])

    text_col = cfg["text_col"]

    def _extract(split_name: str, cap: int | None) -> List[str]:
        rows = ds[split_name][text_col]
        if cap is not None:
            rows = rows[:cap]
        return [r for r in rows if isinstance(r, str) and len(r.strip()) > 20]

    # CodeParrot has one split — use first 80% for train, last 20% for test
    if cfg["train_split"] == cfg["test_split"]:
        all_rows = _extract(cfg["train_split"], cfg["max_train"])
        split_idx = int(len(all_rows) * 0.8)
        train_texts = all_rows[:split_idx]
        test_texts = all_rows[split_idx:]
        # honour max_test cap
        if cfg["max_test"] is not None:
            test_texts = test_texts[: cfg["max_test"]]
    else:
        train_texts = _extract(cfg["train_split"], cfg["max_train"])
        test_texts = _extract(cfg["test_split"], cfg["max_test"])

    print(f"  Train texts: {len(train_texts):,}")
    print(f"  Test  texts: {len(test_texts):,}")

    return {
        "train": train_texts,
        "test": test_texts,
        "label": cfg["label"],
        "domain": cfg["domain"],
    }
