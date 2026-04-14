"""
Dataset registry and normalisation layer for multi-dataset robustness experiments.

Each dataset is normalised to a common dict:
    {"train": List[str], "test": List[str]}

This shields the rest of the codebase from differences in HuggingFace split names,
column names, and dataset sizes.
"""
from __future__ import annotations

import urllib.request
from typing import Dict, List

from datasets import load_dataset as hf_load_dataset


# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry describes how to fetch and normalise one dataset.
#
# Fields:
#   loader       - "hf" (HuggingFace) or "url" (direct URL download)
#
# For loader="hf":
#   hf_path      - first positional arg to load_dataset()
#   hf_config    - second positional arg (subset name); None if not needed
#   train_split  - HF split name that contains training / index-building text
#   test_split   - HF split name used for evaluation
#   text_col     - column name that holds the raw text
#   max_train    - cap on number of train examples to load (None = no cap)
#   max_test     - cap on number of test examples to load (None = no cap)
#
# For loader="url":
#   train_url    - URL to raw training text (one sentence per line)
#   test_url     - URL to raw test text
#   max_train / max_test — same meaning as above
#
# Common fields:
#   label        - human-readable label used in result tables and plots
#   domain       - short domain tag for display

DATASET_REGISTRY: Dict[str, Dict] = {
    "wikitext2": {
        "loader": "hf",
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
        # HuggingFace's ptb_text_only uses a legacy loading script no longer
        # supported by datasets>=2.x. Load the canonical files directly from
        # the wojzaremba/lstm GitHub mirror (same source the HF script uses).
        "loader": "url",
        "train_url": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
        "test_url":  "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
        "max_train": None,
        "max_test": None,
        "label": "Penn Treebank",
        "domain": "News (WSJ)",
    },
    "ag_news": {
        "loader": "hf",
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
        "loader": "hf",
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


def _load_url_lines(url: str, cap: int | None) -> List[str]:
    """Download a plain-text file and return non-empty lines, optionally capped."""
    print(f"  Fetching {url} …")
    with urllib.request.urlopen(url) as resp:
        lines = resp.read().decode("utf-8").splitlines()
    lines = [l for l in lines if l.strip()]
    if cap is not None:
        lines = lines[:cap]
    return lines


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

    # ── URL-based loader ──────────────────────────────────────────────────────
    if cfg["loader"] == "url":
        train_texts = _load_url_lines(cfg["train_url"], cfg["max_train"])
        test_texts  = _load_url_lines(cfg["test_url"],  cfg["max_test"])
        # PTB lines are already full sentences — no length filter needed,
        # but keep consistent with the >20 char filter used elsewhere.
        train_texts = [t for t in train_texts if len(t.strip()) > 20]
        test_texts  = [t for t in test_texts  if len(t.strip()) > 20]

    # ── HuggingFace loader ────────────────────────────────────────────────────
    else:
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
            if cfg["max_test"] is not None:
                test_texts = test_texts[: cfg["max_test"]]
        else:
            train_texts = _extract(cfg["train_split"], cfg["max_train"])
            test_texts  = _extract(cfg["test_split"],  cfg["max_test"])

    print(f"  Train texts: {len(train_texts):,}")
    print(f"  Test  texts: {len(test_texts):,}")

    return {
        "train": train_texts,
        "test":  test_texts,
        "label": cfg["label"],
        "domain": cfg["domain"],
    }
