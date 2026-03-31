"""Central configuration for all experiments."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # ── Model / data ──────────────────────────────────────────────────────────
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    seed: int = 42

    # ── Benchmark ─────────────────────────────────────────────────────────────
    k_values: List[int] = field(default_factory=lambda: [512, 1_000, 2_000, 5_000, 10_000])
    n_bench_prompts: int = 50
    bench_max_new_tokens: int = 64
    ppl_max_tokens: int = 4096
    prompt_len: int = 128          # tokens per bench prompt

    # ── Router training ───────────────────────────────────────────────────────
    router_train_epochs: int = 3
    router_batch_size: int = 256
    router_lr: float = 3e-4
    router_hidden_dim: int = 512
    router_out_dim: int = 256
    temperature: float = 0.05

    # ── Dual-encoder (completion-based) ──────────────────────────────────────
    completion_len: int = 16
    n_index_completions: int = 8_000
    n_train_pairs: int = 5_000
    n_retrieve: int = 32
    refresh_every: int = 16
    prefix_len_cap: int = 64       # max prefix tokens fed to LLM during training

    # ── k-means cluster router ────────────────────────────────────────────────
    n_kmeans_clusters: int = 256

    # ── Attention graph router ────────────────────────────────────────────────
    attn_top_positions: int = 16
    attn_neighbour_k: int = 64

    # ── MLP transition graph ──────────────────────────────────────────────────
    graph_edges_per_node: int = 32
    graph_anchor_k: int = 16

    # ── Speculative decoding (original chain-based) ───────────────────────────
    spec_draft_len: int = 4
    spec_n_prompts: int = 50
    spec_max_new_tokens: int = 64
    spec_draft_len_sweep: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # ── Retrieval-based speculative decoding ──────────────────────────────────
    ret_spec_completion_len: int = 8      # tokens per indexed completion
    ret_spec_n_completions: int = 50_000  # size of the completion index
    ret_spec_top_k: int = 8               # candidates retrieved per step (default)
    ret_spec_draft_len: int = 8           # draft tokens attempted per candidate (default)
    ret_spec_n_prompts: int = 50
    ret_spec_max_new_tokens: int = 64
    ret_spec_top_k_sweep: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    ret_spec_draft_len_sweep: List[int] = field(default_factory=lambda: [4, 8, 16])
    ret_spec_sweep_n_prompts: int = 20   # fewer prompts for the sweep to keep runtime down

    # ── Paths ─────────────────────────────────────────────────────────────────
    results_dir: Path = Path("results")

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(exist_ok=True)

    # Derived paths
    @property
    def baseline_csv(self) -> Path:
        return self.results_dir / "baseline_summary.csv"

    @property
    def static_csv(self) -> Path:
        return self.results_dir / "static_router_results.csv"

    @property
    def cosine_csv(self) -> Path:
        return self.results_dir / "cosine_router_results.csv"

    @property
    def cluster_csv(self) -> Path:
        return self.results_dir / "cluster_router_results.csv"

    @property
    def dual_enc_csv(self) -> Path:
        return self.results_dir / "dual_encoder_results.csv"

    @property
    def prefetch_csv(self) -> Path:
        return self.results_dir / "dual_encoder_prefetch_results.csv"

    @property
    def shortlist_sizes_npy(self) -> Path:
        return self.results_dir / "shortlist_sizes.npy"

    @property
    def attn_csv(self) -> Path:
        return self.results_dir / "attn_router_results.csv"

    @property
    def graph_csv(self) -> Path:
        return self.results_dir / "graph_router_results.csv"

    @property
    def spec_csv(self) -> Path:
        return self.results_dir / "speculative_decoding_results.csv"

    @property
    def spec_sweep_csv(self) -> Path:
        return self.results_dir / "spec_draft_len_sweep.csv"

    @property
    def ret_spec_index_path(self) -> Path:
        return self.results_dir / "ret_spec_completion_index.pt"

    @property
    def ret_spec_csv(self) -> Path:
        return self.results_dir / "ret_spec_results.csv"

    @property
    def ret_spec_sweep_csv(self) -> Path:
        return self.results_dir / "ret_spec_sweep.csv"

    @property
    def all_results_csv(self) -> Path:
        return self.results_dir / "all_results.csv"

    @property
    def router_checkpoint(self) -> Path:
        return self.results_dir / "router_checkpoint.pt"

    @property
    def graph_path(self) -> Path:
        return self.results_dir / "mlp_transition_graph.pt"


# Singleton used by all scripts
CFG = Config()
