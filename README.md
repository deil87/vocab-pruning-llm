# Context-Aware Vocabulary Pruning for LLM Inference Acceleration

Research and implementation of a **Dual-Encoder Router** that prunes the LLM vocabulary at each
decoding step, reducing the cost of the `lm_head` matrix multiplication without changing model
weights or requiring speculative decoding.

**Paper:** [`paper/research_paper.pdf`](paper/research_paper.pdf)  
**Preprint:** [zenodo.org/records/19565831](https://zenodo.org/records/19565831) — DOI: [10.5281/zenodo.19565831](https://doi.org/10.5281/zenodo.19565831)

---

## How it works

At each decoding step the model's hidden state is used to retrieve semantically plausible sentence
completions from a prebuilt index. The union of tokens across those completions forms a pruned
vocabulary shortlist; the `lm_head` matmul is then performed only over that shortlist.

```
hidden state h_T
      │
      ▼
Dual-Encoder Router  (Tower A: MLP on frozen LLM hidden state)
      │               (Tower B: sentence completion embeddings)
      ▼
Retrieve top-K sentence completions from index
      │
      ▼
Union of tokens across K completions  →  pruned vocabulary shortlist
      │
      ▼
lm_head matmul over shortlist only  (cheaper + faster)
```

---

## Key results

Benchmarked on **Llama-3.2-1B-Instruct** and **Llama-3.1-8B-Instruct**, WikiText-2,
single NVIDIA RTX PRO 4500 Blackwell GPU (32 GB), Triton kernel.

### Baseline

| Model | Tok/s | Step (ms) | `lm_head` fraction | PPL |
|---|---|---|---|---|
| Llama-3.2-1B | 77.75 | 12.86 | 5.17% | 26.26 |
| Llama-3.1-8B | 37.93 | 26.37 | 4.98% | 16.62 |

### DE (step) router — best trade-off operating points

| Model | K | Acceptance | Tok/s | ΔPPL |
|---|---|---|---|---|
| 1B | 5k | 99.4% | 64.62 | +1.70 |
| 1B | 10k | 99.4% | 64.58 | **+1.11** |
| 8B | 5k | 99.5% | 34.13 | +0.44 |
| 8B | 10k | 99.5% | 33.91 | **+0.19** ← near-lossless |

The 8B model at K=10k achieves **+0.19 PPL degradation** (near-lossless quality) at 99.5%
token acceptance. Throughput is ~11% below baseline — the `lm_head` fraction (≈5%) is too
small on current hardware for net speedup; the break-even point is projected at 70B+ scale.

---

## Project layout

```
.
├── src/
│   ├── config.py               # Central config (all hyperparameters)
│   ├── baseline.py             # Full-vocabulary baseline inference
│   ├── dual_encoder.py         # Dual-encoder router (training + inference)
│   ├── routers.py              # Cosine, static, and other router variants
│   ├── router_kernel.py        # Triton kernel for pruned lm_head
│   ├── evaluate.py             # Perplexity and throughput evaluation
│   ├── plots.py                # All figure generation
│   ├── model_utils.py          # Model / tokenizer loading helpers
│   ├── mlp_graph.py            # MLP transition graph router
│   ├── speculative_retrieval.py# Retrieval-based speculative decoding (Path B)
│   └── logging_utils.py
│
├── run_baseline.py             # Step 1 — measure baseline
├── run_dual_encoder.py         # Step 2 — train router + benchmark DE variants
├── run_static.py               # Cosine / static router benchmarks
├── run_cluster.py              # k-means cluster router
├── run_graph.py                # MLP transition graph router
├── run_attention.py            # Attention graph router
├── run_speculative.py          # Standard speculative decoding baseline
├── run_speculative_retrieval.py# Retrieval-based speculative decoding
├── run_plots.py                # Regenerate all figures
├── run_all.py                  # Run the full pipeline end-to-end
│
├── results/
│   ├── results_1b_gpu_kernel/  # 1B Blackwell + Triton results & plots
│   └── results_8b_gpu_kernel/  # 8B Blackwell + Triton results & plots
│
├── paper/
│   └── research_paper.pdf      # Compiled paper (pdflatex)
│
├── benchmarks.ipynb            # Interactive exploration notebook
├── runpod_run_1b.ipynb         # RunPod notebook — 1B experiments
├── runpod_run_8b.ipynb         # RunPod notebook — 8B experiments
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model access

The experiments use gated Llama models from Hugging Face. You need to:

1. Accept the Meta Llama license at
   [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
   and
   [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
2. Log in via the CLI:

```bash
huggingface-cli login
```

### 3. (Optional) Triton kernel

The Triton-accelerated `lm_head` kernel (`src/router_kernel.py`) requires a CUDA GPU.
It is compiled on first use. PyTorch fallback is used automatically on CPU or when Triton
is unavailable.

---

## Reproducing results

All scripts write CSVs and PNGs into a `results/` subdirectory. Use the `MODEL_NAME`
environment variable to switch between models.

### 1B model (Llama-3.2-1B-Instruct)

```bash
# Baseline throughput + lm_head profiling
python run_baseline.py

# Train dual-encoder router and run all DE benchmarks
python run_dual_encoder.py

# Static / cosine router benchmarks
python run_static.py

# k-means cluster router
python run_cluster.py

# MLP transition graph router
python run_graph.py

# Regenerate all figures
python run_plots.py
```

### 8B model (Llama-3.1-8B-Instruct)

```bash
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct python run_baseline.py
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct python run_dual_encoder.py
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct python run_plots.py
```

### Full pipeline (all routers, one model)

```bash
python run_all.py                                          # 1B
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct python run_all.py  # 8B
```

### Speculative decoding baselines

```bash
# Standard (chain-based) speculative decoding
python run_speculative.py

# Retrieval-based speculative decoding (Path B)
python run_speculative_retrieval.py
```

### Key config knobs (`src/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `k_values` | `[512, 1k, 2k, 5k, 10k]` | Shortlist sizes to sweep |
| `n_bench_prompts` | 50 | Prompts for throughput benchmarks |
| `n_index_completions` | 8000 | Completions in the retrieval index |
| `n_retrieve` | 32 | Completions retrieved per step |
| `refresh_every` | 16 | Steps between router index refreshes |

---

## Research paths

### Path A — Vocabulary Pruning (implemented)

Use retrieved completions to build a pruned vocabulary shortlist at each decoding step.
Run `lm_head` only over that shortlist. Evaluated against: static top-K, cosine similarity,
k-means cluster, MLP transition graph, and attention graph routers.

- **Metric:** tokens/sec, perplexity degradation, token acceptance rate vs. full-vocab baseline
- **Finding:** DE (step) at K=10k achieves 99.5% acceptance and +0.19 PPL on 8B (near-lossless).
  Net throughput is currently ~11% below baseline due to the small `lm_head` fraction at 1B/8B
  scale; break-even is projected at ~70B.

### Path B — Speculative Decoding via Retrieval (prototype)

Feed retrieved sentence completions directly to the model as draft candidates, replacing the
small draft model used in standard speculative decoding.

- Retrieved completions are semantically coherent multi-token sequences grounded in real text
- Stronger prior than a blind small draft model
- **Metric:** acceptance rate vs. standard speculative decoding, tokens/sec vs. baseline

---

## Building the paper

Requires Docker with the `texlive/texlive` image.

```bash
docker run --rm -v "$(pwd):/workspace" texlive/texlive:latest \
  bash -c "cd /workspace/paper && \
           pdflatex -interaction=nonstopmode research_paper.tex && \
           pdflatex -interaction=nonstopmode research_paper.tex"
```

Output: `paper/research_paper.pdf`

---

## Citation

```bibtex
@misc{spiridonov2026vocabpruning,
  title     = {Context-Aware Vocabulary Pruning for {LLM} Inference Acceleration},
  author    = {Andrey Spiridonov},
  year      = {2026},
  doi       = {10.5281/zenodo.19565831},
  url       = {https://zenodo.org/records/19565831},
  publisher = {Zenodo}
}
```

---

## License

Code: [Apache 2.0](LICENSE).  
Paper (`paper/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Andrey Spiridonov.
