# Research Discussion Log

## Date: 2026-03-28

---

## Project Overview

**Topic:** Optimisation of inference time for LLMs, specifically via **Dynamic Vocabulary Pruning** with a **Dual-Encoder Router** trained using Multiple Negatives Ranking Loss (MNRL).

---

## Raw Research Articles Reviewed

1. `Optimisations around speculative decoding.md` — Deep-dive conversation covering:
   - How self-attention builds the hidden state ("vibe") matrix
   - Distinction between frozen weights and the temporary KV cache / hidden state
   - Why the full vocabulary is always searched at each step (lm_head bottleneck)
   - Dynamic Vocabulary Pruning / Compression as a concept (DynaSpec, FastMTP)
   - The lightweight router idea: pre-filter vocab using context before the heavy matmul
   - The "blind spot" risk: over-pruning can prevent the model from generating rare/creative tokens

2. `building vibe aware graph or vector space.md` — Deep-dive conversation covering:
   - Dual-Encoder (Siamese / Bi-Encoder) architecture
   - Training process: Anchor (prefix) → Positive (completion), in-batch negatives
   - Contrastive learning with MNRL / InfoNCE loss
   - INSTRUCTOR-style instruction-tuned embeddings
   - How instruction prepending shifts the embedding space at inference time
   - "One model, many tasks" advantage of instruction-tuned dual encoders

3. `nebius token factory.md` — Overview of the Nebius Token Factory platform:
   - Managed inference platform for open-source LLMs (Llama, DeepSeek, Qwen)
   - Uses speculative decoding + KV caching to cut inference costs up to 70%
   - Context: production-scale motivation for inference optimisation research

---

## Research Plan (Agreed)

### Core Thesis

At each decoding step, the LLM's hidden state already encodes enough contextual signal ("vibe") to predict which region of the vocabulary is relevant. A small dual-encoder trained with MNRL on `(prefix hidden state → full sentence)` pairs can learn to map the hidden state to a shortlist of candidate tokens — replacing the brute-force full-vocabulary `lm_head` projection with a fast nearest-neighbour lookup over a pruned candidate set.

---

### Research Questions

| # | Question |
|---|----------|
| Q1 | What fraction of per-token latency is consumed by the `lm_head` projection on MPS/CPU? |
| Q2 | Can a cosine-distance router (no training) serve as a strong baseline for shortlisting? |
| Q3 | Does a dual-encoder trained with MNRL on `(prefix hidden state → token embedding)` outperform the cosine baseline in token acceptance rate and perplexity, **measured relative to the full-vocabulary baseline**? |
| Q4 | What is the accuracy/speed trade-off across shortlist sizes K ∈ {512, 1k, 2k, 5k, 10k}? |
| Q5 | How does cluster-based routing (k-means over the vocab embedding space) compare to the learned dual-encoder? |

---

### Techniques to Benchmark

| Technique | Role |
|-----------|------|
| Baseline | Full-vocab autoregressive decode — reference for all comparisons |
| Static Top-K by norm | Sanity check — context-free shortlist |
| Cosine router | Context-aware, zero-shot, no training needed |
| Cluster-based router | Offline vocab clustering, cheap inference-time lookup |
| Dual-Encoder router (MNRL) | **Main contribution** — trained router mapping hidden state → token shortlist |

---

### Dual-Encoder Training Design

- **Anchor:** prefix (first 40–70% of tokens) from a training sentence; represented as the frozen LLM's final hidden state of the last prefix token
- **Positive:** full sentence, mean-pooled token embeddings
- **Negatives:** in-batch negatives (other completions in the same batch)
- **Tower A:** frozen base LLM hidden state (no gradient through LLM)
- **Tower B:** small MLP head (2–3 layers) projecting mean-pooled token embeddings
- **Loss:** MNRL / InfoNCE
- **Inference:** pre-build a FAISS or brute-force cosine index over all vocab token embeddings; at each step retrieve top-K, run `lm_head` only over that shortlist

---

### Dataset

- **Router training:** `wikitext-2-raw-v1` (public, small, text-only)
- **Evaluation / perplexity:** held-out split of the same dataset
- **Latency benchmark:** fixed 50-prompt slice from WikiText-2 (no creative/predictable distinction for now)

---

### Models

| Role | Model |
|------|-------|
| Main LLM | `meta-llama/Llama-3.2-1B-Instruct` |
| Fallback | `microsoft/Phi-3-mini-4k-instruct` |
| Router encoder | Thin MLP head on top of frozen LLM hidden states |

---

### Hardware

- **Primary:** Apple Silicon Mac (MPS backend)
- **Fallback:** CPU

---

### Metrics

| Metric | Tool |
|--------|------|
| Tokens/sec | `time.perf_counter()`, mean ± std over 50 prompts |
| `lm_head` time % | Manual timer wrapping the projection call |
| Token acceptance rate | % of steps where gold greedy token ∈ shortlist (vs. full-vocab baseline) |
| Perplexity | NLL over held-out WikiText-2 slice, pruned vs. full-vocab baseline |
| Shortlist quality curve | Acceptance rate vs. K (plot) |
| Speed vs. accuracy Pareto | tokens/sec vs. perplexity (plot) |

---

### Notebook Structure (`benchmarks.ipynb`)

```
0. Setup, imports, hardware info
1. Baseline profiling
     └── lm_head time as % of total, tokens/sec
2. Vocab space exploration
     └── PCA/UMAP of token embeddings, cluster structure
3. Static & cosine router baselines
     └── acceptance rate vs. K, perplexity, tokens/sec
4. Cluster-based router
     └── k-means over vocab, same metrics
5. Dual-Encoder router (main contribution)
     ├── 5a. Data prep from WikiText-2
     ├── 5b. Training loop (MNRL loss)
     ├── 5c. Evaluation: acceptance rate vs. K, relative to full-vocab baseline
     └── 5d. Integrated decode benchmark
6. Pareto charts: accuracy vs. speed
7. Discussion, limitations, future work
```

---

### Paper Structure (`paper/research_paper.md`)

```
1. Abstract
2. Introduction
3. Background
     3.1 Self-attention & the hidden state "vibe"
     3.2 The lm_head as computational bottleneck
     3.3 Speculative decoding & related work (Medusa, DynaSpec)
     3.4 Dual-encoders & contrastive learning (MNRL, INSTRUCTOR)
4. Method
     4.1 Problem formalisation
     4.2 Router taxonomy
     4.3 Dual-Encoder router design & training
     4.4 Inference integration
5. Experiments
     5.1 Setup
     5.2 Baseline profiling results
     5.3 Router comparison (all vs. full-vocab baseline)
     5.4 Accuracy / speed trade-off
6. Discussion
     6.1 The blind-spot risk
     6.2 Limitations
7. Conclusion & Future Work
```

---

## Key Design Decisions

1. **Tower B uses the same frozen LLM** to embed completions — conceptually cleaner, keeps the router operating entirely within the LLM's own latent space.
2. **No distinction between predictive and creative prompts** in the benchmark suite — use a single public dataset (WikiText-2) uniformly.
3. All pruning methods evaluated **relative to the full-vocabulary baseline** so accuracy loss is directly interpretable.
4. Dual-encoder router is the **primary research contribution**, not a stretch goal.
