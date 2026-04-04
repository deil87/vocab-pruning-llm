# Context-Aware Vocabulary Pruning for LLM Inference Acceleration: A Dual-Encoder Retrieval Approach

**Status:** 1B + 8B GPU benchmarks complete — April 2026

---

## Abstract

Autoregressive decoding in large language models (LLMs) is dominated by two costs at each token step: KV-cache attention and the `lm_head` vocabulary projection — a matrix multiply that reads the full `[V × d]` weight matrix from GPU memory on every step. At batch size 1, a regime typical of interactive and agentic inference, this operation is **memory-bandwidth-bound rather than compute-bound**: the GPU's arithmetic units sit idle while 525 MB of weights are streamed from VRAM per token. We address this bottleneck through **dynamic vocabulary pruning**, a decoding-level inference acceleration technique that restricts the `lm_head` projection to a context-dependent shortlist of K candidate tokens at each step, reducing memory traffic by up to 99.6%.

We propose a **Dual-Encoder Router** — a lightweight MLP (0.06% of model parameters) trained with Multiple Negatives Ranking Loss (MNRL) to map the model's current hidden state to a query vector in a learned space, then retrieve the top-K token candidates by nearest-neighbour search over a pre-indexed corpus of sentence completions. We benchmark four routing strategies against a full-vocabulary baseline on WikiText-2 (`Llama-3.2-1B-Instruct`), sweeping shortlist sizes K ∈ {512, 1k, 2k, 5k, 10k}. We additionally evaluate on `Llama-3.1-8B-Instruct` across two T4 GPUs (`device_map="auto"`). On a single T4, the `lm_head` accounts for 6.1% of per-token latency for the 1B model; on the sharded 8B model this fraction drops counter-intuitively to 4.69%, explained by cross-GPU communication overhead inflating total step time. The primary demonstrated gain is a 92–99.6% reduction in `lm_head` memory bandwidth. Wall-clock speedup requires GPU-native kernels or larger models (70B+) where the memory-bound regime is more pronounced.

---

## 1. Introduction

### 1.1 The Autoregressive Bottleneck

Modern LLMs based on the transformer architecture generate text one token at a time. Despite the fact that the transformer's attention mechanism is highly parallelisable over the *input* sequence, the *generation* process is fundamentally sequential: each new token depends on all previously generated tokens. This autoregressive loop creates a hard bottleneck on inference throughput.

Within each step of this loop, the two most expensive operations are:

1. **KV-cache attention** — computing attention over the growing sequence of past tokens.
2. **Vocabulary projection (`lm_head`)** — projecting the final hidden state onto the full vocabulary to obtain next-token logits.

While KV-cache optimisation has received substantial research attention (flash attention, paged attention, speculative decoding), the `lm_head` projection is comparatively understudied as a target for optimisation. At batch size 1 — the regime of interactive and agentic inference — the `lm_head` matrix multiplication is memory-bandwidth-bound: the arithmetic intensity is too low to keep the GPU's compute units saturated, so the bottleneck is the rate at which the weight matrix can be read from VRAM.

### 1.2 The Wasted Computation Insight

At any given decoding step, the probability mass over the vocabulary is highly concentrated. In predictable contexts (e.g., "The Eiffel Tower is located in...") the top-1 token may carry >99% probability. The model still computes logits for all 32k–128k vocabulary entries.

This observation motivates a natural question: **can the model's own internal representation predict which vocabulary region is relevant before the full projection is run?**

### 1.3 Contributions

This paper makes the following contributions:

1. **Profiling:** We quantify the `lm_head` fraction of per-token latency for open-source LLMs on NVIDIA T4 GPU(s).
2. **Router taxonomy:** We define and implement four vocabulary routing strategies of increasing sophistication.
3. **Dual-Encoder Router:** We propose a Dual-Encoder trained with MNRL that learns to map the LLM's hidden state to a compact token shortlist, operating entirely within the model's own latent space.
4. **Benchmark:** We evaluate all methods against a full-vocabulary baseline on WikiText-2, reporting token acceptance rate, perplexity, and tokens/sec across shortlist sizes K ∈ {512, 1k, 2k, 5k, 10k}.
5. **Scale evaluation:** The full router suite (excluding cosine) re-evaluated on `Llama-3.1-8B-Instruct` on two T4 GPUs (`device_map="auto"`), quantifying how overhead and quality scale with model size.

---

## 2. Background

### 2.1 Self-Attention and the Hidden State

A transformer processes an input sequence by stacking each token's embedding into a matrix (the hidden state). At each layer, the self-attention mechanism allows every token's representation to incorporate information from all preceding tokens, weighted by learned relevance scores.

By the time the sequence reaches the final layer, the last row of the hidden state matrix — corresponding to the most recently generated token — has absorbed the mathematical influence of every prior token. This dense vector is what we refer to as the **"vibe"**: a single point in a high-dimensional space that summarises the full context.

Formally, given an input sequence of $T$ tokens, the hidden state at layer $L$ is a matrix $H^{(L)} \in \mathbb{R}^{T \times d}$, where $d$ is the model dimension. The prediction of the next token is made entirely from the last row $h_T = H^{(L)}_{T,:} \in \mathbb{R}^d$.

### 2.2 The `lm_head` as a Computational Bottleneck

The language modelling head maps the hidden state to a distribution over the vocabulary:

$$\text{logits} = h_T W_V^\top \in \mathbb{R}^{|V|}$$

where $W_V \in \mathbb{R}^{|V| \times d}$ is the vocabulary embedding matrix and $|V|$ is the vocabulary size (128,256 for both Llama-3.2-1B and Llama-3.1-8B). At 16-bit precision, $W_V$ occupies 525 MB (1B) or 1,049 MB (8B). Reading this per token step creates a hard ceiling on achievable tokens/sec independent of model depth.

### 2.3 Speculative Decoding and Related Work

**Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) accelerates autoregressive generation by interposing a small "draft" model that proposes several candidate tokens. A large "verifier" model evaluates the entire candidate sequence in a single parallel forward pass. Extensions include **Medusa** (Cai et al., 2024) and tree-based speculative decoding. Our approach is orthogonal: it targets the `lm_head` projection cost directly and can be combined with any of the above methods.

### 2.4 Dual-Encoders and Contrastive Learning

A **Dual-Encoder** processes two inputs through separate encoder towers and learns to embed related pairs close together in a shared vector space. **Multiple Negatives Ranking Loss (MNRL)** (Henderson et al., 2017; Karpukhin et al., 2020) is the standard training objective, using all other positives in the batch as in-batch negatives.

---

## 3. Method

### 3.1 Problem Formalisation

Let $f_\theta$ be a frozen autoregressive LLM with vocabulary $V$, model dimension $d$, and vocabulary projection matrix $W_V$. At decoding step $t$:

$$\hat{y}_t = \arg\max_{v \in V} (h_t W_V^\top)_v$$

**Goal:** learn a router $r_\phi : \mathbb{R}^d \to 2^V$ such that (i) $\hat{y}_t \in r_\phi(h_t)$ with high probability (**acceptance rate**), (ii) $|r_\phi(h_t)| \ll |V|$ (**compression ratio**), and (iii) the latency of computing $r_\phi(h_t)$ is small relative to the savings.

### 3.2 Router Taxonomy

| Router | Description | Requires Training |
|--------|-------------|-------------------|
| **Static Top-K** | Select K tokens with highest $\|W_V[v,:]\|_2$ (context-free) | No |
| **Cosine Router** | Select K tokens with highest $\cos(h_t, W_V[v,:])$ | No |
| **Cluster Router** | Pre-cluster $W_V$ with k-means; select top-C clusters by $\cos(h_t, \mu_c)$ | Offline only |
| **Dual-Encoder Router** | Learned MLP that maps $h_t$ to a query vector; select K tokens by ANN search | Yes |

### 3.3 Dual-Encoder Router Design

#### Architecture

A lightweight **MLP projection head** $g_\phi : \mathbb{R}^d \to \mathbb{R}^{d_r}$ trained to map the frozen LLM's hidden state to a query vector in a shared embedding space. The vocabulary token embeddings $W_V$ serve as the corpus searched at inference time. Both query and keys live in the LLM's own latent space; no cross-model alignment is required.

#### Training Data Construction

From WikiText-2 training split: (1) tokenise each sentence; (2) for each sentence of length $T \geq 8$, sample a split point $s \sim \text{Uniform}(\lfloor 0.4T \rfloor, \lfloor 0.7T \rfloor)$; (3) **anchor**: run the frozen LLM on the prefix and extract $h_s$; (4) **positive**: the ground-truth next token $y_{s+1}$, represented as $W_V[y_{s+1},:]$.

#### Training Objective

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\cos(g_\phi(h_i), W_V[y_i,:]) / \tau)}{\sum_{j=1}^B \exp(\cos(g_\phi(h_i), W_V[y_j,:]) / \tau)}$$

with in-batch negatives (all $y_j \neq y_i$ in the batch).

#### Inference Integration

1. Pre-compute and cache all rows of $W_V$ normalised to unit length.
2. At step $t$: compute $q_t = g_\phi(h_t)$; retrieve the K token indices with highest $\cos(q_t, W_V[v,:])$ via brute-force matmul.
3. Slice $W_V$ to the shortlist and compute logits only over those K rows.

---

## 4. Experimental Setup

### 4.1 Models

**1B model:** `meta-llama/Llama-3.2-1B-Instruct` — 1B parameters, 2,048-dim hidden state, 128,256-token vocabulary, 16 transformer layers, bfloat16.

**8B model:** `meta-llama/Llama-3.1-8B-Instruct` — 8B parameters, 4,096-dim hidden state, 128,256-token vocabulary, 32 transformer layers, bfloat16. Loaded with `device_map="auto"` across two GPUs (layers 0–17 on cuda:0, 18–31 on cuda:1).

### 4.2 Dataset

- **WikiText-2-raw-v1** (Merity et al., 2016). Standard language modelling benchmark; plain English prose.
- **Router training:** `train` split (~2M tokens).
- **Perplexity evaluation:** `test` split (~240k tokens).
- **Latency benchmark:** 50 randomly sampled contiguous passages of 128 tokens from the `validation` split.

### 4.3 Hardware

**1B experiments:** NVIDIA T4 GPU (16 GB VRAM), single GPU, CUDA backend, bfloat16.

**8B experiments:** Two NVIDIA T4 GPUs (2×16 GB VRAM), `device_map="auto"` sharding via HuggingFace Accelerate, CUDA backend, bfloat16.

All latency measurements taken after a 5-step warm-up; reported as mean over 50 prompts.

### 4.4 Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **Token acceptance rate** | Fraction of decoding steps where $\hat{y}_t^{\text{full}} \in \text{shortlist}_t$ |
| **Perplexity** | $\exp\bigl(-\frac{1}{N}\sum_t \log P_\theta(y_t \mid y_{<t})\bigr)$, computed over `test` split |
| **Perplexity degradation** | $\Delta\text{PPL} = \text{PPL}_{\text{pruned}} - \text{PPL}_{\text{full}}$ |
| **Tokens/sec** | Wall-clock generation speed; mean over 50 prompts |
| **`lm_head` time %** | Fraction of per-token time spent in the vocabulary projection |

---

## 5. Results

All 1B experiments: NVIDIA T4 GPU (single), CUDA, bfloat16, `Llama-3.2-1B-Instruct`. All 8B experiments: two NVIDIA T4 GPUs, `device_map="auto"`, `Llama-3.1-8B-Instruct`. Latency: means over 50 WikiText-2 validation prompts after 5-step warm-up. Perplexity: full WikiText-2 test split.

### 5.1 Baseline Profiling

| Metric | 1B (T4) | 8B (T4×2) |
|--------|---------|-----------|
| Tokens/sec (full vocab) | **26.60** | **11.48** |
| Mean step time | 37.60 ms | 87.14 ms |
| `lm_head` time (mean) | 2.30 ms | 4.08 ms |
| `lm_head` time % | **6.1%** | **4.69%** |
| Baseline PPL (WikiText-2 test) | **26.24** | **16.63** |
| `lm_head` weight matrix | 525 MB | 1,049 MB |

The `lm_head` accounts for 6.1% of per-token latency on the 1B model and 4.69% on the 8B model. Counter-intuitively, the fraction *decreases* with model size: on the T4×2 setup, `device_map="auto"` shards the 8B model across two GPUs, introducing cross-GPU communication overhead that inflates total step time while `lm_head` time grows more modestly. Any router whose overhead exceeds these absolute values will make pruning net-negative (see Section 6.3).

![Baseline decode step profiling: per-operation time breakdown.](../results/baseline_profile.png)

### 5.2 Token Acceptance Rate vs. K

**1B model (T4, single GPU):**

| Router | K=512 | K=1k | K=2k | K=5k | K=10k |
|--------|------:|-----:|-----:|-----:|------:|
| Static Top-K | 0.09% | 0.25% | 0.50% | 2.5% | 7.7% |
| Cluster | 33.8% | 54.8% | 68.1% | 76.6% | 83.3% |
| Cosine | 98.4% | 98.4% | 98.4% | 98.4% | 98.4% |
| Dual-Encoder (step) | 99.5% | 99.5% | 99.5% | 99.5% | 99.5% |
| MLP Graph | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| Attention Graph | 82.7% | 99.8% | 99.8% | 99.8% | 99.8% |

**8B model (T4×2):** Cosine not evaluated (N/A†); Attention Graph not evaluated (prohibitive overhead on T4×2).

| Router | K=512 | K=1k | K=2k | K=5k | K=10k |
|--------|------:|-----:|-----:|-----:|------:|
| Static Top-K | 3.9% | 4.0% | 4.4% | 5.8% | 8.4% |
| Cluster | 44.1% | 60.2% | 66.1% | 76.4% | 85.2% |
| Cosine | N/A† | N/A† | N/A† | N/A† | N/A† |
| Dual-Encoder (step) | 99.8% | 99.8% | 99.8% | 99.8% | 99.8% |
| DE (prefetch+refresh) | — | — | — | 87.9% | — |
| MLP Graph | 99.8% | 99.8% | 99.8% | 99.8% | 99.8% |

†Cosine router PPL evaluation failed silently on T4×2 (`lm_head` on cuda:1); no result written.

**Key observations (1B):** Static Top-K is near-random (<8% at K=10k). Cluster saturates at 83%. Cosine acceptance is flat at 98.4% across all K values. Dual-Encoder (step) and MLP Graph achieve 99.5% and ≥99.9% respectively, essentially independent of K above 512.

**Key observations (8B):** DE (step) and MLP Graph transfer perfectly, both achieving **99.8%** acceptance at all K. Cluster improves to 85.2% at K=10k (vs. 83.3% on 1B). Static Top-K remains near-random.

![Token acceptance rate vs. shortlist size K for all routers.](../results/acceptance_rate_vs_k.png)

### 5.3 Perplexity Degradation vs. K

**1B model** (baseline PPL = 26.24):

| Router | K=512 | K=1k | K=2k | K=5k | K=10k |
|--------|------:|-----:|-----:|-----:|------:|
| Static Top-K | +945,974,736 | +891,715,212 | +804,467,165 | +382,485,536 | +125,028,282 |
| Cluster | +19,513,799 | +370,921 | +14,841 | +3,060 | +834 |
| Cosine | +18.998 | +10.834 | +6.790 | +3.541 | **+2.726** |
| Dual-Encoder (step) | +24.641 | +8.945 | +3.926 | +1.684 | **+1.030** |
| DE (prefetch+refresh)† | — | — | — | +146.153 | — |
| MLP Graph | +39.594 | +13.225 | +6.391 | +2.891 | **+2.105** |
| Attention Graph | +15,025 | +21.580 | +5.908 | +2.104 | **+1.327** |

**8B model** (baseline PPL = 16.63; Static Top-K and Cluster omitted — extreme values ≫10³; Cosine N/A†):

| Router | K=512 | K=1k | K=2k | K=5k | K=10k |
|--------|------:|-----:|-----:|-----:|------:|
| Dual-Encoder (step) | +6.63 | +2.09 | +1.15 | +0.55 | **+0.29** |
| DE (prefetch+refresh)† | — | — | — | +122.85 | — |
| MLP Graph | +10.82 | +3.57 | +2.10 | +1.20 | **+0.82** |

†Effective K≈4,634 (1B) / 4,112 (8B) averaged over the evaluation.

**Key observations:** At K=10k on 1B, the best methods cluster closely: DE step (+1.03), Attention Graph (+1.33), MLP Graph (+2.11), Cosine (+2.73). On the 8B model, quality improves significantly at the same K: DE (step) at K=10k reaches ΔPPL=**+0.29** (vs. +1.03 on 1B), and MLP Graph reaches **+0.82** (vs. +2.11). This improvement likely reflects the higher-capacity 8B model producing a richer hidden state that the router can use more precisely.

![Perplexity degradation (ΔPPL) vs. shortlist size K. Static Top-K and Cluster are omitted from this plot due to extreme values.](../results/ppl_degradation_vs_k.png)

### 5.4 Tokens/sec vs. K

**1B model** (baseline = 26.60 tok/s, single T4):

| Router | K=512 | K=1k | K=2k | K=5k | K=10k | vs baseline |
|--------|------:|-----:|-----:|-----:|------:|------------:|
| **Baseline** | — | — | — | — | 26.60 | — |
| Static Top-K | 26.07 | 26.40 | 26.36 | 26.82 | 26.70 | −1% to +1% |
| Cluster | 24.34 | 24.18 | 24.25 | 24.27 | 23.99 | −9% to −10% |
| Cosine | 24.27 | 24.17 | 23.77 | 23.51 | 23.72 | −9% to −12% |
| DE (prefetch+refresh) | — | — | — | **27.67** | — | **+4%** |
| Dual-Encoder (step) | 23.27 | 22.97 | 23.20 | 23.16 | 22.39 | −13% to −16% |
| MLP Graph | 24.28 | 24.31 | 23.81 | 22.89 | 22.83 | −14% to −16% |
| Attention Graph | 22.72 | 20.71 | 21.30 | 20.12 | 19.98 | −15% to −25% |

**8B model** (baseline = 11.48 tok/s, T4×2; Attention Graph not evaluated):

| Router | K=512 | K=1k | K=2k | K=5k | K=10k | vs baseline |
|--------|------:|-----:|-----:|-----:|------:|------------:|
| **Baseline** | — | — | — | — | 11.48 | — |
| Static Top-K | 11.31 | 11.33 | 11.32 | 11.28 | 11.21 | −1% to −2% |
| Cluster | 11.16 | 11.16 | 11.15 | 11.11 | 11.04 | −3% to −4% |
| DE (prefetch+refresh) | — | — | — | 11.26 | — | −2% |
| Dual-Encoder (step) | 10.17 | 10.15 | 10.10 | 10.00 | 9.89 | −11% to −14% |
| MLP Graph | 10.23 | 10.22 | 10.17 | 10.06 | 9.94 | −11% to −13% |

On the 1B model, the DE (prefetch+refresh) variant is the only pruned method to exceed baseline throughput (+4%, 27.67 tok/s), by amortising router cost across many decoding steps — though at a severe quality penalty (+146 PPL). On the 8B model, the overhead gap narrows compared to 1B: DE (step) costs −11% to −14% (vs. −13% to −16% on 1B), consistent with the break-even analysis in Section 6.3.

![Wall-clock tokens/sec vs. shortlist size K. Dashed line = full-vocab baseline.](../results/throughput_vs_k.png)

### 5.5 Pareto Frontier: Quality vs. Speed

**1B model (T4):**

| Method | K | Tok/s | ΔPPL | Note |
|--------|--:|------:|-----:|------|
| DE (prefetch) | ~4,634 | **27.67** | +146.2 | Fastest; quality unusable |
| Cluster | 10,000 | 23.99 | +833.9 | Quality unusable |
| Cosine | 10,000 | 23.72 | +2.73 | Good quality |
| **DE (step)** | **5,000** | **23.16** | **+1.68** | **Best quality–speed trade-off** |
| DE (step) | 10,000 | 22.39 | +1.03 | Marginally better PPL |

On the 1B model, DE (step) at K=5,000 is the recommended operating point: 23.16 tok/s (−13% vs. baseline) with ΔPPL=+1.68 — 96.1% memory bandwidth reduction at minimal quality cost.

On the 8B model, no pruned method exceeds the 11.48 tok/s baseline. The best quality-preserving option is DE (step) at K=5,000: 10.00 tok/s (−13% vs. baseline) with ΔPPL=+0.55 — 96.1% bandwidth reduction and near-baseline quality.

![Pareto frontier: throughput vs. perplexity degradation.](../results/pareto_frontier.png)

![Mean shortlist size growth vs. K (Dual-Encoder prefetch+refresh variant).](../results/shortlist_growth.png)

### 5.6 Graph-Based Routers

**MLP Transition Graph:** A directed graph over vocabulary tokens where edge weights encode how frequently token $v$ follows token $u$ in training. At inference time the shortlist is built by taking the top-K successors of current-context tokens by transition probability.

- **1B:** Acceptance ≥99.9%; throughput 22.83–24.31 tok/s (−14% to −16%); PPL at K=10k: +2.11.
- **8B:** Acceptance 99.8%; throughput 9.94–10.23 tok/s (−11% to −13%); PPL at K=10k: **+0.82** — notably better quality than on 1B.

**Attention Graph:** Uses the model's own attention patterns to identify which previously seen tokens the model attends to at step $t$, then retrieves their vocabulary neighbours.

- **1B:** Acceptance at K≥1k: 99.8%; throughput 19.98–22.72 tok/s (−15% to −25%); PPL at K=10k: +1.33 — second-best quality.
- **8B:** Not evaluated (prohibitive overhead on T4×2 with `output_attentions=True` and cross-GPU transfers).

The attention graph's quality (ΔPPL=+1.33 on 1B) is second-best after DE step. A CUDA-native implementation reading pre-computed attention patterns without re-running attention would recover this overhead.

### 5.7 Speculative Decoding via Retrieval

Two speculative decoding approaches were implemented and benchmarked (1B model only).

#### 5.7.1 Chain-based speculative decoding (v1)

Token 1 selected from the combined Attn+Graph shortlist argmax; tokens 2..D from a 1-hop graph walk. The full model runs two forward passes per step, and `output_attentions=True` forces eager attention throughout.

| Config | Tok/s | Draft acceptance rate | Speedup vs. baseline |
|--------|------:|----------------------:|---------------------:|
| Cosine (D=4) | 2.92 | 27.7% | 0.110× |
| Combined Attn+Graph (D=4) | 2.88 | 28.1% | 0.108× |
| **Full-vocab baseline** | **26.60** | — | **1.0×** |

At 2.88–2.92 tok/s this is ~11% of baseline. Two compounding factors: (1) two full forward passes per step with no parallelism gain; (2) `output_attentions=True` adding eager-attention overhead per step.

![Speculative decoding (chain-based v1): effective tok/s and draft acceptance rate vs. draft length D.](../results/speculative_decoding_sweep.png)

#### 5.7.2 Retrieval-based speculative decoding (v2)

An offline index of 50,000 completion sequences (8 tokens each) is built. At inference time: (1) one probe forward pass with KV-cache captures `h_T`; (2) top-K completions retrieved by cosine similarity; (3) each candidate verified using the cached KV-cache — O(D) attention cost instead of O(T+D). No `output_attentions`; standard SDPA throughout.

| K | D | Tok/s | Tok/step | Draft acc. | Speedup |
|--:|--:|------:|---------:|-----------:|--------:|
| 4 | 4 | 2.44 | 1.19 | 1.2% | 0.092× |
| 4 | 8 | 2.38 | 1.16 | 0.5% | 0.090× |
| 8 | 4 | 1.63 | 1.27 | 0.8% | 0.061× |
| 8 | 8 | 1.58 | 1.24 | 0.4% | 0.060× |
| 16 | 4 | 1.03 | 1.40 | 0.6% | 0.039× |
| 32 | 4 | 0.63 | 1.59 | 0.5% | 0.024× |

The KV-cache fix eliminated the O(T²) quadratic cost from v1, but draft acceptance rates collapsed to **0.1–1.2%**. The root cause is a **retrieval signal mismatch**: the index embeds prefix hidden states from an offline training-data pass; inference-time `h_T` differs systematically. The retrieval signal needs to be discriminatively trained.

### 5.8 Cost Analysis

Using GPU on-demand pricing (A100 @ $3.00/hr = $0.000833/s):

| Method | K | Tok/s | $/1M tokens | vs baseline |
|--------|--:|------:|------------:|------------:|
| **Baseline** | 128,256 | 26.60 | **$31.33** | — |
| Cosine | 10,000 | 23.72 | $35.13 | +12% |
| DE (step) | 2,000 | 23.20 | $35.92 | +15% |
| DE (step) | 5,000 | 23.16 | $35.99 | +15% |
| DE (step) | 10,000 | 22.39 | $37.22 | +19% |
| MLP Graph | 10,000 | 22.83 | $36.50 | +16% |
| Attn Graph | 10,000 | 19.98 | $41.72 | +33% |
| Spec decoding (v1) | D=4 | 2.88 | ~$289 | +822% |

On GPU the cost overhead of pruned methods is modest: 12–33% above baseline for the viable methods. Break-even between router cost and `lm_head` savings still requires a larger model (see Section 6.3).

---

## 6. Discussion

### 6.1 The Blind-Spot Risk

Dynamic vocabulary pruning introduces a fundamental tension: the router can only shortlist tokens it predicts are relevant, but creative or unexpected tokens by definition have low prior probability given context.

Our results directly quantify this risk. DE (step) achieves 99.5% acceptance at all K values from 512 to 10k on 1B, and 99.8% on 8B — meaning the correct token is in the shortlist at nearly every step regardless of K. The limiting factor is not shortlist size but which embedding-space neighbours the router retrieves: the ~0.2% of missed tokens are systematically hard cases (rare tokens, domain shifts) that the router misses at any K.

### 6.2 Limitations

- Evaluation limited to 1B and 8B models on a single dataset (WikiText-2 English prose); 70B+ and domain generalisation remain open.
- The 8B evaluation used `device_map="auto"` across two T4 GPUs, introducing cross-GPU communication overhead. A single A100 evaluation would give a cleaner `lm_head` fraction measurement for 8B.
- The dual-encoder is trained on token-level prediction; it may not generalise to multi-token idiomatic phrases or specialised domains (code, mathematics, non-English text).
- Router training uses only WikiText-2; domain mismatch could reduce acceptance rates.

### 6.3 The lm_head Fraction Problem

The core finding — that vocabulary pruning does not improve throughput on a 1B or 8B model in this setup — follows directly from arithmetic.

**1B model (T4, single GPU):**
```
lm_head saving  =  37.60 ms × 0.061 × 0.922 = 2.12 ms
Router overhead (DE, K=10k) = 44.67 − 37.60 = 7.07 ms
Net: +4.95 ms/step → −13% throughput
```

**8B model (T4×2, device_map="auto"):**
```
lm_head saving  =  87.14 ms × 0.0469 × 0.922 = 3.77 ms
Router overhead (DE, K=10k) = 101.07 − 87.14 = 13.93 ms
Net: +10.16 ms/step → −13% throughput
```

For pruning to be net-positive, the `lm_head` fraction must exceed:

$$f_{\text{break-even}} = \frac{\text{router\_overhead\_ms}}{\text{step\_ms} \times (1 - K/V)}$$

| Model | f_break-even | f_measured | Gap |
|-------|-------------|------------|-----|
| 1B (T4, single) | ~20% | 6.1% | 3.3× short |
| 8B (T4×2) | ~17% | 4.69% | 3.6× short |
| 70B (est. single A100) | ~0.6% | >6% | break-even reached |

On both evaluated models, break-even is not reached. The 8B model's lower measured fraction (4.69% vs. 6.1%) is an artefact of T4×2 sharding overhead inflating total step time. On a single A100 with no cross-GPU communication, the 8B `lm_head` fraction would likely be *higher* than 1B (larger matrix reads), making the gap smaller. At 70B scale (step ≈1,200 ms), break-even drops to ~0.6% — well below the expected `lm_head` fraction. This is the regime where vocabulary pruning becomes genuinely useful.

Importantly, the **memory bandwidth saving is real at all scales**. Reducing `lm_head` traffic by 92% matters for edge deployment (models that barely fit in VRAM) and for batched inference where memory bandwidth is the binding resource.

### 6.4 Speculative Decoding via Retrieval

Two implementations were evaluated. The chain-based v1 achieved ~28% draft acceptance while penalised by two full forward passes per step and eager attention overhead. The retrieval-based v2 eliminated both overhead sources via KV-cache reuse and standard SDPA, but draft acceptance collapsed to 0.1–1.2%. Key factors:

1. **Retrieval signal mismatch:** index embeddings are from offline training-data prefixes; inference-time `h_T` differs systematically.
2. **No discriminative training:** the index uses general-purpose LLM representations, not representations trained to distinguish accepted from rejected draft sequences.
3. **Sequential verifier passes:** a tree-structured verifier pass over all K candidates in a single batched forward pass would reduce per-step cost from O(K·D) to O(D).

The underlying architecture — retrieval of completion sequences, KV-cache reuse in the verifier, standard SDPA — is sound. The missing ingredient is a discriminatively trained retrieval model.

---

## 7. Conclusion and Future Work

We implemented and benchmarked six vocabulary pruning strategies and two retrieval-based speculative decoders against a full-vocabulary baseline on both `Llama-3.2-1B-Instruct` (NVIDIA T4, single GPU) and `Llama-3.1-8B-Instruct` (two NVIDIA T4 GPUs, `device_map="auto"`), using WikiText-2.

**The memory bandwidth result is the genuine contribution.** Pruning `lm_head` to K=10k reduces the weight matrix loaded per step from 525 MB to 41 MB (1B) or 1,049 MB to 82 MB (8B) — a 92.2% reduction — while DE (step) at K=5k keeps ΔPPL to +1.68 (1B) and **+0.55** (8B) above their respective baselines.

**The throughput result is an honest negative on both models.** The `lm_head` fraction is 6.1% on the 1B T4 and 4.69% on the 8B T4×2 (the latter counter-intuitively lower due to sharding overhead). Break-even requires f≈20% (1B) or ≈17% (8B); no pruned method reduces wall-clock cost at these model scales. The DE (prefetch+refresh) variant exceeds 1B baseline throughput (+4%, 27.67 tok/s) by amortising router cost, but at +146 PPL. On 8B, the overhead gap is narrowing: DE (step) costs −11% to −14% (vs. −13% to −16% on 1B). Break-even for quality-preserving pruning requires a much larger model (70B+) or a GPU-native router kernel reducing overhead below ~1 ms.

**Scale generalisation.** DE (step) and MLP Graph transfer perfectly to 8B (99.8% acceptance), and quality improves: DE step K=10k reaches ΔPPL=+0.29 on 8B vs. +1.03 on 1B. This confirms the routing approach is robust to model scale; the throughput bottleneck is the overhead budget, not acceptance quality.

**Speculative decoding via retrieval** was evaluated in two implementations. The chain-based v1 achieved 2.88 tok/s (11% of baseline) with 28% draft acceptance. The retrieval-based v2 removed both overhead sources but draft acceptance collapsed to 0.1–1.2% due to retrieval signal mismatch. A useful retrieval-based speculative decoder requires a *discriminatively trained* retrieval model.

**Future work:**

- **GPU-native router kernel:** implement the DE router as a fused CUDA kernel to reduce overhead from ~7 ms (1B) / ~14 ms (8B) to <1 ms — the key step for making pruning viable on 7B+ models.
- **Single-GPU 8B / 70B evaluation:** reproduce 8B on a single A100 to isolate the true `lm_head` fraction without sharding overhead; extend to 70B+ where break-even is predicted.
- **Instruction-tuned router (INSTRUCTOR-style):** prepend task instructions to enable a single router to handle multiple domains without retraining.
- **Discriminative retrieval for speculative decoding:** train a contrastive retrieval model on accepted vs. rejected draft sequences. Combined with a compact Medusa head for verification, this is the most promising path to retrieval-based speculative speedup.
- **Online router adaptation:** fine-tune the router MLP online using accepted/rejected token signal from each inference step.

---

## References

- Cai, T., et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.*
- Chen, C., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.*
- Henderson, M., et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply.*
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.*
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.*
- Merity, S., et al. (2016). *Pointer Sentinel Mixture Models.* (WikiText-2 dataset)
- Su, H., et al. (2022). *One Embedder, Any Task: Instruction-Finetuned Text Embeddings.*
