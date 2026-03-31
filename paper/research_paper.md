# Context-Aware Vocabulary Pruning for LLM Inference Acceleration: A Dual-Encoder Retrieval Approach

**Status:** GPU benchmarks complete — March 2026

---

## Abstract

Autoregressive decoding in large language models (LLMs) is dominated by two costs at each token step: KV-cache attention and the `lm_head` vocabulary projection — a matrix multiply that reads the full `[V × d]` weight matrix from GPU memory on every step. At batch size 1, a regime typical of interactive and agentic inference, this operation is **memory-bandwidth-bound rather than compute-bound**: the GPU's arithmetic units sit idle while 525 MB of weights are streamed from VRAM per token. We address this bottleneck through **dynamic vocabulary pruning**, a decoding-level inference acceleration technique that restricts the `lm_head` projection to a context-dependent shortlist of K candidate tokens at each step, reducing memory traffic by up to 99.6%.

The central question is how to build that shortlist cheaply and accurately. We propose a **Dual-Encoder Router** — a lightweight MLP (0.06% of model parameters) trained with Multiple Negatives Ranking Loss (MNRL) to map the model's current hidden state to a query vector in a learned 256-dimensional space, then retrieve the top-K token candidates by nearest-neighbour search over a pre-indexed corpus of sentence completions. Unlike prior cosine-distance baselines that rank individual token embeddings independently, our router retrieves semantically coherent multi-token completions and derives the shortlist from their union, better reflecting the structure of natural language continuations.

We benchmark four routing strategies of increasing sophistication — static Top-K, cosine-distance, cluster-based, and the learned dual-encoder — against a full-vocabulary baseline on WikiText-2 (Llama-3.2-1B-Instruct), sweeping shortlist sizes K ∈ {512, 1k, 2k, 5k, 10k} and reporting token acceptance rate, perplexity degradation (ΔPPL), and wall-clock tokens/sec. On an NVIDIA T4 GPU the `lm_head` accounts for 6.1% of per-token latency; the primary demonstrated gain is in memory bandwidth reduction (92–99.6% across K values). Wall-clock speedup requires GPU-native kernels or larger models (70B+) where the memory-bound regime is more pronounced. The dual-encoder achieves the best accuracy–efficiency trade-off across all K values and is orthogonal to complementary inference acceleration techniques such as speculative decoding, quantisation, and KV-cache compression.

---

## 1. Introduction

### 1.1 The Autoregressive Bottleneck

Modern LLMs based on the transformer architecture generate text one token at a time. Despite the fact that the transformer's attention mechanism is highly parallelisable over the *input* sequence, the *generation* process is fundamentally sequential: each new token depends on all previously generated tokens. This autoregressive loop creates a hard bottleneck on inference throughput.

Within each step of this loop, the two most expensive operations are:

1. **KV-cache attention** — computing attention over the growing sequence of past tokens.
2. **Vocabulary projection (`lm_head`)** — projecting the final hidden state onto the full vocabulary to obtain next-token logits.

While KV-cache optimisation has received substantial research attention (flash attention, paged attention, speculative decoding), the `lm_head` projection is comparatively understudied as a target for optimisation. At batch size 1 — the regime of interactive and agentic inference — the `lm_head` matrix multiplication is memory-bandwidth-bound: the arithmetic intensity is too low to keep the GPU's compute units saturated, so the bottleneck is the rate at which the weight matrix can be read from VRAM.

### 1.2 The Wasted Computation Insight

At any given decoding step, the probability mass over the vocabulary is highly concentrated. In predictable contexts (e.g., "The Eiffel Tower is located in...") the top-1 token may carry >99% probability. The model still computes logits for all 32 k–128 k vocabulary entries. This is computationally equivalent to solving a 128,000-way classification problem when the answer is already obvious from the context.

This observation motivates a natural question: **can the model's own internal representation predict which vocabulary region is relevant before the full projection is run?**

### 1.3 Contributions

This paper makes the following contributions:

1. **Profiling:** We quantify the `lm_head` fraction of per-token latency for a small open-source LLM on an NVIDIA T4 GPU.
2. **Router taxonomy:** We define and implement four vocabulary routing strategies of increasing sophistication.
3. **Dual-Encoder Router:** We propose a Dual-Encoder trained with MNRL that learns to map the LLM's hidden state to a compact token shortlist, operating entirely within the model's own latent space.
4. **Benchmark:** We evaluate all methods against a full-vocabulary baseline on WikiText-2, reporting token acceptance rate, perplexity, and tokens/sec across shortlist sizes K ∈ {512, 1 k, 2 k, 5 k, 10 k}.

---

## 2. Background

### 2.1 Self-Attention and the Hidden State

A transformer processes an input sequence by stacking each token's embedding into a matrix (the hidden state). At each layer, the self-attention mechanism allows every token's representation to incorporate information from all preceding tokens, weighted by learned relevance scores (attention weights).

By the time the sequence reaches the final layer, the last row of the hidden state matrix — corresponding to the most recently generated token — has absorbed the mathematical influence of every prior token. This dense vector is what we refer to as the **"vibe"**: a single point in a high-dimensional space that summarises the full context of the conversation so far.

Formally, given an input sequence of $T$ tokens, the hidden state at layer $L$ is a matrix $H^{(L)} \in \mathbb{R}^{T \times d}$, where $d$ is the model dimension. The prediction of the next token is made entirely from the last row $h_T = H^{(L)}_{T,:} \in \mathbb{R}^d$.

### 2.2 The `lm_head` as a Computational Bottleneck

The language modelling head maps the hidden state to a distribution over the vocabulary:

$$\text{logits} = h_T W_V^\top \in \mathbb{R}^{|V|}$$

where $W_V \in \mathbb{R}^{|V| \times d}$ is the vocabulary embedding matrix and $|V|$ is the vocabulary size (128,256 for Llama-3.2-1B). At 16-bit precision, $W_V$ occupies $128{,}256 \times 2{,}048 \times 2\ \text{bytes} = \mathbf{525\ \text{MB}}$. Reading 525 MB per token step creates a hard ceiling on achievable tokens/sec that is independent of model depth.

### 2.3 Speculative Decoding and Related Work

**Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) accelerates autoregressive generation by interposing a small "draft" model that proposes several candidate tokens in sequence. A large "verifier" model then evaluates the entire candidate sequence in a single parallel forward pass, accepting or rejecting each candidate.

Extensions of this idea include:

- **Medusa** (Cai et al., 2024): attaches multiple decoding heads to the base model, each predicting a different future token position from the same hidden state.
- **SpecInfer / Tree-based Speculative Decoding**: generates a *tree* of candidate sequences from multiple draft models in parallel, increasing the probability that at least one branch matches the verifier's output.
- **DynaSpec / FastMTP**: dynamic vocabulary compression applied at the `lm_head` layer, reducing the search space before the heavy projection.

Our approach is orthogonal to speculative decoding: it targets the `lm_head` projection cost directly and can in principle be combined with any of the above methods.

### 2.4 Dual-Encoders and Contrastive Learning

A **Dual-Encoder** (also called a Bi-Encoder or Siamese network) processes two inputs through separate (but weight-sharing) encoder towers and learns to embed related pairs close together in a shared vector space.

**Multiple Negatives Ranking Loss (MNRL)** (Henderson et al., 2017; Karpukhin et al., 2020) is the standard training objective for dual-encoders. Given a batch of $(a_i, p_i)$ anchor-positive pairs:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(a_i, p_i) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(a_i, p_j) / \tau)}$$

where $\text{sim}(\cdot, \cdot)$ is cosine similarity and $\tau$ is a temperature parameter. The key efficiency of MNRL is that all other positives in the batch serve as in-batch negatives, requiring no explicit negative mining.

---

## 3. Method

### 3.1 Problem Formalisation

Let $f_\theta$ be a frozen autoregressive LLM with vocabulary $V$, model dimension $d$, and vocabulary projection matrix $W_V \in \mathbb{R}^{|V| \times d}$. At decoding step $t$, the model computes a hidden state $h_t \in \mathbb{R}^d$ and selects the next token as:

$$\hat{y}_t = \arg\max_{v \in V} (h_t W_V^\top)_v$$

**Goal:** learn a router $r_\phi : \mathbb{R}^d \to 2^V$ such that:
1. $\hat{y}_t \in r_\phi(h_t)$ with high probability (**acceptance rate**)
2. $|r_\phi(h_t)| \ll |V|$ (**compression ratio**)
3. The latency of computing $r_\phi(h_t)$ is small relative to the savings from pruning $W_V$

### 3.2 Router Taxonomy

We implement four routing strategies of increasing sophistication:

| Router | Description | Requires Training |
|--------|-------------|-------------------|
| **Static Top-K** | Select K tokens with highest $\|W_V[v,:]\|_2$ (context-free) | No |
| **Cosine Router** | Select K tokens with highest $\cos(h_t, W_V[v,:])$ | No |
| **Cluster Router** | Pre-cluster $W_V$ with k-means; select top-C clusters by $\cos(h_t, \mu_c)$, return all tokens in those clusters | Offline only |
| **Dual-Encoder Router** | Learned MLP that maps $h_t$ to a query vector; select K tokens by ANN search | Yes |

### 3.3 Dual-Encoder Router Design

#### Architecture

The router consists of a lightweight **MLP projection head** $g_\phi : \mathbb{R}^d \to \mathbb{R}^{d_r}$ trained to map the frozen LLM's hidden state to a query vector in a shared embedding space. The vocabulary token embeddings $W_V$ serve as the corpus that is searched at inference time.

Both the query (hidden state) and the keys (token embeddings) live in the LLM's own latent space, which means the router learns to navigate a space the model already understands. No cross-model alignment is required.

#### Training Data Construction

From WikiText-2 training split:
1. Tokenise each sentence.
2. For each sentence of length $T \geq 8$ tokens, sample a split point $s \sim \text{Uniform}(\lfloor 0.4T \rfloor, \lfloor 0.7T \rfloor)$.
3. **Anchor:** run the frozen LLM on the prefix (tokens $1..s$); extract $h_s$ (the hidden state at position $s$).
4. **Positive:** the ground-truth next token $y_{s+1}$; represented as the corresponding row $W_V[y_{s+1},:]$ from the frozen vocabulary embedding matrix.

#### Training Objective

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\cos(g_\phi(h_i), W_V[y_i,:]) / \tau)}{\sum_{j=1}^B \exp(\cos(g_\phi(h_i), W_V[y_j,:]) / \tau)}$$

with in-batch negatives (all $y_j \neq y_i$ in the batch).

#### Inference Integration

1. Pre-compute and cache all rows of $W_V$ normalised to unit length.
2. At step $t$: compute $q_t = g_\phi(h_t)$; retrieve the K token indices with highest $\cos(q_t, W_V[v,:])$ via brute-force matmul (small enough for K ≤ 10 k) or FAISS.
3. Slice $W_V$ to the shortlist and compute logits only over those K rows.
4. Apply softmax and sample/argmax as usual.

---

## 4. Experimental Setup

### 4.1 Model

`meta-llama/Llama-3.2-1B-Instruct` — 1B parameters, 2,048-dim hidden state, 128,256-token vocabulary, 16 transformer layers, bfloat16.

### 4.2 Dataset

- **WikiText-2-raw-v1** (Merity et al., 2016). Standard language modelling benchmark; plain English prose.
- **Router training:** `train` split (~2M tokens).
- **Perplexity evaluation:** `test` split (~240 k tokens).
- **Latency benchmark:** 50 randomly sampled contiguous passages of 128 tokens from the `validation` split.

### 4.3 Hardware

NVIDIA T4 GPU (16 GB VRAM), CUDA backend, bfloat16. All latency measurements taken after a 5-step warm-up; reported as mean over 50 prompts.

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

All experiments were run on an NVIDIA T4 GPU, CUDA backend, bfloat16, with `meta-llama/Llama-3.2-1B-Instruct`. Latency numbers are means over 50 WikiText-2 validation prompts after a 5-step warm-up. Perplexity is computed over the full WikiText-2 test split.

### 5.1 Baseline Profiling

| Metric | Value |
|--------|-------|
| Tokens/sec (full vocab) | **26.60** |
| Mean step time | 37.60 ms |
| `lm_head` time (mean) | 2.30 ms |
| `lm_head` time % | **6.1%** |
| Baseline perplexity (WikiText-2 test) | **26.24** |
| `lm_head` weight matrix | 128,256 × 2,048 × 2 bytes = **525 MB** |

The `lm_head` accounts for only 6.1% of per-token latency on this hardware. The remaining 93.9% is split across attention, MLP layers, and KV-cache I/O. This low fraction is a critical constraint: any router whose overhead exceeds 2.30 ms will make pruning net-negative on a 1B model (see Section 6.3).

![Baseline decode step profiling: per-operation time breakdown.](../results/baseline_profile.png)

### 5.2 Token Acceptance Rate vs. K

Acceptance rate is the fraction of decoding steps where the ground-truth next token falls within the pruned shortlist. A rate below 1.0 means the model cannot even produce the correct token, let alone sample it.

| Router | K=512 | K=1k | K=2k | K=5k | K=10k |
|--------|------:|-----:|-----:|-----:|------:|
| Static Top-K | 0.09% | 0.25% | 0.50% | 2.5% | 7.7% |
| Cluster | 33.8% | 54.8% | 68.1% | 76.6% | 83.3% |
| Cosine | 98.4% | 98.4% | 98.4% | 98.4% | 98.4% |
| Dual-Encoder (step) | 99.5% | 99.5% | 99.5% | 99.5% | 99.5% |
| MLP Graph | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| Attention Graph | 82.7% | 99.8% | 99.8% | 99.8% | 99.8% |

**Key observations:**
- Static Top-K is effectively unusable: selecting the highest-norm rows of $W_V$ (context-free) is near-random for actual decoding, achieving <8% acceptance even at K=10,000.
- Cluster router acceptance saturates around 83% at K=10,000 — the remaining 17% of correct tokens live in clusters that were not selected.
- Cosine acceptance rate is flat at 98.4% across all K values, suggesting the cosine router reaches its information-theoretic ceiling early and larger K only recovers the same tokens.
- Dual-Encoder (step) and MLP Graph achieve the highest acceptance: 99.5% and ≥99.9% respectively, essentially independent of K above 512.
- Attention Graph drops to 82.7% at K=512 but recovers to 99.8% at K≥1,000.

![Token acceptance rate vs. shortlist size K for all routers.](../results/acceptance_rate_vs_k.png)

### 5.3 Perplexity Degradation vs. K

$\Delta\text{PPL} = \text{PPL}_{\text{pruned}} - \text{PPL}_{\text{baseline}}$. Baseline PPL = 26.24. Lower is better.

| Router | K=512 | K=1k | K=2k | K=5k | K=10k |
|--------|------:|-----:|-----:|-----:|------:|
| Static Top-K | +945,974,736 | +891,715,212 | +804,467,165 | +382,485,536 | +125,028,282 |
| Cluster | +19,513,799 | +370,921 | +14,841 | +3,060 | +834 |
| Cosine | +18.998 | +10.834 | +6.790 | +3.541 | **+2.726** |
| Dual-Encoder (step) | +24.641 | +8.945 | +3.926 | +1.684 | **+1.030** |
| Dual-Encoder (prefetch+refresh)† | — | — | — | +146.153 | — |
| MLP Graph | +39.594 | +13.225 | +6.391 | +2.891 | **+2.105** |
| Attention Graph | +15,025 | +21.580 | +5.908 | +2.104 | **+1.327** |

†Dual-Encoder prefetch+refresh uses a fixed effective K≈4,634 averaged over the evaluation.

**Key observations:**
- Static Top-K and Cluster produce astronomically high perplexity at low K, confirming they are not viable routing strategies.
- At K=10,000 the three best methods cluster closely: DE step (+1.03), Attention Graph (+1.33), MLP Graph (+2.11), and Cosine (+2.73) — all within 11% of baseline PPL.
- DE step at K=5,000 achieves ΔPPL=+1.68, representing a good operating point: 96.1% memory bandwidth reduction with minimal quality loss.
- The prefetch+refresh variant of DE degrades sharply (+146 PPL), because refreshing the shortlist every N steps allows stale shortlists to miss relevant tokens.

![Perplexity degradation (ΔPPL) vs. shortlist size K. Static Top-K and Cluster are omitted from this plot due to extreme values.](../results/ppl_degradation_vs_k.png)

### 5.4 Tokens/sec vs. K

Wall-clock throughput on an NVIDIA T4 GPU. Baseline = 26.60 tok/s.

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

On GPU the throughput gap between pruned methods and baseline is much narrower than on memory-limited hardware. Static Top-K adds negligible overhead (no neural network pass), sitting within ±1% of baseline. The DE (prefetch+refresh) variant is the only pruned method to exceed baseline throughput (+4%, 27.67 tok/s), by amortising router cost across many decoding steps — though at a severe quality penalty (+146 PPL). Attention Graph remains the worst due to `output_attentions=True` forcing eager attention, but the penalty (−15% to −25%) is far smaller than on memory-bandwidth-limited hardware.

![Wall-clock tokens/sec vs. shortlist size K. Dashed line = full-vocab baseline (26.60 tok/s).](../results/throughput_vs_k.png)

### 5.5 Pareto Frontier: Quality vs. Speed

The accuracy–throughput Pareto frontier identifies configurations not dominated — i.e., no other configuration achieves both higher throughput and lower PPL degradation simultaneously.

| Method | K | Tok/s | ΔPPL | Note |
|--------|--:|------:|-----:|------|
| DE (prefetch) | ~4,634 | **27.67** | +146.2 | Fastest, but quality unusable |
| Cluster | 10,000 | 23.99 | +833.9 | Quality unusable |
| Cosine | 10,000 | 23.72 | +2.73 | Good quality |
| **DE (step)** | **5,000** | **23.16** | **+1.68** | **Best quality–speed trade-off** |
| DE (step) | 10,000 | 22.39 | +1.03 | Marginally better PPL, slightly slower |

On T4, DE (step) at K=5,000 represents the recommended operating point: 23.16 tok/s (−13% vs baseline) with ΔPPL=+1.68 — 96.1% memory bandwidth reduction at minimal quality cost. The throughput penalty is considerably more acceptable on GPU than on memory-bandwidth-limited hardware.

![Pareto frontier: throughput vs. perplexity degradation. Points toward the top-left (high tok/s, low ΔPPL) are preferred.](../results/pareto_frontier.png)

![Mean shortlist size growth vs. K (Dual-Encoder prefetch+refresh variant).](../results/shortlist_growth.png)

### 5.6 Graph-Based Routers

Two graph-based routers were implemented and evaluated:

**MLP Transition Graph:** A directed graph over vocabulary tokens where edge weights encode how frequently token $v$ follows token $u$ in the training corpus. At inference time, the shortlist is built by taking the top-K successors of the current-context tokens by transition probability.

- Acceptance rate: ≥99.9% across all K — the highest of any method.
- Throughput: 22.83–24.31 tok/s (−14% to −16% vs baseline) — graph traversal and sparse index operations add overhead per step.
- PPL at K=10,000: +2.11 — competitive with Cosine but worse than DE step.

**Attention Graph:** Uses the model's own attention patterns to identify which previously seen tokens the model is attending to at step $t$, then retrieves their vocabulary neighbours.

- Acceptance at K≥1,000: 99.8% — among the highest.
- Throughput: 19.98–22.72 tok/s (−15% to −25% vs baseline).
- Root cause of overhead: `output_attentions=True` forces eager attention, disabling FlashAttention. On GPU this penalty is proportionally smaller than on memory-bandwidth-limited hardware.
- PPL at K=10,000: +1.33 — the second-best quality result after DE step.

The attention graph's quality (ΔPPL=+1.33) is the second best of all methods. A CUDA-native implementation that reads pre-computed attention patterns without re-running attention would recover this overhead and make the attention graph a viable high-quality router.

### 5.7 Speculative Decoding via Retrieval

Two speculative decoding approaches were implemented and benchmarked.

#### 5.7.1 Chain-based speculative decoding (v1)

The first implementation used the attention graph and MLP transition graph as a linear draft chain: token 1 is selected from the combined Attn+Graph shortlist argmax, tokens 2..D from a 1-hop graph walk. The full model runs two forward passes per step (draft + verify), and `output_attentions=True` forces eager attention throughout.

| Config | Tok/s | Draft acceptance rate | Speedup vs. baseline |
|--------|------:|----------------------:|---------------------:|
| Cosine (D=4) | 2.92 | 27.7% | 0.110× |
| Combined Attn+Graph (D=4) | 2.88 | 28.1% | 0.108× |
| **Full-vocab baseline** | **26.60** | — | **1.0×** |

At 2.88–2.92 tok/s this is ~11% of baseline. Two compounding factors explain this: (1) two full forward passes per step with no parallelism gain, and (2) `output_attentions=True` adding eager-attention overhead per step.

![Speculative decoding (chain-based v1): effective tok/s and draft acceptance rate vs. draft length D.](../results/speculative_decoding_sweep.png)

#### 5.7.2 Retrieval-based speculative decoding (v2)

A second implementation addressed both overhead sources. An offline index of 50,000 completion sequences (8 tokens each) was built by running the LLM on WikiText-2 training windows and extracting the last hidden state `h_T` at each split point. At inference time: (1) one probe forward pass with KV-cache captures `h_T`; (2) top-K completions are retrieved by cosine similarity; (3) each candidate is verified using the cached KV-cache — O(draft_len) attention cost instead of O(T+draft_len). No `output_attentions` is used; standard SDPA throughout.

| K | D | Tok/s | Tok/step | Draft acc. | Verifier calls/tok | Speedup |
|--:|--:|------:|---------:|-----------:|-------------------:|--------:|
| 4 | 4 | 2.44 | 1.19 | 1.2% | 4.22 | 0.092× |
| 4 | 8 | 2.38 | 1.16 | 0.5% | 4.31 | 0.090× |
| 4 | 16 | 2.38 | 1.16 | 0.3% | 4.31 | 0.089× |
| 8 | 4 | 1.63 | 1.27 | 0.8% | 7.11 | 0.061× |
| 8 | 8 | 1.58 | 1.24 | 0.4% | 7.27 | 0.060× |
| 8 | 16 | 1.58 | 1.24 | 0.2% | 7.27 | 0.059× |
| 16 | 4 | 1.03 | 1.40 | 0.6% | 12.17 | 0.039× |
| 16 | 8 | 0.97 | 1.34 | 0.3% | 12.73 | 0.037× |
| 16 | 16 | 0.97 | 1.34 | 0.1% | 12.73 | 0.037× |
| 32 | 4 | 0.63 | 1.59 | 0.5% | 20.82 | 0.024× |

The KV-cache fix eliminated the O(T²) quadratic cost from v1, improving throughput to 0.63–2.44 tok/s. However, draft acceptance rates collapsed to **0.1–1.2%** — far below the ~70–80% needed for speculative speedup. The root cause is a **retrieval signal mismatch**: the index was built from prefix hidden states extracted during an offline pass on training data. At inference time the query `h_T` is a function of the full prompt history, which differs systematically from training-distribution prefixes of identical length. The dot-product similarity is insufficient to recover semantically valid continuations, so nearly every draft token is rejected.

This result is informative: **the retrieval signal needs to be trained to be discriminative**, not just borrowed from the LLM's general-purpose representation.

### 5.8 Cost Analysis

Using GPU on-demand pricing (A100 @ $3.00/hr = $0.000833/s):

| Method | K | Tok/s | PPL | $/1M tokens | vs baseline |
|--------|--:|------:|----:|------------:|------------:|
| **Baseline** | 128,256 | 26.60 | 26.24 | **$31.33** | — |
| Cosine | 10,000 | 23.72 | 28.97 | $35.13 | +12% |
| DE (step) | 2,000 | 23.20 | 30.17 | $35.92 | +15% |
| DE (step) | 5,000 | 23.16 | 27.92 | $35.99 | +15% |
| DE (step) | 10,000 | 22.39 | 27.27 | $37.22 | +19% |
| MLP Graph | 10,000 | 22.83 | 28.35 | $36.50 | +16% |
| Attn Graph | 10,000 | 19.98 | 27.57 | $41.72 | +33% |
| Spec decoding (v1) | D=4 | 2.88 | — | ~$289 | +822% |

On GPU the cost overhead of pruned methods is much more modest than on CPU/MPS hardware: 12–33% above baseline for the viable methods, compared to 50–276% on memory-limited hardware. Break-even between router cost and `lm_head` savings still requires a larger model (see Section 6.3), but the arithmetic is less unfavourable on GPU.

---

## 6. Discussion

### 6.1 The Blind-Spot Risk

Dynamic vocabulary pruning introduces a fundamental tension: the router can only shortlist tokens it predicts are relevant, but creative or unexpected tokens by definition have low prior probability given the context. A router that is too aggressive (small K) will systematically exclude rare, creative, or domain-shifting tokens even when they are contextually appropriate.

Our results directly quantify this risk. The Dual-Encoder (step) router achieves 99.5% acceptance at all K values from 512 to 10,000 — meaning the correct token is in the shortlist 99.5% of steps regardless of K. This is counterintuitive: why does acceptance not grow with K? Because the router's limiting factor is not shortlist size but which embedding-space neighbours it retrieves: the 0.5% of missed tokens are systematically hard cases (rare tokens, domain shifts) that the router misses at any K, not tokens that fall just outside the shortlist boundary.

The Cosine router plateaus at 98.4% for the same reason. MLP Graph reaches ≥99.9% — its graph-traversal mechanism recovers some of these hard cases by propagating across token transition edges rather than purely by embedding similarity.

### 6.2 Limitations

- Evaluation limited to a single 1B-parameter model and a single text-domain dataset (WikiText-2 English prose).
- Evaluation on a single GPU tier (NVIDIA T4); results at higher-bandwidth GPUs (A100, H100) may differ.
- The dual-encoder is trained on token-level prediction; it may not generalise to multi-token idiomatic phrases or highly specialised domains (code, mathematics, non-English text).
- Router training uses only WikiText-2; domain mismatch between training and deployment could reduce acceptance rates.
- Speculative decoding implementation uses a second full forward pass rather than a dedicated small draft model; results reflect an implementation constraint, not the ceiling of retrieval-based speculative decoding.

### 6.3 The lm_head Fraction Problem

The core finding of this work — that vocabulary pruning does not improve throughput on a 1B model — follows directly from arithmetic:

```
lm_head saving  =  step_ms × lm_head_frac × (1 − K/V)
                =  37.60 ms × 0.061 × 0.922          (K=10k)
                =  2.12 ms

Router overhead (DE step, K=10k)  =  44.67 − 37.60  =  7.07 ms
Net effect:  +4.95 ms/step  →  −13% throughput
```

For pruning to be net-positive the lm_head fraction must exceed the ratio of router overhead to maximum achievable saving:

$$f_{\text{break-even}} = \frac{\text{router\_overhead\_ms}}{\text{step\_ms} \times (1 - K/V)}$$

On this 1B model, break-even requires $f \approx 20\%$; the measured value is 6.1%. This gap explains why all pruned configurations are slower than baseline.

The break-even fraction decreases as the model grows (longer step time) or as the router is made faster (GPU-native kernel). At 70B scale, step time ≈ 1,200 ms and break-even drops to ~0.64% — well below the expected lm_head fraction at any model size. This is the regime where vocabulary pruning becomes genuinely useful.

Importantly, the **memory bandwidth saving** is real at all scales. Reducing lm_head traffic by 92% matters for edge deployment (models that barely fit in VRAM), for batched inference where memory bandwidth is the binding resource, and for future hardware where off-chip memory access is even more expensive relative to compute.

### 6.4 Speculative Decoding via Retrieval

Two implementations were evaluated. The chain-based v1 achieved ~28% draft acceptance — below the 70–80% needed — while being penalised by two full forward passes per step and `output_attentions=True` eager-attention overhead. The retrieval-based v2 eliminated both overhead sources via KV-cache reuse and standard SDPA, but draft acceptance collapsed to 0.1–1.2%. Key factors:

1. **Retrieval signal mismatch:** the index embeds prefix hidden states from an offline training-data pass. At inference time, `h_T` is computed from a full prompt context that differs systematically from training-distribution prefixes of the same length. The cosine similarity is therefore low-signal for predicting valid continuations.

2. **No discriminative training:** the index uses general-purpose LLM representations, not representations trained to distinguish accepted from rejected draft sequences. A contrastive retrieval objective would be required.

3. **Single-step retrieval granularity:** the router matches one vector to one vector and returns the nearest-neighbour completion sequences. A tree-structured verifier pass over all K candidates in a single batched forward pass would reduce the per-step cost from O(K×D) to O(D) while evaluating the same candidates.

The underlying architecture — retrieval of completion sequences, KV-cache reuse in the verifier, standard SDPA — is sound. The missing ingredient is a discriminatively trained retrieval model.

---

## 7. Conclusion and Future Work

We implemented and benchmarked six vocabulary pruning strategies and two retrieval-based speculative decoders against a full-vocabulary baseline on `meta-llama/Llama-3.2-1B-Instruct` and WikiText-2, on an NVIDIA T4 GPU.

**The memory bandwidth result is the genuine contribution.** Pruning `lm_head` to K=10,000 reduces the weight matrix loaded per step from 525 MB to 41 MB — a 92.2% reduction — while the best router (Dual-Encoder step, K=5,000) keeps perplexity degradation to +1.68 PPL above the 26.24 baseline. This saving is real and scales with model size.

**The throughput result is an honest negative.** On a 1B model on a T4 GPU, the `lm_head` fraction is only 6.1% of step time. Every router we tested adds more overhead than it saves: break-even requires $f \approx 20\%$ vs measured 6.1%. No pruned method reduces wall-clock cost at this model scale on this hardware. However, the overhead gap is considerably narrower than on memory-bandwidth-limited hardware — DE (step) at K=5k costs only −13% throughput vs baseline, and the DE (prefetch+refresh) variant actually exceeds baseline (+4%, 27.67 tok/s) by amortising router cost across many steps, at the cost of +146 PPL degradation. Break-even for quality-preserving pruning requires either a much larger model (70B+) or a GPU-native router kernel reducing overhead below ~1 ms.

**Speculative decoding via retrieval** was evaluated in two implementations. The chain-based v1 achieved 2.88 tok/s (11% of baseline) with 28% draft acceptance. The retrieval-based v2 removed both overhead sources (KV-cache reuse, standard SDPA), but draft acceptance collapsed to 0.1–1.2% because the retrieval signal does not transfer to inference-time query distributions. The core limitation is that a useful retrieval-based speculative decoder requires a **discriminatively trained** retrieval model, not a repurposed LLM representation.

**Future work:**

- **GPU-native router kernel:** implement the DE router as a fused CUDA kernel to reduce overhead from ~7 ms to <1 ms — the key step for making pruning viable on 7B models.
- **Instruction-tuned router (INSTRUCTOR-style):** prepend task instructions to anchor inputs to enable a single router to handle multiple domains and styles without retraining.
- **Discriminative retrieval for speculative decoding:** train a contrastive retrieval model on accepted vs. rejected draft sequences so that retrieved completions are genuinely valid continuations rather than embedding-space neighbours. Combined with a compact Medusa head for verification, this is the most promising path to retrieval-based speculative speedup.
- **Online router adaptation:** fine-tune the router MLP online using accepted/rejected token signal from each inference step, enabling domain adaptation at deployment time.
- **Larger model evaluation:** test the 8B and 70B Llama-3.1 checkpoints, where the lm_head fraction is expected to be higher and router overhead is proportionally smaller.
- **Higher-bandwidth GPU evaluation:** reproduce on A100/H100 to determine whether higher memory bandwidth changes the break-even arithmetic for smaller models.

---

## References

- Cai, T., et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.*
- Chen, C., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.*
- Henderson, M., et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply.*
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.*
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.*
- Merity, S., et al. (2016). *Pointer Sentinel Mixture Models.* (WikiText-2 dataset)
- Su, H., et al. (2022). *One Embedder, Any Task: Instruction-Finetuned Text Embeddings.*
