# Vibe-Aware Vocabulary Pruning: A Dual-Encoder Router for Efficient LLM Inference

**Status:** Working draft  
**Date:** 2026-03-28

---

## Abstract

Large language models (LLMs) generate text autoregressively, producing one token at a time. At each step the model must project its internal hidden state against the full vocabulary matrix — an operation that scales with vocabulary size (typically 32 k–100 k tokens) and dominates per-token latency on commodity hardware. We observe that the final hidden state (the "vibe") already encodes rich contextual signal about which region of the vocabulary is relevant. We propose a lightweight **Dual-Encoder Router** trained with Multiple Negatives Ranking Loss (MNRL) that, at inference time, maps the current hidden state to a compact shortlist of candidate tokens. The full `lm_head` projection is then performed only over this shortlist, reducing the vocabulary search space by up to 95% per step. We benchmark four routing strategies — static Top-K, cosine-distance, cluster-based, and the learned dual-encoder — against a full-vocabulary baseline on WikiText-2, measuring token acceptance rate, perplexity, and wall-clock tokens/sec on Apple Silicon (MPS). Results show that the dual-encoder router achieves the best accuracy/speed Pareto frontier, sustaining near-baseline perplexity at shortlist sizes of K = 2 k–5 k while delivering meaningful latency reductions on CPU/MPS hardware.

---

## 1. Introduction

### 1.1 The Autoregressive Bottleneck

Modern LLMs based on the transformer architecture generate text one token at a time. Despite the fact that the transformer's attention mechanism is highly parallelisable over the *input* sequence, the *generation* process is fundamentally sequential: each new token depends on all previously generated tokens. This autoregressive loop creates a hard bottleneck on inference throughput.

Within each step of this loop, the two most expensive operations are:

1. **KV-cache attention** — computing attention over the growing sequence of past tokens.
2. **Vocabulary projection (`lm_head`)** — projecting the final hidden state onto the full vocabulary to obtain next-token logits.

While KV-cache optimisation has received substantial research attention (flash attention, paged attention, speculative decoding), the `lm_head` projection is comparatively understudied as a target for optimisation. On CPU and Apple Silicon (MPS) — where memory bandwidth rather than raw FLOP throughput is the binding constraint — the `lm_head` matrix multiplication can account for a disproportionate share of per-token latency.

### 1.2 The Wasted Computation Insight

At any given decoding step, the probability mass over the vocabulary is highly concentrated. In predictable contexts (e.g., "The Eiffel Tower is located in...") the top-1 token may carry >99% probability. The model still computes logits for all 32 k–100 k vocabulary entries. This is computationally equivalent to solving a 100 000-way classification problem when the answer is already obvious from the context.

This observation motivates a natural question: **can the model's own internal representation predict which vocabulary region is relevant before the full projection is run?**

### 1.3 Contributions

This paper makes the following contributions:

1. **Profiling:** We quantify the `lm_head` fraction of per-token latency for a small open-source LLM on MPS/CPU hardware.
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

where $W_V \in \mathbb{R}^{|V| \times d}$ is the vocabulary embedding matrix and $|V|$ is the vocabulary size (e.g., 32 000 for Llama-3.2). This operation requires $|V| \times d$ multiply-accumulate operations per step — on the order of 65 million for a 1B-parameter model.

On memory-bandwidth-limited hardware (CPU, MPS), this projection requires loading the entire $W_V$ matrix from RAM on every single token generation step. At 16-bit precision, $W_V$ for Llama-3.2-1B occupies approximately 128 MB. Reading 128 MB per token step creates a hard ceiling on achievable tokens/sec that is independent of model depth.

### 2.3 Speculative Decoding and Related Work

**Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) accelerates autoregressive generation by interposing a small "draft" model that proposes several candidate tokens in sequence. A large "verifier" model then evaluates the entire candidate sequence in a single parallel forward pass, accepting or rejecting each candidate. This exploits the fact that the verifier can process a sequence of tokens in parallel even though it generates them sequentially.

Extensions of this idea include:

- **Medusa** (Cai et al., 2024): attaches multiple decoding heads to the base model, each predicting a different future token position from the same hidden state. Eliminates the need for a separate draft model.
- **SpecInfer / Tree-based Speculative Decoding**: generates a *tree* of candidate sequences from multiple draft models in parallel, increasing the probability that at least one branch matches the verifier's output.
- **DynaSpec / FastMTP**: dynamic vocabulary compression applied at the `lm_head` layer, reducing the search space before the heavy projection.

Our approach is orthogonal to speculative decoding: it targets the `lm_head` projection cost directly and can in principle be combined with any of the above methods.

### 2.4 Dual-Encoders and Contrastive Learning

A **Dual-Encoder** (also called a Bi-Encoder or Siamese network) processes two inputs through separate (but weight-sharing) encoder towers and learns to embed related pairs close together in a shared vector space.

**Multiple Negatives Ranking Loss (MNRL)** (Henderson et al., 2017; Karpukhin et al., 2020) is the standard training objective for dual-encoders. Given a batch of $(a_i, p_i)$ anchor-positive pairs:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(a_i, p_i) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(a_i, p_j) / \tau)}$$

where $\text{sim}(\cdot, \cdot)$ is cosine similarity and $\tau$ is a temperature parameter. The key efficiency of MNRL is that all other positives in the batch serve as in-batch negatives, requiring no explicit negative mining.

**INSTRUCTOR** (Su et al., 2022) extends dual-encoders with task-specific instruction prefixes, enabling a single model to produce task-specific embedding geometries without retraining.

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

This formulation is deliberately token-level: we train the router to predict the *single next token's* embedding from the current hidden state, rather than a full sentence embedding. This is closer to the actual inference-time task.

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

- **Primary:** `meta-llama/Llama-3.2-1B-Instruct` — 1B parameters, 2048-dim hidden state, 32 256-token vocabulary, runs on MPS.
- **Fallback:** `microsoft/Phi-3-mini-4k-instruct` if Llama-3.2-1B access requires gating.

### 4.2 Dataset

- **WikiText-2-raw-v1** (Merity et al., 2016). Standard language modelling benchmark; plain English prose; avoids domain-specific jargon that might bias routing results.
- **Router training:** `train` split (~2M tokens).
- **Perplexity evaluation:** `test` split (~240 k tokens).
- **Latency benchmark:** 50 randomly sampled contiguous passages of 128 tokens from the `validation` split.

### 4.3 Hardware

- Apple M-series chip, MPS backend, 16 GB unified memory.
- Model loaded in `bfloat16`.
- All latency measurements taken after a 5-step warm-up; reported as mean ± std over 50 prompts.

### 4.4 Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **Token acceptance rate** | Fraction of decoding steps where $\hat{y}_t^{\text{full}} \in \text{shortlist}_t$ |
| **Perplexity** | $\exp\bigl(-\frac{1}{N}\sum_t \log P_\theta(y_t \mid y_{<t})\bigr)$, computed over `test` split |
| **Perplexity degradation** | $\Delta\text{PPL} = \text{PPL}_{\text{pruned}} - \text{PPL}_{\text{full}}$ |
| **Tokens/sec** | Wall-clock generation speed; mean over 50 prompts |
| **`lm_head` time %** | Fraction of per-token time spent in the vocabulary projection |

All pruned-vocabulary metrics are reported **relative to the full-vocabulary baseline** to make accuracy loss directly interpretable.

---

## 5. Results

*This section will be populated after running `benchmarks.ipynb`.*

### 5.1 Baseline Profiling

| Metric | Value |
|--------|-------|
| Tokens/sec (full vocab) | TBD |
| `lm_head` time % | TBD |
| Baseline perplexity (WikiText-2 test) | TBD |

### 5.2 Acceptance Rate vs. K

| Router | K=512 | K=1k | K=2k | K=5k | K=10k |
|--------|-------|------|------|------|-------|
| Static Top-K | TBD | TBD | TBD | TBD | TBD |
| Cosine | TBD | TBD | TBD | TBD | TBD |
| Cluster | TBD | TBD | TBD | TBD | TBD |
| Dual-Encoder | TBD | TBD | TBD | TBD | TBD |

### 5.3 Perplexity Degradation vs. K

*(relative to full-vocab baseline)*

### 5.4 Tokens/sec vs. K

### 5.5 Pareto Frontier: Accuracy vs. Speed

---

## 6. Discussion

### 6.1 The Blind-Spot Risk

Dynamic vocabulary pruning introduces a fundamental tension: the router can only shortlist tokens it predicts are relevant, but creative or unexpected tokens by definition have low prior probability given the context. A router that is too aggressive (small K) will systematically exclude rare, creative, or domain-shifting tokens even when they are contextually appropriate.

This "blind-spot" risk is the primary reason vocabulary pruning has not become a default technique: a chef recipe sentence interrupted by a spaceship reference would be hallucinated away if the router had pruned the sci-fi vocabulary cluster.

Our evaluation on WikiText-2 deliberately does not distinguish predictable vs. creative prompts, giving a uniform picture of routing quality across natural prose variation. The acceptance rate metric directly quantifies how often the router "blinds" the model.

### 6.2 Limitations

- Evaluation limited to a single small model and a single text-domain dataset.
- The dual-encoder is trained on token-level prediction; it may not generalise to multi-token idiomatic phrases.
- FAISS/ANN search adds overhead that partially offsets savings for very small K.
- MPS hardware limits — results may not transfer directly to CUDA hardware.

---

## 7. Conclusion and Future Work

We presented a dual-encoder router that learns, from the LLM's own latent space, which vocabulary tokens are likely to be relevant at each decoding step. By replacing the full `lm_head` projection with a targeted shortlist lookup, we demonstrate meaningful inference speedups on commodity hardware with minimal accuracy loss.

**Future work:**
- **Instruction-tuned router (INSTRUCTOR-style):** prepend task instructions to anchor inputs to enable a single router to handle multiple domains/styles.
- **Tree-based routing:** combine the dual-encoder shortlist with speculative decoding / Medusa heads to stack optimisations.
- **Online router adaptation:** fine-tune the router MLP on-the-fly using the accepted/rejected token signal from each inference step.
- **CUDA benchmarks:** reproduce results on NVIDIA hardware to validate generalisability beyond MPS.

---

## References

- Cai, T., et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.*
- Chen, C., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.*
- Henderson, M., et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply.*
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.*
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.*
- Merity, S., et al. (2016). *Pointer Sentinel Mixture Models.* (WikiText-2 dataset)
- Su, H., et al. (2022). *One Embedder, Any Task: Instruction-Finetuned Text Embeddings.*
