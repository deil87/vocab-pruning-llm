# Project
Research and development around the topic of optimisation of modern transformer based models

## Goals
- research the topic of optimisation of the inference time for LLMs
- build a notebook/code that runs benchmarks to compare alternative directions for optimisation

---

## Architecture: Dual-Encoder Router

At each decoding step, the LLM's hidden state ("vibe") is used to retrieve a set of
semantically plausible sentence completions from a prebuilt index. The union of tokens
across those retrieved completions forms a pruned vocabulary shortlist. The `lm_head`
matrix multiplication is then performed only over that shortlist.

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

## Research Paths

### Path A — Vocabulary Pruning (current)
Use retrieved completions to build a pruned vocabulary shortlist at each decoding step.
Run `lm_head` only over that shortlist. No speculative decoding needed.
- Simpler, benchmarkable immediately
- Metric: tokens/sec speedup, perplexity degradation, token acceptance rate vs. full-vocab baseline

### Path B — Speculative Decoding via Retrieval (future)
Feed the retrieved sentence completions directly to an Executive model as draft candidates,
replacing the small sequential draft model used in standard speculative decoding.
- Retrieved completions are semantically coherent multi-token sequences grounded in real text
- Stronger prior than a blind small draft model
- Can be combined with Path A (pruned vocab) or used standalone
- Metric: acceptance rate vs. standard speculative decoding, tokens/sec vs. baseline

