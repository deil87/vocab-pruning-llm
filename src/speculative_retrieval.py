"""
Retrieval-based speculative decoding (Option B).

Architecture
------------
Offline:
  1. Sample N completions from WikiText-2 train (windows of `completion_len` tokens).
  2. For each completion, run the LLM on the preceding prefix and extract h_T
     (last hidden state at the split point). L2-normalise → completion index.
  3. Save (token_seqs, index_norm) to disk.

Online (per decoding step):
  1. One forward pass on current input_ids (standard SDPA — no output_attentions).
     Extract h_T, L2-normalise.
  2. Dot-product h_T against the index → retrieve top-K completions.
  3. For each of the K candidates, run one verifier pass on [input_ids + draft_tokens].
     Find the longest prefix where draft == verifier greedy tokens.
  4. Commit the best (longest) accepted prefix across all K candidates.
     Fall back to the verifier's correction token if no draft token matched.

Cost per step: 1 probe pass + K verifier passes.
No router MLP, no output_attentions, no eager attention penalty.
"""
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.config import CFG
from src.model_utils import DEVICE


# ── Index build / load ────────────────────────────────────────────────────────

def build_retrieval_index(
    model,
    tokenizer,
    raw_dataset,
    n_completions: int = CFG.ret_spec_n_completions,
    completion_len: int = CFG.ret_spec_completion_len,
    prefix_len_cap: int = CFG.prefix_len_cap,
    seed: int = CFG.seed,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Build a completion index from the WikiText-2 train split.

    For each sampled split point s:
      - prefix  = train_tokens[s - prefix_len_cap : s]   (up to prefix_len_cap tokens)
      - completion = train_tokens[s : s + completion_len]
      - index embedding = h_T at position s (last hidden state of prefix forward pass)

    Returns
    -------
    token_seqs  : List[LongTensor[completion_len]]  — the completion token sequences
    index_norm  : Tensor[N, d] float32 on CPU       — L2-normalised prefix embeddings
    """
    index_path: Path = CFG.ret_spec_index_path

    if index_path.exists():
        print(f"Loading retrieval index from {index_path} …")
        saved = torch.load(index_path, weights_only=False)
        token_seqs = saved["token_seqs"]
        index_norm = saved["index_norm"]
        print(f"  Loaded {len(token_seqs):,} completions, index shape {index_norm.shape}")
        return token_seqs, index_norm

    print("Tokenising WikiText-2 train split …")
    train_lines = [l for l in raw_dataset["train"]["text"] if len(l.strip()) > 20]
    train_parts = [tokenizer.encode(l, return_tensors="pt")[0] for l in train_lines]
    train_tokens = torch.cat(train_parts)
    print(f"  Total training tokens: {len(train_tokens):,}")

    min_prefix = max(prefix_len_cap, 16)
    max_start = len(train_tokens) - completion_len - 1
    rng = np.random.default_rng(seed)
    split_points = rng.choice(
        np.arange(min_prefix, max_start),
        size=n_completions,
        replace=False,
    )
    split_points = np.sort(split_points)

    token_seqs: List[torch.Tensor] = []
    embeddings: List[torch.Tensor] = []

    print(f"Encoding {n_completions:,} completions (completion_len={completion_len}) …")
    model.eval()
    with torch.no_grad():
        for sp in tqdm(split_points, desc="Building index"):
            sp = int(sp)
            prefix_ids = train_tokens[sp - prefix_len_cap : sp].unsqueeze(0).to(DEVICE)
            comp_ids = train_tokens[sp : sp + completion_len]

            out = model(input_ids=prefix_ids, output_hidden_states=True)
            h_T = out.hidden_states[-1][0, -1, :].float().cpu()  # [d]

            token_seqs.append(comp_ids.cpu())
            embeddings.append(h_T)

    index_raw = torch.stack(embeddings, dim=0)          # [N, d]
    index_norm = F.normalize(index_raw, dim=-1)          # [N, d]

    torch.save({"token_seqs": token_seqs, "index_norm": index_norm}, index_path)
    print(f"Saved retrieval index → {index_path}  ({index_norm.shape})")
    return token_seqs, index_norm


# ── Per-step retrieval ─────────────────────────────────────────────────────────

def retrieve_completions(
    h_T_norm: torch.Tensor,             # [d] float32 CPU, L2-normalised
    index_norm: torch.Tensor,           # [N, d] float32 CPU
    token_seqs: List[torch.Tensor],
    top_k: int,
) -> List[torch.Tensor]:
    """Return the top_k completion token sequences closest to h_T_norm."""
    sims = index_norm @ h_T_norm        # [N]  — pure dot product (both normalised)
    topk_idx = sims.topk(top_k).indices
    return [token_seqs[i] for i in topk_idx.tolist()]


# ── Single speculative step ────────────────────────────────────────────────────

def speculative_step(
    model,
    input_ids: torch.Tensor,            # [1, T] on DEVICE
    index_norm: torch.Tensor,           # [N, d] CPU
    token_seqs: List[torch.Tensor],
    top_k: int,
    draft_len: int,
) -> Tuple[List[int], int, int]:
    """
    One retrieval-based speculative step.

    Returns
    -------
    accepted_tokens : list of committed token ids (length >= 1)
    n_accepted      : number of draft positions accepted (before bonus)
    n_draft_total   : total draft positions attempted (top_k * draft_len)
    """
    # ── Probe pass: get h_T + KV-cache ──────────────────────────────────────
    with torch.no_grad():
        out_probe = model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
    h_T = out_probe.hidden_states[-1][0, -1, :].float().cpu()
    h_T_norm = F.normalize(h_T, dim=0)
    probe_pkv = out_probe.past_key_values   # KV-cache covering all T prefix tokens

    # ── Retrieve top-K completions ───────────────────────────────────────────
    candidates = retrieve_completions(h_T_norm, index_norm, token_seqs, top_k)

    T = input_ids.shape[1]
    best_tokens: List[int] = []
    best_accepted: int = -1   # -1 so that even n_acc=0 candidates get recorded

    # ── Verify each candidate using the cached prefix ────────────────────────
    # We feed only the draft tokens to the model, reusing the probe KV-cache.
    # This reduces each verifier forward pass from O(T+D) to O(D) attention cost.
    for cand in candidates:
        draft = cand[:draft_len].to(DEVICE)                     # [draft_len]

        with torch.no_grad():
            out_v = model(
                input_ids=draft.unsqueeze(0),   # [1, draft_len] — prefix already cached
                past_key_values=probe_pkv,
                use_cache=False,
            )

        # out_v.logits shape: [1, draft_len, V]
        # position 0 predicts token at T (first draft position)
        # position d predicts token at T+d
        verify_logits = out_v.logits[0]                          # [draft_len, V]
        verifier_greedy = verify_logits.argmax(dim=-1)           # [draft_len]

        # Count how many leading draft tokens the verifier agrees with
        matches = (draft.cpu() == verifier_greedy.cpu())
        n_acc = int(matches.cumprod(0).sum().item())   # longest matching prefix

        if n_acc > best_accepted:
            if n_acc == draft_len:
                # All draft tokens accepted — bonus token is the last verifier prediction
                # (verify_logits[-1] predicts the token after the last draft token)
                bonus = verify_logits[-1].argmax().item()
                committed = verifier_greedy.tolist() + [bonus]
            else:
                # Partial accept (including 0): verifier's correction at first mismatch
                committed = verifier_greedy[:n_acc].tolist() + [verifier_greedy[n_acc].item()]

            best_accepted = n_acc
            best_tokens = committed

    # Guarantee at least 1 token is committed.
    # Fallback: if somehow all candidates produced no verifier output, use probe logit.
    if not best_tokens:
        probe_next = out_probe.logits[0, -1, :].argmax().item()
        best_tokens = [probe_next]
        best_accepted = 0

    n_draft_total = top_k * draft_len
    return best_tokens, best_accepted, n_draft_total


# ── Full evaluation loop ───────────────────────────────────────────────────────

def evaluate_retrieval_speculative(
    model,
    bench_prompts: List[torch.Tensor],
    index_norm: torch.Tensor,
    token_seqs: List[torch.Tensor],
    baseline_tps: float,
    top_k: int = CFG.ret_spec_top_k,
    draft_len: int = CFG.ret_spec_draft_len,
    n_prompts: int = CFG.ret_spec_n_prompts,
    max_new_tokens: int = CFG.ret_spec_max_new_tokens,
) -> Dict:
    """
    Evaluate retrieval-based speculative decoding over bench_prompts.

    Metrics returned
    ----------------
    top_k, draft_len,
    effective_tps          : tokens committed per wall-clock second
    tokens_per_step        : mean tokens committed per speculative step
    draft_acceptance_rate  : mean accepted draft positions / draft_len (excl. bonus)
    verifier_calls_per_tok : verifier forward passes per committed token
    speedup_vs_baseline    : effective_tps / baseline_tps
    """
    total_committed = 0
    total_accepted_draft = 0
    total_draft_positions = 0
    total_steps = 0
    total_wall = 0.0

    for prompt in tqdm(bench_prompts[:n_prompts], desc=f"RetSpec K={top_k} D={draft_len}"):
        input_ids = prompt.unsqueeze(0).to(DEVICE)
        generated = 0

        while generated < max_new_tokens:
            remaining = max_new_tokens - generated
            effective_draft_len = min(draft_len, remaining)

            t0 = time.perf_counter()
            committed, n_acc, n_draft = speculative_step(
                model, input_ids, index_norm, token_seqs,
                top_k=top_k, draft_len=effective_draft_len,
            )
            total_wall += time.perf_counter() - t0

            new_ids = torch.tensor(committed, device=DEVICE).unsqueeze(0)
            input_ids = torch.cat([input_ids, new_ids], dim=1)
            generated += len(committed)

            total_committed += len(committed)
            total_accepted_draft += n_acc
            total_draft_positions += n_draft
            total_steps += 1

    effective_tps = total_committed / total_wall
    tokens_per_step = total_committed / total_steps
    # draft acceptance rate: fraction of individual draft positions that matched
    draft_acc = total_accepted_draft / total_draft_positions if total_draft_positions > 0 else 0.0
    # verifier calls per token: (1 probe + top_k verifier) passes / committed tokens
    verifier_calls_per_tok = (total_steps * (1 + top_k)) / total_committed

    print(f"  top_k={top_k} draft_len={draft_len}")
    print(f"  Effective TPS:          {effective_tps:.2f}  (baseline {baseline_tps:.2f})")
    print(f"  Tokens/step:            {tokens_per_step:.2f}")
    print(f"  Draft acceptance rate:  {draft_acc:.3f}")
    print(f"  Verifier calls/token:   {verifier_calls_per_tok:.2f}")
    print(f"  Speedup vs baseline:    {effective_tps / baseline_tps:.3f}x")

    return {
        "top_k": top_k,
        "draft_len": draft_len,
        "effective_tps": effective_tps,
        "tokens_per_step": tokens_per_step,
        "draft_acceptance_rate": draft_acc,
        "verifier_calls_per_tok": verifier_calls_per_tok,
        "speedup_vs_baseline": effective_tps / baseline_tps,
        "n_prompts": n_prompts,
    }
