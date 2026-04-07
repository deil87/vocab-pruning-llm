"""All evaluation functions: latency benchmark + perplexity for every router."""
import math
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.config import CFG
from src.model_utils import DEVICE


def _sync(device=None):
    """Synchronise the given device (or DEVICE if not specified)."""
    dev = device if device is not None else DEVICE
    if str(dev).startswith("mps"):
        torch.mps.synchronize()
    elif str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)


# ── Generic router evaluator ──────────────────────────────────────────────────

def evaluate_router(
    model,
    bench_prompts: List[torch.Tensor],
    router_fn: Callable[[torch.Tensor, int], torch.Tensor],
    k: int,
    n_prompts: int = CFG.n_bench_prompts,
    max_new_tokens: int = CFG.bench_max_new_tokens,
) -> Dict:
    """
    Greedy decode with a pruned vocabulary supplied by router_fn(hidden, k).

    router_fn receives hidden[0] — a [d] tensor on DEVICE (float16/bfloat16).
    Routers that previously required a CPU float32 tensor now accept GPU tensors
    directly; the fast-path router_fn (FusedRouter) never moves data off-GPU.
    Legacy router_fns that still need CPU can call .cpu() internally.
    Returns latency + acceptance rate metrics.
    """
    accepted, total_steps = 0, 0
    total_times, lmhead_times = [], []

    for prompt in tqdm(bench_prompts[:n_prompts], desc=f"K={k}", leave=False):
        input_ids = prompt.unsqueeze(0).to(DEVICE)
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                t_start = time.perf_counter()

                out = model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1][:, -1, :]   # [1, d] on DEVICE
                past_key_values = out.past_key_values

                lm_dev = model.lm_head.weight.device
                full_logits = hidden.to(lm_dev) @ model.lm_head.weight.T
                gold_token = full_logits[0].argmax().item()

                # Pass GPU tensor directly — no CPU round-trip (Change 1)
                shortlist = router_fn(hidden[0], k)

                pruned_weight = model.lm_head.weight[shortlist.to(lm_dev)]  # on lm_dev
                t_lm0 = time.perf_counter()
                pruned_logits = hidden.to(lm_dev) @ pruned_weight.T
                _sync(lm_dev)
                t_lm1 = time.perf_counter()

                pruned_best = shortlist[pruned_logits[0].argmax().item()].item()
                accepted += int(pruned_best == gold_token)
                total_steps += 1

                input_ids = torch.tensor([[gold_token]], device=DEVICE)
                _sync(DEVICE)
                total_times.append(time.perf_counter() - t_start)
                lmhead_times.append(t_lm1 - t_lm0)

    arr_total = np.array(total_times)
    arr_lm = np.array(lmhead_times)
    return {
        "k": k,
        "acceptance_rate": accepted / total_steps,
        "tokens_per_sec": 1.0 / arr_total.mean(),
        "lmhead_frac": arr_lm.mean() / arr_total.mean(),
        "mean_total_ms": arr_total.mean() * 1000,
        "mean_lmhead_ms": arr_lm.mean() * 1000,
    }


# ── Generic perplexity ────────────────────────────────────────────────────────

def compute_perplexity(
    model,
    tokenizer,
    raw_dataset,
    router_fn: Callable[[torch.Tensor, int], torch.Tensor],
    k: int,
    max_tokens: int = CFG.ppl_max_tokens,
) -> float:
    lines = [l for l in raw_dataset["test"]["text"] if len(l.strip()) > 20]
    tokens = torch.cat([tokenizer.encode(l, return_tensors="pt")[0] for l in lines])[:max_tokens]
    input_ids = tokens.unsqueeze(0).to(DEVICE)
    stride, seq_len, nlls = 512, input_ids.size(1), []

    with torch.no_grad():
        for begin in tqdm(range(0, seq_len - 1, stride), desc=f"PPL K={k}", leave=False):
            end = min(begin + stride, seq_len - 1)
            chunk = input_ids[:, begin : end + 1]
            out = model(input_ids=chunk, output_hidden_states=True)
            hiddens = out.hidden_states[-1][:, :-1, :]
            targets = chunk[:, 1:]

            for t in range(hiddens.size(1)):
                h = hiddens[0, t].float().cpu()
                shortlist = router_fn(h, k)
                pruned_w = model.lm_head.weight[shortlist].float().cpu()
                logits = h @ pruned_w.T
                target = targets[0, t].item()
                if target in shortlist.tolist():
                    local_idx = (shortlist == target).nonzero(as_tuple=True)[0][0].item()
                    nll = -F.log_softmax(logits, dim=-1)[local_idx].item()
                else:
                    nll = -math.log(1e-9)
                nlls.append(nll)

    return math.exp(np.mean(nlls))


# ── Vectorised cosine perplexity ──────────────────────────────────────────────

def compute_perplexity_cosine(
    model,
    tokenizer,
    raw_dataset,
    lm_head_norm_dev: torch.Tensor,
    k: int,
    max_tokens: int = CFG.ppl_max_tokens,
) -> float:
    """
    Vectorised cosine-router perplexity.
    Instead of calling router_fn once per token position (O(T) matmuls),
    batches all positions in a stride chunk into a single [T, V] matmul on-device.
    lm_head_norm_dev: F.normalize(lm_head.weight.float(), dim=-1) [V, d] on DEVICE.
    """
    lines = [l for l in raw_dataset["test"]["text"] if len(l.strip()) > 20]
    tokens = torch.cat([tokenizer.encode(l, return_tensors="pt")[0] for l in lines])[:max_tokens]
    input_ids = tokens.unsqueeze(0).to(DEVICE)
    stride, seq_len, nlls = 512, input_ids.size(1), []

    with torch.no_grad():
        for begin in tqdm(range(0, seq_len - 1, stride), desc=f"PPL cosine K={k}", leave=False):
            end = min(begin + stride, seq_len - 1)
            chunk = input_ids[:, begin : end + 1]
            out = model(input_ids=chunk, output_hidden_states=True)
            hiddens = out.hidden_states[-1][:, :-1, :]   # [1, T, d]
            targets = chunk[:, 1:]                        # [1, T]

            # Batch cosine sim: [T, d] @ [d, V] → [T, V] — one MPS matmul
            h_all = hiddens[0].float()                    # [T, d] on DEVICE
            h_norm = F.normalize(h_all, dim=-1)           # [T, d]
            sims = h_norm @ lm_head_norm_dev.T            # [T, V] on DEVICE
            topk_vals, topk_idx = sims.topk(k, dim=1)    # [T, k] each

            # NLL per position
            T = hiddens.size(1)
            for t in range(T):
                shortlist = topk_idx[t]                   # [k] on DEVICE
                h_t = h_all[t]                            # [d] on lm_head device
                pruned_w = model.lm_head.weight.float()[shortlist.to(model.lm_head.weight.device)]  # [k, d] on lm_head device
                logits = h_t.to(pruned_w.device) @ pruned_w.T  # [k] on lm_head device
                target = targets[0, t].item()

                # Check if target is in shortlist
                match = (shortlist == target).nonzero(as_tuple=True)[0]
                if match.numel() > 0:
                    local_idx = match[0].item()
                    nll = -F.log_softmax(logits, dim=-1)[local_idx].item()
                else:
                    nll = -math.log(1e-9)
                nlls.append(nll)

    return math.exp(np.mean(nlls))

def evaluate_attention_router(
    model,
    bench_prompts: List[torch.Tensor],
    attn_shortlist_fn: Callable,   # (hidden, k, seq_ids, attn_weights) -> LongTensor
    k: int,
    n_prompts: int = CFG.n_bench_prompts,
    max_new_tokens: int = CFG.bench_max_new_tokens,
) -> Dict:
    accepted, total_steps = 0, 0
    total_times, lmhead_times = [], []

    for prompt in tqdm(bench_prompts[:n_prompts], desc=f"Attn K={k}", leave=False):
        input_ids = prompt.unsqueeze(0).to(DEVICE)
        past_key_values = None
        seq_ids = prompt.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                t_start = time.perf_counter()

                out = model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                hidden = out.hidden_states[-1][:, -1, :]
                past_key_values = out.past_key_values
                h_cpu = hidden[0].float().cpu()

                lm_dev = model.lm_head.weight.device
                full_logits = hidden.to(lm_dev) @ model.lm_head.weight.T
                gold_token = full_logits[0].argmax().item()

                shortlist = attn_shortlist_fn(h_cpu, k, seq_ids, out.attentions)

                pruned_weight = model.lm_head.weight[shortlist]  # on lm_dev
                t_lm0 = time.perf_counter()
                pruned_logits = hidden.to(lm_dev) @ pruned_weight.T
                _sync(lm_dev)
                t_lm1 = time.perf_counter()

                pruned_best = shortlist[pruned_logits[0].argmax().item()].item()
                accepted += int(pruned_best == gold_token)
                total_steps += 1

                input_ids = torch.tensor([[gold_token]], device=DEVICE)
                seq_ids = torch.cat([seq_ids, torch.tensor([gold_token])])

                _sync(DEVICE)
                total_times.append(time.perf_counter() - t_start)
                lmhead_times.append(t_lm1 - t_lm0)

    arr_total = np.array(total_times)
    arr_lm = np.array(lmhead_times)
    return {
        "k": k,
        "acceptance_rate": accepted / total_steps,
        "tokens_per_sec": 1.0 / arr_total.mean(),
        "lmhead_frac": arr_lm.mean() / arr_total.mean(),
        "mean_total_ms": arr_total.mean() * 1000,
        "mean_lmhead_ms": arr_lm.mean() * 1000,
    }


def compute_perplexity_attention(
    model,
    tokenizer,
    raw_dataset,
    attn_shortlist_fn: Callable,
    k: int,
    max_tokens: int = CFG.ppl_max_tokens,
) -> float:
    lines = [l for l in raw_dataset["test"]["text"] if len(l.strip()) > 20]
    tokens = torch.cat([tokenizer.encode(l, return_tensors="pt")[0] for l in lines])[:max_tokens]
    input_ids = tokens.unsqueeze(0).to(DEVICE)
    stride, seq_len, nlls = 512, input_ids.size(1), []

    with torch.no_grad():
        for begin in tqdm(range(0, seq_len - 1, stride), desc=f"PPL Attn K={k}", leave=False):
            end = min(begin + stride, seq_len - 1)
            chunk = input_ids[:, begin : end + 1]
            out = model(input_ids=chunk, output_hidden_states=True, output_attentions=True)
            hiddens = out.hidden_states[-1][:, :-1, :]
            targets = chunk[:, 1:]
            chunk_ids = chunk[0].cpu()

            for t in range(hiddens.size(1)):
                h = hiddens[0, t].float().cpu()
                attn_slice = tuple(a[:, :, t : t + 1, : t + 1] for a in out.attentions)
                shortlist = attn_shortlist_fn(h, k, chunk_ids[: t + 1], attn_slice)
                pruned_w = model.lm_head.weight[shortlist].float().cpu()
                logits = h @ pruned_w.T
                target = targets[0, t].item()
                if target in shortlist.tolist():
                    local_idx = (shortlist == target).nonzero(as_tuple=True)[0][0].item()
                    nll = -F.log_softmax(logits, dim=-1)[local_idx].item()
                else:
                    nll = -math.log(1e-9)
                nlls.append(nll)

    return math.exp(np.mean(nlls))


# ── Prefetch + refresh evaluator ──────────────────────────────────────────────

def evaluate_prefetch_router(
    model,
    bench_prompts: List[torch.Tensor],
    prefetch_fn: Callable,    # (prompt_ids) -> LongTensor shortlist
    refresh_fn: Callable,     # (shortlist, h_T) -> LongTensor shortlist
    n_prompts: int = CFG.n_bench_prompts,
    max_new_tokens: int = CFG.bench_max_new_tokens,
    refresh_every: int = CFG.refresh_every,
) -> Dict:
    accepted, total_steps = 0, 0
    total_times, lmhead_times = [], []
    all_shortlist_sizes = []

    for prompt in tqdm(bench_prompts[:n_prompts], desc="Prefetch+Refresh"):
        shortlist = prefetch_fn(prompt)
        input_ids = prompt.unsqueeze(0).to(DEVICE)
        past_key_values = None
        generated_count = 0

        with torch.no_grad():
            for step in range(max_new_tokens):
                t_start = time.perf_counter()

                out = model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1][:, -1, :]   # [1, d] on DEVICE
                past_key_values = out.past_key_values
                h_gpu = hidden[0]  # [d] on DEVICE — no CPU copy (Change 1)

                lm_dev = model.lm_head.weight.device
                full_logits = hidden.to(lm_dev) @ model.lm_head.weight.T
                gold_token = full_logits[0].argmax().item()

                if generated_count > 0 and generated_count % refresh_every == 0:
                    shortlist = refresh_fn(shortlist, h_gpu)

                pruned_weight = model.lm_head.weight[shortlist.to(lm_dev)]  # on lm_dev
                t_lm0 = time.perf_counter()
                pruned_logits = hidden.to(lm_dev) @ pruned_weight.T
                _sync(lm_dev)
                t_lm1 = time.perf_counter()

                pruned_best = shortlist[pruned_logits[0].argmax().item()].item()
                accepted += int(pruned_best == gold_token)
                total_steps += 1
                all_shortlist_sizes.append(len(shortlist))

                input_ids = torch.tensor([[gold_token]], device=DEVICE)
                generated_count += 1

                _sync(DEVICE)
                total_times.append(time.perf_counter() - t_start)
                lmhead_times.append(t_lm1 - t_lm0)

    arr_total = np.array(total_times)
    arr_lm = np.array(lmhead_times)
    return {
        "k": int(np.mean(all_shortlist_sizes)),
        "acceptance_rate": accepted / total_steps,
        "tokens_per_sec": 1.0 / arr_total.mean(),
        "lmhead_frac": arr_lm.mean() / arr_total.mean(),
        "mean_total_ms": arr_total.mean() * 1000,
        "mean_lmhead_ms": arr_lm.mean() * 1000,
        "mean_shortlist_size": float(np.mean(all_shortlist_sizes)),
        "shortlist_sizes": all_shortlist_sizes,
    }


def compute_perplexity_prefetch(
    model,
    tokenizer,
    raw_dataset,
    prefetch_fn: Callable,
    refresh_fn: Callable,
    max_tokens: int = CFG.ppl_max_tokens,
    refresh_every: int = CFG.refresh_every,
) -> float:
    lines = [l for l in raw_dataset["test"]["text"] if len(l.strip()) > 20]
    tokens = torch.cat([tokenizer.encode(l, return_tensors="pt")[0] for l in lines])[:max_tokens]
    input_ids = tokens.unsqueeze(0).to(DEVICE)
    stride, seq_len, nlls = 512, input_ids.size(1), []

    with torch.no_grad():
        for begin in tqdm(range(0, seq_len - 1, stride), desc="PPL prefetch", leave=False):
            end = min(begin + stride, seq_len - 1)
            chunk = input_ids[:, begin : end + 1]
            prompt_chunk = chunk[0, :-1]
            shortlist = prefetch_fn(prompt_chunk)

            out = model(input_ids=chunk, output_hidden_states=True)
            hiddens = out.hidden_states[-1][:, :-1, :]
            targets = chunk[:, 1:]

            for t in range(hiddens.size(1)):
                if t > 0 and t % refresh_every == 0:
                    shortlist = refresh_fn(shortlist, hiddens[0, t].float().cpu())

                h = hiddens[0, t].float().cpu()
                pruned_w = model.lm_head.weight[shortlist].float().cpu()
                logits = h @ pruned_w.T
                target = targets[0, t].item()
                if target in shortlist.tolist():
                    local_idx = (shortlist == target).nonzero(as_tuple=True)[0][0].item()
                    nll = -F.log_softmax(logits, dim=-1)[local_idx].item()
                else:
                    nll = -math.log(1e-9)
                nlls.append(nll)

    return math.exp(np.mean(nlls))


# ── Speculative decoding ──────────────────────────────────────────────────────

def _make_combined_draft(
    h_T: torch.Tensor,
    input_ids_seq: torch.Tensor,
    attn_weights: tuple,
    graph_edges: torch.Tensor,
    lm_head_norm_dev: torch.Tensor,
    lm_head_weight_cpu: torch.Tensor,
    draft_len: int = CFG.spec_draft_len,
) -> Tuple[List[int], List[float]]:
    """
    Generate a draft chain using combined Attention + MLP Graph signal.
    lm_head_norm_dev: F.normalize(lm_head.weight.float(), dim=-1) [V, d] on DEVICE.
    Token 1: combined shortlist from h_T.
    Tokens 2..D: 1-hop graph walk from previous draft token.
    """
    draft_tokens: List[int] = []
    draft_scores: List[float] = []

    # ── Draft token 1 ──
    last_layer_attn = attn_weights[-1][0]                       # [H, T_q, T_k]
    attn_score = last_layer_attn[:, -1, :].mean(0).float().cpu()
    n_pos = min(CFG.attn_top_positions, attn_score.shape[0])
    top_pos = attn_score.topk(n_pos).indices
    attended_ids = input_ids_seq.cpu()[top_pos]                  # [n_pos] CPU

    # On-device attention neighbour lookup
    attn_nbrs = lm_head_norm_dev[attended_ids.to(DEVICE)] @ lm_head_norm_dev.T  # [n_pos, V]
    attn_nbr_ids = attn_nbrs.topk(CFG.attn_neighbour_k, dim=1).indices.cpu().reshape(-1)

    # On-device cosine anchor
    h_dev = h_T.to(DEVICE).float().unsqueeze(0)
    h_norm = F.normalize(h_dev, dim=-1)
    cos_sims_h = (h_norm @ lm_head_norm_dev.T).squeeze(0)       # [V] on DEVICE
    anchor_ids = cos_sims_h.topk(CFG.graph_anchor_k).indices.cpu()
    graph_nbr_ids = graph_edges[anchor_ids].reshape(-1).long()

    shortlist_1 = torch.cat([attended_ids, attn_nbr_ids, anchor_ids, graph_nbr_ids]).unique()
    h_cpu = h_T.float().cpu()
    lm_w = lm_head_weight_cpu[shortlist_1]                      # [S1, d]
    logits_1 = h_cpu @ lm_w.T                                   # [S1]
    best_local = logits_1.argmax().item()
    token_1 = shortlist_1[best_local].item()
    draft_tokens.append(token_1)
    draft_scores.append(logits_1[best_local].item())

    # ── Draft tokens 2..D: 1-hop graph walk ──
    prev_token = token_1
    for _ in range(1, draft_len):
        nbrs = graph_edges[prev_token].long()                   # [M]
        lm_w2 = lm_head_weight_cpu[nbrs]                       # [M, d]
        logits_d = h_cpu @ lm_w2.T                             # [M]
        best = logits_d.argmax().item()
        next_token = nbrs[best].item()
        draft_tokens.append(next_token)
        draft_scores.append(logits_d[best].item())
        prev_token = next_token

    return draft_tokens, draft_scores


def evaluate_speculative(
    model,
    bench_prompts: List[torch.Tensor],
    graph_edges: torch.Tensor,
    lm_head_norm_dev: torch.Tensor,
    lm_head_weight_cpu: torch.Tensor,
    baseline_tps: float,
    n_prompts: int = CFG.spec_n_prompts,
    max_new_tokens: int = CFG.spec_max_new_tokens,
    draft_len: int = CFG.spec_draft_len,
) -> Dict:
    total_accepted = 0
    total_spec_steps = 0
    total_draft_positions = 0
    total_wall_time = 0.0

    for prompt in tqdm(bench_prompts[:n_prompts], desc="Spec decoding"):
        input_ids = prompt.unsqueeze(0).to(DEVICE)
        seq_ids = prompt.clone()
        generated = 0

        while generated < max_new_tokens:
            t_spec_start = time.perf_counter()

            with torch.no_grad():
                out_v = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=False,
                )
            h_T = out_v.hidden_states[-1][0, -1, :].float().cpu()

            remaining = max_new_tokens - generated
            actual_draft_len = min(draft_len, remaining)
            draft_tokens, _ = _make_combined_draft(
                h_T, seq_ids, out_v.attentions,
                graph_edges, lm_head_norm_dev, lm_head_weight_cpu,
                draft_len=actual_draft_len,
            )

            draft_tensor = torch.tensor(draft_tokens, device=DEVICE)
            verify_input = torch.cat([input_ids[0], draft_tensor]).unsqueeze(0)

            with torch.no_grad():
                out_check = model(input_ids=verify_input, use_cache=False)

            T = input_ids.shape[1]
            # [D+1, V] — one extra position for bonus token
            verify_logits = out_check.logits[0, T - 1 : T + actual_draft_len, :]
            verifier_tokens = verify_logits[:actual_draft_len].argmax(dim=-1).cpu().tolist()

            accepted_len = 0
            for d_tok, v_tok in zip(draft_tokens, verifier_tokens):
                if d_tok == v_tok:
                    accepted_len += 1
                else:
                    break
            accepted_len = max(accepted_len, 1)

            if accepted_len == actual_draft_len:
                bonus_token = verify_logits[actual_draft_len].argmax().item()
                accepted_tokens = draft_tokens[:accepted_len] + [bonus_token]
                accepted_len += 1
            else:
                accepted_tokens = draft_tokens[: accepted_len - 1] + [verifier_tokens[accepted_len - 1]]

            new_ids = torch.tensor(accepted_tokens, device=DEVICE).unsqueeze(0)
            input_ids = torch.cat([input_ids, new_ids], dim=1)
            seq_ids = torch.cat([seq_ids, torch.tensor(accepted_tokens)])
            generated += len(accepted_tokens)

            total_accepted += len(accepted_tokens)
            total_spec_steps += 1
            total_draft_positions += actual_draft_len
            total_wall_time += time.perf_counter() - t_spec_start

    tokens_per_call = total_accepted / total_spec_steps
    effective_tps = total_accepted / total_wall_time
    draft_accept_rate = total_accepted / total_draft_positions

    print(f"  Tokens/call:      {tokens_per_call:.3f}  (draft_len={draft_len})")
    print(f"  Effective TPS:    {effective_tps:.2f}")
    print(f"  Draft accept rate:{draft_accept_rate:.3f}")
    print(f"  Speedup vs base:  {effective_tps/baseline_tps:.2f}x")

    return {
        "draft_len": draft_len,
        "tokens_per_call": tokens_per_call,
        "effective_tps": effective_tps,
        "draft_acceptance_rate": draft_accept_rate,
        "speedup_vs_baseline": effective_tps / baseline_tps,
    }


def evaluate_cosine_speculative(
    model,
    bench_prompts: List[torch.Tensor],
    lm_head_norm_dev: torch.Tensor,
    baseline_tps: float,
    n_prompts: int = CFG.spec_n_prompts,
    max_new_tokens: int = CFG.spec_max_new_tokens,
    draft_len: int = CFG.spec_draft_len,
) -> Dict:
    """Naive speculative decoding baseline: draft = top-1 cosine token repeated D times.
    lm_head_norm_dev: F.normalize(lm_head.weight.float(), dim=-1) [V, d] on DEVICE.
    """
    total_accepted, total_spec_steps = 0, 0
    total_draft_positions = 0
    total_wall_time = 0.0

    for prompt in tqdm(bench_prompts[:n_prompts], desc="Cosine spec baseline"):
        input_ids = prompt.unsqueeze(0).to(DEVICE)
        generated = 0

        while generated < max_new_tokens:
            t0 = time.perf_counter()

            with torch.no_grad():
                out_v = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            h_T_dev = out_v.hidden_states[-1][0, -1, :].float()   # [d] on DEVICE

            remaining = max_new_tokens - generated
            actual_draft_len = min(draft_len, remaining)

            h_norm = F.normalize(h_T_dev.unsqueeze(0), dim=-1)
            top1 = (h_norm @ lm_head_norm_dev.T).squeeze(0).argmax().item()
            draft_tokens = [top1] * actual_draft_len

            draft_tensor = torch.tensor(draft_tokens, device=DEVICE)
            verify_input = torch.cat([input_ids[0], draft_tensor]).unsqueeze(0)
            with torch.no_grad():
                out_check = model(input_ids=verify_input, use_cache=False)

            T = input_ids.shape[1]
            verify_logits = out_check.logits[0, T - 1 : T + actual_draft_len, :]  # [D+1, V]
            verifier_tokens = verify_logits.argmax(dim=-1).cpu().tolist()

            accepted_len = 0
            for d_tok, v_tok in zip(draft_tokens, verifier_tokens[:actual_draft_len]):
                if d_tok == v_tok:
                    accepted_len += 1
                else:
                    break
            accepted_len = max(accepted_len, 1)

            if accepted_len == actual_draft_len:
                bonus_token = verify_logits[actual_draft_len].argmax().item()
                accepted_token_ids = verify_logits[:actual_draft_len].argmax(dim=-1).tolist() + [bonus_token]
                accepted_len += 1
            else:
                accepted_token_ids = (
                    verify_logits[: accepted_len - 1].argmax(dim=-1).tolist()
                    + [verifier_tokens[accepted_len - 1]]
                )

            new_ids = torch.tensor(accepted_token_ids, device=DEVICE).unsqueeze(0)
            input_ids = torch.cat([input_ids, new_ids], dim=1)
            generated += accepted_len

            total_accepted += accepted_len
            total_spec_steps += 1
            total_draft_positions += actual_draft_len
            total_wall_time += time.perf_counter() - t0

    tokens_per_call = total_accepted / total_spec_steps
    effective_tps = total_accepted / total_wall_time
    draft_accept_rate = total_accepted / total_draft_positions

    print(f"  [Cosine] Tokens/call: {tokens_per_call:.3f}  TPS: {effective_tps:.2f}  "
          f"Draft acc: {draft_accept_rate:.3f}")

    return {
        "draft_len": draft_len,
        "tokens_per_call": tokens_per_call,
        "effective_tps": effective_tps,
        "draft_acceptance_rate": draft_accept_rate,
        "speedup_vs_baseline": effective_tps / baseline_tps,
    }
