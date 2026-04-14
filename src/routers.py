"""All shortlist / router functions."""
from typing import List, Optional

import torch
import torch.nn.functional as F

from src.config import CFG
from src.model_utils import DEVICE


def _sync():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ── Static Top-K ──────────────────────────────────────────────────────────────

def build_static_index(lm_head_weight: torch.Tensor) -> torch.Tensor:
    """Return token indices sorted by lm_head weight norm (descending)."""
    norms = lm_head_weight.norm(dim=-1)          # [V]
    return norms.argsort(descending=True)         # [V]


def get_static_shortlist(static_sorted_idx: torch.Tensor, k: int) -> torch.Tensor:
    return static_sorted_idx[:k]


# ── Cosine router ─────────────────────────────────────────────────────────────

def get_cosine_shortlist(hidden: torch.Tensor, lm_head_norm_dev: torch.Tensor,
                         k: int) -> torch.Tensor:
    """
    On-device cosine shortlist.
    hidden: [d] — any device/dtype (will be moved to lm_head_norm_dev's device, float32).
    lm_head_norm_dev: F.normalize(lm_head.weight, dim=-1) [V, d] on its device, float32.
    Returns: [k] LongTensor on CPU.
    """
    lm_dev = lm_head_norm_dev.device
    h = hidden.to(lm_dev).float().unsqueeze(0)                # [1, d] float32 on lm_dev
    h_norm = F.normalize(h, dim=-1)                           # [1, d]
    sims = (h_norm @ lm_head_norm_dev.T).squeeze(0)           # [V] on lm_dev
    topk_idx = sims.topk(k).indices.cpu()                     # [k] on CPU
    return topk_idx


# ── Cluster router ────────────────────────────────────────────────────────────

def get_cluster_shortlist(hidden: torch.Tensor, cluster_centers_norm: torch.Tensor,
                          cluster_to_tokens: dict, k: int) -> torch.Tensor:
    """Select clusters by cosine sim to hidden until we have ≥ k tokens."""
    h_norm = F.normalize(hidden.unsqueeze(0), dim=-1)    # [1, d]
    cc_norm = cluster_centers_norm.to(device=hidden.device, dtype=hidden.dtype)
    sims = (h_norm @ cc_norm.T).squeeze(0)               # [C]
    sorted_clusters = sims.argsort(descending=True)
    selected = []
    for c in sorted_clusters.tolist():
        selected.append(cluster_to_tokens[c])
        if sum(len(s) for s in selected) >= k:
            break
    return torch.cat(selected)[:k]


# ── Dual-encoder step-level ───────────────────────────────────────────────────

def get_dual_encoder_shortlist(
    hidden: torch.Tensor,
    router,
    completion_index_proj: torch.Tensor,
    completion_token_lists: List[torch.Tensor],
    lm_head_norm_dev: torch.Tensor,
    k: int,
    # Optional GPU-native acceleration (Change 1 + 2)
    fused_router=None,
    completion_token_tensor: Optional[torch.Tensor] = None,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """
    hidden: [d] — any device/dtype.  Kept on GPU when fused_router is provided.
    lm_head_norm_dev: F.normalize(lm_head.weight.float(), dim=-1) [V, d] on DEVICE.

    Fast path (fused_router + completion_token_tensor provided):
      - No GPU→CPU→GPU round-trip for h_T
      - GPU-native token union via boolean mask scatter
      - MLP + similarity fused (Triton if available, else PyTorch)

    Slow path (legacy): CPU list comprehension, unchanged behaviour.
    """
    if fused_router is not None and completion_token_tensor is not None:
        # ── Fast path ────────────────────────────────────────────────────────
        from src.router_kernel import gpu_token_union
        h_dev = hidden.to(DEVICE).float()
        if h_dev.dim() == 2:
            h_dev = h_dev.squeeze(0)
        with torch.no_grad():
            top_idx = fused_router(h_dev)   # [n_retrieve] on GPU, no CPU round-trip
        vs = vocab_size or lm_head_norm_dev.shape[0]
        return gpu_token_union(
            completion_token_tensor, top_idx, vs, k, lm_head_norm_dev, h_dev
        )

    # ── Slow / legacy path ────────────────────────────────────────────────────
    h_dev = hidden.to(DEVICE).float()
    with torch.no_grad():
        q = router(h_dev.unsqueeze(0)).squeeze(0)              # [d_r] on DEVICE
        sims = completion_index_proj @ q                       # [N] on DEVICE
        top_idx = sims.topk(CFG.n_retrieve).indices.cpu()     # [N_RETRIEVE] CPU

    token_id_sets = [completion_token_lists[i.item()] for i in top_idx]
    all_ids = torch.cat(token_id_sets).unique()

    if len(all_ids) >= k:
        return all_ids[:k]

    # Pad with cosine-nearest on device
    lm_dev = lm_head_norm_dev.device
    h_dev = hidden.to(lm_dev).float().unsqueeze(0)
    h_norm = F.normalize(h_dev, dim=-1)
    cos_sims = (h_norm @ lm_head_norm_dev.T).squeeze(0).cpu()
    cos_sims[all_ids] = -1e9
    n_pad = k - len(all_ids)
    pad_ids = cos_sims.topk(n_pad).indices
    return torch.cat([all_ids, pad_ids])


# ── Dual-encoder prefetch + refresh ──────────────────────────────────────────

def prefetch_shortlist(
    model,
    prompt_ids: torch.Tensor,
    router,
    completion_index_proj: torch.Tensor,
    completion_token_lists: List[torch.Tensor],
    # Optional GPU-native acceleration
    fused_router=None,
    completion_token_tensor: Optional[torch.Tensor] = None,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Single forward pass over full prompt → batch RouterMLP → union of token IDs.
    Returns 1-D LongTensor.

    Fast path: if fused_router + completion_token_tensor are provided, the
    per-position top-K retrieval stays on GPU and the union is built via
    boolean mask scatter (no CPU loop, no round-trips).
    """
    router.eval()
    with torch.no_grad():
        ids = prompt_ids.unsqueeze(0).to(DEVICE)
        out = model(input_ids=ids, output_hidden_states=True)
        all_h = out.hidden_states[-1][0].float().to(DEVICE)   # [T, d] on DEVICE

        if fused_router is not None and completion_token_tensor is not None:
            # Fast path: batched matmul stays on GPU
            q_all = router(all_h)                              # [T, d_r] on DEVICE
            sims = q_all @ completion_index_proj.T             # [T, N] on DEVICE
            top_idx_gpu = sims.topk(CFG.n_retrieve, dim=1).indices  # [T, n_retrieve] GPU

            vs = vocab_size or completion_token_tensor.shape[0]
            mask = torch.zeros(
                completion_index_proj.shape[0] if vs is None else vs,
                dtype=torch.bool, device=DEVICE
            )
            # Flatten all retrieved completion indices → gather token IDs
            flat_idx = top_idx_gpu.reshape(-1)                 # [T * n_retrieve]
            retrieved = completion_token_tensor[flat_idx]      # [T*n_retrieve, comp_len]
            vs2 = vocab_size or completion_token_tensor.max().item() + 1
            mask2 = torch.zeros(vs2, dtype=torch.bool, device=DEVICE)
            mask2.scatter_(0, retrieved.reshape(-1), True)
            return mask2.nonzero(as_tuple=False).squeeze(1)

        # Legacy path
        queries = router(all_h)                                # [T, d_r] on DEVICE
        sims = queries @ completion_index_proj.T               # [T, N] on DEVICE
        top_idx = sims.topk(CFG.n_retrieve, dim=1).indices.cpu()  # [T, N_RETRIEVE] CPU

    token_id_sets = [
        completion_token_lists[ci.item()]
        for row in top_idx
        for ci in row
    ]
    return torch.cat(token_id_sets).unique()


def refresh_shortlist(
    current_shortlist: torch.Tensor,
    h_T: torch.Tensor,
    router,
    completion_index_proj: torch.Tensor,
    completion_token_lists: List[torch.Tensor],
    # Optional GPU-native acceleration
    fused_router=None,
    completion_token_tensor: Optional[torch.Tensor] = None,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """Union-expand shortlist using h_T from the most recently generated token.

    Fast path: h_T stays on GPU, union built via boolean mask scatter.
    """
    router.eval()
    with torch.no_grad():
        if fused_router is not None and completion_token_tensor is not None:
            h_dev = h_T.to(DEVICE).float()
            if h_dev.dim() == 2:
                h_dev = h_dev.squeeze(0)
            top_idx_gpu = fused_router(h_dev)                  # [n_retrieve] on GPU
            retrieved = completion_token_tensor[top_idx_gpu]   # [n_retrieve, comp_len]
            vs = vocab_size or retrieved.max().item() + 1
            mask = torch.zeros(vs, dtype=torch.bool, device=DEVICE)
            mask.scatter_(0, retrieved.reshape(-1), True)
            new_ids = mask.nonzero(as_tuple=False).squeeze(1)  # on GPU
            # Union with current_shortlist
            cur = current_shortlist.to(DEVICE)
            mask2 = torch.zeros(vs, dtype=torch.bool, device=DEVICE)
            mask2[cur] = True
            mask2[new_ids] = True
            return mask2.nonzero(as_tuple=False).squeeze(1)

        # Legacy path
        q = router(h_T.to(DEVICE).float().unsqueeze(0)).squeeze(0)  # [d_r] on DEVICE
        sims = completion_index_proj @ q                             # [N] on DEVICE
        top_idx = sims.topk(CFG.n_retrieve).indices.cpu()           # [N_RETRIEVE] CPU

    new_ids = torch.cat([completion_token_lists[ci.item()] for ci in top_idx]).unique()
    return torch.cat([current_shortlist, new_ids]).unique()


# ── Attention graph router ────────────────────────────────────────────────────

def get_attention_shortlist(
    hidden: torch.Tensor,
    k: int,
    input_ids_seq: torch.Tensor,
    attn_weights: tuple,
    lm_head_norm_dev: torch.Tensor,
) -> torch.Tensor:
    """
    lm_head_norm_dev: F.normalize(lm_head.weight.float(), dim=-1) [V, d] on DEVICE.
    1. Aggregate last-layer attention → position scores.
    2. Top-ATTN_TOP_POSITIONS positions.
    3. For each: find ATTN_NEIGHBOUR_K cosine-nearest tokens (on device).
    4. Union → k tokens.
    """
    last_layer = attn_weights[-1][0]                              # [H, T_q, T_k]
    attn_score = last_layer[:, -1, :].mean(dim=0).float().cpu()   # [T_k]

    T = attn_score.shape[0]
    n_pos = min(CFG.attn_top_positions, T)
    top_pos = attn_score.topk(n_pos).indices                      # [n_pos]

    ids_cpu = input_ids_seq.cpu()
    attended_token_ids = ids_cpu[top_pos]                         # [n_pos] CPU

    # Token-neighbour lookup on device
    lm_dev = lm_head_norm_dev.device
    attended_embs = lm_head_norm_dev[attended_token_ids.to(lm_dev)]  # [n_pos, d]
    sims = attended_embs @ lm_head_norm_dev.T                        # [n_pos, V]
    nbr_k = min(CFG.attn_neighbour_k, lm_head_norm_dev.shape[0])
    nbr_idx = sims.topk(nbr_k, dim=1).indices.cpu()                  # [n_pos, nbr_k]

    all_ids = torch.cat([attended_token_ids, nbr_idx.reshape(-1)]).unique()

    if len(all_ids) >= k:
        return all_ids[:k]

    h_dev = hidden.to(lm_dev).float().unsqueeze(0)
    h_norm = F.normalize(h_dev, dim=-1)
    cos_sims = (h_norm @ lm_head_norm_dev.T).squeeze(0).cpu()
    cos_sims[all_ids] = -1e9
    pad_ids = cos_sims.topk(k - len(all_ids)).indices
    return torch.cat([all_ids, pad_ids])


# ── Hybrid Attention + MLP Graph router ──────────────────────────────────────
#
# Design goals vs. the original Attention Graph router:
#   1. No full-vocabulary cosine scan per attended position.
#      The original router did [n_pos, d] @ [d, V] per step — O(n_pos * V * d).
#      Here we replace that with a cheap CPU lookup into graph_edges — O(n_pos * M).
#   2. Only the last transformer layer needs to materialise attention weights.
#      All other layers continue to use SDPA/FlashAttention unchanged.
#      Overhead is therefore ~1/num_layers of the full eager penalty.
#
# Algorithm per decode step:
#   attended token IDs  ← top-N positions in mean-head attention of last layer
#   graph neighbours    ← 1-hop walk in MLP transition graph from attended tokens
#   union               ← attended IDs + all graph neighbours
#   pad if < k          ← cosine-nearest from h_T (no additional full-vocab scan
#                          beyond what padding requires, and only when |union| < k)


def register_last_layer_hook(model) -> tuple:
    """
    Register a forward hook on the last transformer layer's self-attention
    module so that attention weights are captured into a shared dict without
    forcing the entire model into eager mode.

    Only the last layer needs to run in eager/non-fused mode to materialise
    attention weights. All other layers keep their default implementation
    (SDPA or FlashAttention).

    Returns:
        (captured_dict, hook_handle)
        captured_dict['attn']:  None | Tensor [H, T_q, T_k] — set after each forward
        hook_handle: call .remove() to detach the hook when done
    """
    captured: dict = {"attn": None}
    last_layer_attn = model.model.layers[-1].self_attn

    # Force only the last attention layer to eager so it materialises weights
    if hasattr(last_layer_attn, "config"):
        # Some model versions expose config on the sub-module
        pass  # can't easily override per-layer; rely on output hook below

    def _hook(module, inputs, output):
        # HuggingFace LlamaAttention returns (attn_output, attn_weights, past_kv)
        # when output_attentions=True *and* eager implementation is used.
        # When the hook fires, output[1] is the weight tensor or None.
        if isinstance(output, (tuple, list)) and len(output) > 1:
            w = output[1]
            if w is not None:
                # w shape: [batch, heads, T_q, T_k]
                captured["attn"] = w[0].detach()  # strip batch dim → [H, T_q, T_k]

    handle = last_layer_attn.register_forward_hook(_hook)
    return captured, handle


def remove_last_layer_hook(handle) -> None:
    """Remove the forward hook registered by register_last_layer_hook."""
    handle.remove()


def get_hybrid_shortlist(
    hidden: torch.Tensor,
    k: int,
    input_ids_seq: torch.Tensor,
    captured_attn: dict,
    graph_edges: torch.Tensor,
    lm_head_norm_dev: torch.Tensor,
) -> torch.Tensor:
    """
    Hybrid Attention + MLP Graph shortlist — no full-vocabulary cosine scan
    for the primary expansion step.

    Args:
        hidden:         [d] hidden state for current step (any device/dtype).
        k:              target shortlist size.
        input_ids_seq:  [T] token IDs for the current sequence so far (CPU).
        captured_attn:  dict with key 'attn' populated by register_last_layer_hook.
                        captured_attn['attn'] is [H, T_q, T_k] or None.
        graph_edges:    [V, M] int32 CPU tensor from build_mlp_transition_graph.
        lm_head_norm_dev: F.normalize(lm_head.weight.float(), dim=-1) [V, d] on device.

    Returns:
        [k] LongTensor on CPU.

    Step 1 — Attention signal (cheap: reads [H, T_q, T_k] already in memory):
        Mean attention over heads for last query position → position scores [T_k].
        Top-hybrid_attn_top_positions positions → attended token IDs.

    Step 2 — MLP Graph expansion (cheap: CPU lookup, O(n_pos * M)):
        For each attended token ID: fetch its M graph neighbours.
        Union of attended IDs + all graph neighbours.

    Step 3 — Pad with cosine (only if |union| < k):
        Full-vocab cosine scan on h_T, masking already-selected tokens.
        This scan is avoided entirely whenever |union| >= k.
    """
    attn_w = captured_attn.get("attn")  # [H, T_q, T_k] or None

    if attn_w is not None:
        # Mean across heads, last query position → [T_k]
        attn_score = attn_w[:, -1, :].mean(dim=0).float().cpu()
        T = attn_score.shape[0]
        n_pos = min(CFG.hybrid_attn_top_positions, T)
        top_pos = attn_score.topk(n_pos).indices        # [n_pos]
        ids_cpu = input_ids_seq.cpu()
        attended_ids = ids_cpu[top_pos]                 # [n_pos] CPU token IDs
    else:
        # Fallback: no attention signal available — use empty attended set
        attended_ids = torch.empty(0, dtype=torch.long)

    # ── 1-hop graph walk from each attended token (pure CPU lookup) ───────────
    if len(attended_ids) > 0:
        neighbour_ids = graph_edges[attended_ids].reshape(-1).long()  # [n_pos * M]
        all_ids = torch.cat([attended_ids, neighbour_ids]).unique()   # CPU
    else:
        all_ids = torch.empty(0, dtype=torch.long)

    if len(all_ids) >= k:
        return all_ids[:k]

    # ── Pad with cosine-nearest on device (only when |union| < k) ────────────
    lm_dev = lm_head_norm_dev.device
    h_dev = hidden.to(lm_dev).float().unsqueeze(0)
    h_norm = F.normalize(h_dev, dim=-1)
    cos_sims = (h_norm @ lm_head_norm_dev.T).squeeze(0).cpu()   # [V]
    if len(all_ids) > 0:
        cos_sims[all_ids] = -1e9
    pad_ids = cos_sims.topk(k - len(all_ids)).indices
    return torch.cat([all_ids, pad_ids])


# ── MLP graph router ──────────────────────────────────────────────────────────

def get_graph_shortlist(
    hidden: torch.Tensor,
    k: int,
    graph_edges: torch.Tensor,
    lm_head_norm_dev: torch.Tensor,
) -> torch.Tensor:
    """
    lm_head_norm_dev: F.normalize(lm_head.weight.float(), dim=-1) [V, d] on DEVICE.
    1. Top-GRAPH_ANCHOR_K tokens by cosine(h_T, lm_head_norm) — on device.
    2. 1-hop graph walk from each anchor.
    3. Union → k tokens.
    """
    lm_dev = lm_head_norm_dev.device
    h_dev = hidden.to(lm_dev).float().unsqueeze(0)
    h_norm = F.normalize(h_dev, dim=-1)
    cos_sims_dev = (h_norm @ lm_head_norm_dev.T).squeeze(0)   # [V] on lm_dev
    anchor_ids = cos_sims_dev.topk(CFG.graph_anchor_k).indices.cpu()  # [A] CPU

    neighbour_ids = graph_edges[anchor_ids].reshape(-1).long()  # [A * M] CPU
    all_ids = torch.cat([anchor_ids, neighbour_ids]).unique()    # CPU

    if len(all_ids) >= k:
        return all_ids[:k]

    cos_sims_cpu = cos_sims_dev.cpu()
    cos_sims_cpu[all_ids] = -1e9
    pad_ids = cos_sims_cpu.topk(k - len(all_ids)).indices
    return torch.cat([all_ids, pad_ids])
