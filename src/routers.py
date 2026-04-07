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
    sims = (h_norm @ cluster_centers_norm.T).squeeze(0)  # [C]
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
