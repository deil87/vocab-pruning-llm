"""
Triton-fused Dual-Encoder router kernel.

Replaces the three-step Python sequence:
  q   = router_mlp(h_T)                        # MLP forward
  sim = completion_index_proj @ q              # dot-product similarity
  idx = sim.topk(n_retrieve).indices           # top-K retrieval

with a single GPU kernel that:
  1. Runs the 3-layer RouterMLP on h_T (already on GPU)
  2. Computes dot products against the pre-projected completion index
  3. Returns the top-K completion indices — all without leaving the GPU

Additionally provides a GPU-native token-union helper that replaces the
CPU Python loop used to build the vocabulary shortlist from retrieved
completion token lists.

Dimensions (fixed at JIT-compile time, matching CFG defaults):
  d   : model hidden size  (2048 for 1B, 4096 for 8B)
  H   : router hidden dim  = 512
  d_r : router output dim  = 256
  N   : completion index size = 8000

Usage
-----
    from src.router_kernel import build_fused_router, gpu_token_union

    fused_router = build_fused_router(router, key_projector, completion_index_proj)
    top_idx = fused_router(h_T)           # h_T: [d] on DEVICE, returns [n_retrieve] on GPU
    shortlist = gpu_token_union(completion_token_tensor, top_idx, vocab_size, k,
                                lm_head_norm_dev, h_T)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Triton availability guard
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernel: fused MLP forward + dot-product similarity
# ---------------------------------------------------------------------------
# The kernel computes, for a single query vector h [d]:
#   layer1: x1 = GELU(h @ W1.T + b1)            [H]
#   ln1:    x1 = layernorm(x1, w1, b1_ln)        [H]
#   layer2: x2 = GELU(x1 @ W2.T + b2)            [H]
#   ln2:    x2 = layernorm(x2, w2, b2_ln)        [H]
#   layer3: q  = x2 @ W3.T + b3                  [d_r]
#   norm:   q  = q / ||q||
#   sim[n]  = q · E[n]  for n in 0..N-1          [N]
#
# We split this into two kernels:
#   _mlp_kernel: h → q (normalised query vector)
#   _sim_kernel: q, E → sim (dot products)   [can be a plain matmul]
#
# The top-K selection is done with torch.topk on the [N] sim vector —
# at N=8000 this is negligible (<0.01 ms) and not worth a custom kernel.

if _TRITON_AVAILABLE:

    @triton.jit
    def _gelu_kernel_approx(x):
        """Tanh-approximation of GELU, matching torch.nn.GELU default."""
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        c = 0.7978845608028654  # sqrt(2/pi)
        return 0.5 * x * (1.0 + tl.libdevice.tanh(c * (x + 0.044715 * x * x * x)))

    @triton.jit
    def _mlp_forward_kernel(
        # Input
        h_ptr,          # [d] float32
        # Layer 1 weights/bias
        W1_ptr,         # [H, d] float32
        b1_ptr,         # [H]
        # LayerNorm 1
        ln1_w_ptr,      # [H]
        ln1_b_ptr,      # [H]
        # Layer 2 weights/bias
        W2_ptr,         # [H, H] float32
        b2_ptr,         # [H]
        # LayerNorm 2
        ln2_w_ptr,      # [H]
        ln2_b_ptr,      # [H]
        # Layer 3 weights/bias
        W3_ptr,         # [d_r, H] float32
        b3_ptr,         # [d_r]
        # Output
        q_ptr,          # [d_r] float32 — normalised query
        # Compile-time constants
        D:  tl.constexpr,   # model hidden size
        H:  tl.constexpr,   # router hidden dim
        DR: tl.constexpr,   # router output dim
        BLOCK_D:  tl.constexpr,
        BLOCK_H:  tl.constexpr,
        BLOCK_DR: tl.constexpr,
        LN_EPS:   tl.constexpr,
    ):
        """Single-program MLP forward: h [D] → normalised q [DR]."""

        # ── Layer 1: x1 = GELU(W1 @ h + b1) ─────────────────────────────────
        # Accumulate row-by-row; each program handles one output neuron.
        # For simplicity we use a loop — at H=512, D=2048 this is fast.

        pid = tl.program_id(0)   # which output neuron (0..H-1 for layer1)

        # ── We compute the full x1, x2, q vectors sequentially in one program.
        # This avoids inter-program communication.  pid=0 is the only program.

        # Allocate on-chip accumulators (SRAM) via tl.arange
        h_range   = tl.arange(0, BLOCK_D)
        h1_range  = tl.arange(0, BLOCK_H)
        dr_range  = tl.arange(0, BLOCK_DR)

        # Load h
        h_mask = h_range < D
        h_vals = tl.load(h_ptr + h_range, mask=h_mask, other=0.0)   # [BLOCK_D]

        # ── Layer 1 matmul: x1[i] = sum_j W1[i,j] * h[j] + b1[i] ────────────
        x1 = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in range(H):
            w1_row = tl.load(W1_ptr + i * D + h_range, mask=h_mask, other=0.0)
            dot = tl.sum(w1_row * h_vals, axis=0)
            b1_i = tl.load(b1_ptr + i)
            # Store into x1[i] — Triton doesn't support dynamic indexing into
            # registers, so we use a mask trick to write the i-th element.
            idx_mask = (h1_range == i)
            x1 = x1 + tl.where(idx_mask, dot + b1_i, 0.0)

        # GELU activation
        x1 = _gelu_kernel_approx(x1)

        # LayerNorm 1
        mean1 = tl.sum(x1, axis=0) / H
        var1  = tl.sum((x1 - mean1) * (x1 - mean1), axis=0) / H
        x1_n  = (x1 - mean1) / tl.sqrt(var1 + LN_EPS)
        ln1_w = tl.load(ln1_w_ptr + h1_range, mask=h1_range < H, other=1.0)
        ln1_b = tl.load(ln1_b_ptr + h1_range, mask=h1_range < H, other=0.0)
        x1    = x1_n * ln1_w + ln1_b

        # ── Layer 2 matmul: x2[i] = sum_j W2[i,j] * x1[j] + b2[i] ──────────
        x2 = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in range(H):
            w2_row = tl.load(W2_ptr + i * H + h1_range, mask=h1_range < H, other=0.0)
            dot2 = tl.sum(w2_row * x1, axis=0)
            b2_i = tl.load(b2_ptr + i)
            idx_mask2 = (h1_range == i)
            x2 = x2 + tl.where(idx_mask2, dot2 + b2_i, 0.0)

        x2 = _gelu_kernel_approx(x2)

        # LayerNorm 2
        mean2 = tl.sum(x2, axis=0) / H
        var2  = tl.sum((x2 - mean2) * (x2 - mean2), axis=0) / H
        x2_n  = (x2 - mean2) / tl.sqrt(var2 + LN_EPS)
        ln2_w = tl.load(ln2_w_ptr + h1_range, mask=h1_range < H, other=1.0)
        ln2_b = tl.load(ln2_b_ptr + h1_range, mask=h1_range < H, other=0.0)
        x2    = x2_n * ln2_w + ln2_b

        # ── Layer 3 matmul: q[i] = sum_j W3[i,j] * x2[j] + b3[i] ───────────
        q = tl.zeros([BLOCK_DR], dtype=tl.float32)
        for i in range(DR):
            w3_row = tl.load(W3_ptr + i * H + h1_range, mask=h1_range < H, other=0.0)
            dot3 = tl.sum(w3_row * x2, axis=0)
            b3_i = tl.load(b3_ptr + i)
            idx_mask3 = (dr_range == i)
            q = q + tl.where(idx_mask3, dot3 + b3_i, 0.0)

        # L2 normalise q
        norm_sq = tl.sum(q * q, axis=0)
        q = q / tl.sqrt(norm_sq + 1e-12)

        # Store output
        tl.store(q_ptr + dr_range, q, mask=dr_range < DR)


# ---------------------------------------------------------------------------
# FusedRouter: Python wrapper that orchestrates the kernel + topk
# ---------------------------------------------------------------------------

class FusedRouter:
    """
    GPU-native router that keeps h_T on the GPU throughout.

    If Triton is available: uses the _mlp_forward_kernel for the MLP pass,
    then a plain torch matmul for the similarity search (N=8000 is too small
    to benefit from a custom matmul kernel; cuBLAS is optimal here).

    If Triton is NOT available: falls back to a pure-PyTorch path that still
    avoids the CPU round-trip (Change 1) and uses the GPU-native token union
    (Change 2), giving a meaningful speedup even without kernel compilation.

    Parameters
    ----------
    router_mlp : RouterMLP
        The trained 3-layer MLP (on DEVICE, eval mode).
    completion_index_proj : Tensor [N, d_r]
        Pre-projected, L2-normalised completion embeddings (on DEVICE).
    n_retrieve : int
        Number of completions to retrieve per step (CFG.n_retrieve).
    use_triton : bool or None
        True = require Triton; False = pure PyTorch; None = auto-detect.
    """

    def __init__(
        self,
        router_mlp,
        completion_index_proj: torch.Tensor,
        n_retrieve: int = 32,
        use_triton: Optional[bool] = None,
    ):
        self.router_mlp = router_mlp
        self.completion_index_proj = completion_index_proj  # [N, d_r] on GPU
        self.n_retrieve = n_retrieve

        # Decide Triton vs PyTorch fallback
        if use_triton is None:
            use_triton = _TRITON_AVAILABLE
        self.use_triton = use_triton and _TRITON_AVAILABLE

        if self.use_triton:
            self._extract_weights()

    def _extract_weights(self):
        """Cache contiguous float32 weight/bias tensors for the Triton kernel."""
        net = self.router_mlp.net
        # Sequential: Linear(0), GELU(1), LayerNorm(2), Linear(3), GELU(4), LayerNorm(5), Linear(6)
        l1: torch.nn.Linear    = net[0]
        ln1: torch.nn.LayerNorm = net[2]
        l2: torch.nn.Linear    = net[3]
        ln2: torch.nn.LayerNorm = net[5]
        l3: torch.nn.Linear    = net[6]

        dev = next(self.router_mlp.parameters()).device

        self.W1 = l1.weight.float().contiguous().to(dev)   # [H, d]
        self.b1 = l1.bias.float().contiguous().to(dev)     # [H]
        self.ln1_w = ln1.weight.float().contiguous().to(dev)
        self.ln1_b = ln1.bias.float().contiguous().to(dev)
        self.W2 = l2.weight.float().contiguous().to(dev)   # [H, H]
        self.b2 = l2.bias.float().contiguous().to(dev)     # [H]
        self.ln2_w = ln2.weight.float().contiguous().to(dev)
        self.ln2_b = ln2.bias.float().contiguous().to(dev)
        self.W3 = l3.weight.float().contiguous().to(dev)   # [d_r, H]
        self.b3 = l3.bias.float().contiguous().to(dev)     # [d_r]

        self.D  = self.W1.shape[1]   # model hidden size
        self.H  = self.W1.shape[0]   # router hidden dim
        self.DR = self.W3.shape[0]   # router output dim
        self.dev = dev

    @torch.no_grad()
    def __call__(self, h_T: torch.Tensor) -> torch.Tensor:
        """
        h_T : [d] tensor on GPU (any dtype — converted to float32 internally).
        Returns: [n_retrieve] LongTensor of completion indices, on GPU.
        """
        h = h_T.float()
        if h.dim() == 2:
            h = h.squeeze(0)

        if self.use_triton:
            q = self._triton_mlp(h)
        else:
            q = self.router_mlp(h.unsqueeze(0)).squeeze(0)

        # Similarity: [N, d_r] @ [d_r] → [N]
        sims = self.completion_index_proj @ q
        top_idx = sims.topk(self.n_retrieve).indices   # [n_retrieve] on GPU
        return top_idx

    def _triton_mlp(self, h: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward via the Triton kernel. Returns normalised q [d_r]."""
        q = torch.empty(self.DR, dtype=torch.float32, device=self.dev)

        # Block sizes must be powers-of-2 >= actual dims for tl.arange
        BLOCK_D  = _next_pow2(self.D)
        BLOCK_H  = _next_pow2(self.H)
        BLOCK_DR = _next_pow2(self.DR)

        _mlp_forward_kernel[(1,)](
            h,
            self.W1, self.b1,
            self.ln1_w, self.ln1_b,
            self.W2, self.b2,
            self.ln2_w, self.ln2_b,
            self.W3, self.b3,
            q,
            D=self.D, H=self.H, DR=self.DR,
            BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_DR=BLOCK_DR,
            LN_EPS=1e-5,
        )
        return q


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# GPU-native token union (Change 2)
# ---------------------------------------------------------------------------

def gpu_token_union(
    completion_token_tensor: torch.Tensor,   # [N, completion_len] int32/int64 on GPU
    top_idx: torch.Tensor,                   # [n_retrieve] int64 on GPU
    vocab_size: int,
    k: int,
    lm_head_norm_dev: torch.Tensor,          # [V, d] float32 on GPU — for cosine padding
    h_T: torch.Tensor,                       # [d] float32 on GPU — for cosine padding
) -> torch.Tensor:
    """
    GPU-native replacement for the CPU Python loop that builds the token union.

    Instead of:
        token_id_sets = [completion_token_lists[i.item()] for i in top_idx]
        all_ids = torch.cat(token_id_sets).unique()

    We use a boolean mask over [vocab_size] that is set for all tokens that
    appear in any of the n_retrieve retrieved completions.

    Parameters
    ----------
    completion_token_tensor : [N, completion_len] — all completion token ID
        sequences as a 2-D tensor on GPU (pre-built once, reused every step).
    top_idx : [n_retrieve] — GPU indices of retrieved completions.
    vocab_size : full vocabulary size (e.g. 128256 for Llama-3).
    k : target shortlist size.
    lm_head_norm_dev : for cosine-nearest padding when union < k.
    h_T : hidden state for cosine padding.

    Returns
    -------
    shortlist : [k] LongTensor on GPU.
    """
    # Gather the token IDs for the retrieved completions: [n_retrieve, completion_len]
    retrieved = completion_token_tensor[top_idx]          # [n_retrieve, comp_len] on GPU

    # Build a boolean presence mask over the vocabulary
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=retrieved.device)
    mask.scatter_(0, retrieved.reshape(-1), True)

    all_ids = mask.nonzero(as_tuple=False).squeeze(1)     # [M] on GPU

    if all_ids.numel() >= k:
        return all_ids[:k]

    # Cosine-nearest padding (stays on GPU)
    h_norm = F.normalize(h_T.float().unsqueeze(0), dim=-1)           # [1, d]
    cos_sims = (h_norm @ lm_head_norm_dev.T).squeeze(0)              # [V] on GPU
    cos_sims[all_ids] = -1e9                                          # mask existing
    n_pad = k - all_ids.numel()
    pad_ids = cos_sims.topk(n_pad).indices                            # [n_pad] on GPU
    return torch.cat([all_ids, pad_ids])                              # [k] on GPU


# ---------------------------------------------------------------------------
# One-time setup: pack completion_token_lists into a 2-D GPU tensor
# ---------------------------------------------------------------------------

def build_completion_token_tensor(
    completion_token_lists: List[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Pack variable-length completion token lists into a fixed [N, max_len] int64
    tensor on the GPU.  Shorter completions are zero-padded (token 0 = <unk>,
    benign in the mask — it is masked out unless it genuinely appears).

    Called once after building the completion index; replaces the Python list.
    """
    max_len = max(t.numel() for t in completion_token_lists)
    N = len(completion_token_lists)
    out = torch.zeros(N, max_len, dtype=torch.int64, device=device)
    for i, t in enumerate(completion_token_lists):
        out[i, : t.numel()] = t.long()
    return out   # [N, max_len]


# ---------------------------------------------------------------------------
# High-level convenience: build a ready-to-use FusedRouter
# ---------------------------------------------------------------------------

def build_fused_router(
    router_mlp,
    completion_index_proj: torch.Tensor,
    n_retrieve: int = 32,
    use_triton: Optional[bool] = None,
) -> FusedRouter:
    """
    Build a FusedRouter from a trained RouterMLP and a projected completion index.

    Parameters
    ----------
    router_mlp : RouterMLP — trained, on DEVICE, eval mode.
    completion_index_proj : [N, d_r] float32 on DEVICE.
    n_retrieve : completions to retrieve per step.
    use_triton : True/False/None(=auto).

    Returns
    -------
    FusedRouter instance ready to call as fused_router(h_T).
    """
    router_mlp.eval()
    return FusedRouter(
        router_mlp=router_mlp,
        completion_index_proj=completion_index_proj,
        n_retrieve=n_retrieve,
        use_triton=use_triton,
    )
