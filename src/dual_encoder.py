"""Dual-encoder router: completion corpus, RouterMLP, InfoNCE training."""
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.config import CFG
from src.model_utils import DEVICE


# ── Completion encoding ───────────────────────────────────────────────────────

def encode_completion(model, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Encode a completion (token IDs) with the frozen LLM.
    Returns mean-pooled last hidden state as float32 CPU vector [d].
    """
    ids = token_ids.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    h = out.hidden_states[-1][0].float().cpu()  # [T, d]
    return h.mean(dim=0)                         # [d]


# ── Build completion index ────────────────────────────────────────────────────

def build_completion_index(model, tokenizer, raw_dataset):
    """
    Sample N_INDEX_COMPLETIONS completions from the WikiText-2 train split,
    encode each, return:
      completion_token_lists : List[Tensor[COMPLETION_LEN]]
      completion_index       : Tensor[N, d]  (raw embeddings, not normalised)
    """
    print("Tokenising WikiText-2 train split…")
    train_lines = [l for l in raw_dataset["train"]["text"] if len(l.strip()) > 20]
    train_parts = [tokenizer.encode(l, return_tensors="pt")[0] for l in train_lines]
    train_tokens = torch.cat(train_parts)
    print(f"  Total training tokens: {len(train_tokens):,}")

    min_prefix = 16
    max_start = len(train_tokens) - CFG.completion_len - 1
    rng = np.random.default_rng(CFG.seed)
    split_points = rng.choice(
        np.arange(min_prefix, max_start),
        size=CFG.n_index_completions,
        replace=False,
    )
    split_points.sort()

    print(f"Encoding {CFG.n_index_completions} completions (len={CFG.completion_len})…")
    completion_token_lists: List[torch.Tensor] = []
    completion_embeddings: List[torch.Tensor] = []

    for sp in tqdm(split_points, desc="Encoding completions"):
        comp_ids = train_tokens[sp : sp + CFG.completion_len]
        comp_emb = encode_completion(model, comp_ids)
        completion_token_lists.append(comp_ids)
        completion_embeddings.append(comp_emb)

    completion_index = torch.stack(completion_embeddings, dim=0)  # [N, d]
    print(f"  Completion index shape: {completion_index.shape}")
    return completion_token_lists, completion_index, train_tokens


# ── RouterMLP ─────────────────────────────────────────────────────────────────

class RouterMLP(nn.Module):
    """3-layer MLP: h_T ∈ R^d → normalised query ∈ R^d_r."""

    def __init__(self, in_dim: int = None, hidden_dim: int = None, out_dim: int = None):
        super().__init__()
        in_dim = in_dim or CFG.router_out_dim  # placeholder; set from HIDDEN_DIM below
        hidden_dim = hidden_dim or CFG.router_hidden_dim
        out_dim = out_dim or CFG.router_out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def mnrl_loss(queries: torch.Tensor, keys: torch.Tensor,
              temperature: float = CFG.temperature) -> torch.Tensor:
    """InfoNCE with in-batch negatives. queries[i] ↔ keys[i] are positives."""
    sim = (queries @ keys.T) / temperature  # [B, B]
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


# ── Training dataset ──────────────────────────────────────────────────────────

class CompletionPairDataset(Dataset):
    """Pre-extracts (h_T, completion_embedding) pairs for router training."""

    def __init__(self, model, train_tokens: torch.Tensor, n_pairs: int = CFG.n_train_pairs):
        self.anchors: List[torch.Tensor] = []
        self.positives: List[torch.Tensor] = []

        max_start = len(train_tokens) - CFG.completion_len - 1
        min_prefix = 16
        rng = np.random.default_rng(CFG.seed + 1)
        pts = rng.choice(np.arange(min_prefix, max_start), size=n_pairs, replace=False)

        print(f"Pre-extracting {n_pairs} training pairs…")
        model.eval()
        with torch.no_grad():
            for sp in tqdm(pts, desc="Building train pairs"):
                sp = int(sp)
                prefix_ids = train_tokens[max(0, sp - CFG.prefix_len_cap) : sp]
                comp_ids = train_tokens[sp : sp + CFG.completion_len]
                if len(prefix_ids) < 4:
                    continue
                pids = prefix_ids.unsqueeze(0).to(DEVICE)
                out_p = model(input_ids=pids, output_hidden_states=True)
                h_T = out_p.hidden_states[-1][0, -1, :].float().cpu()
                comp_emb = encode_completion(model, comp_ids)
                self.anchors.append(h_T)
                self.positives.append(comp_emb)

        print(f"  Training pairs collected: {len(self.anchors):,}")

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx]


# ── Training loop ─────────────────────────────────────────────────────────────

def train_router(model, train_tokens: torch.Tensor) -> Tuple[RouterMLP, nn.Linear]:
    """Train RouterMLP + key_projector via InfoNCE. Returns (router, key_projector)."""
    hidden_dim = model.config.hidden_size

    router = RouterMLP(
        in_dim=hidden_dim,
        hidden_dim=CFG.router_hidden_dim,
        out_dim=CFG.router_out_dim,
    )
    key_projector = nn.Linear(hidden_dim, CFG.router_out_dim, bias=False)

    print(f"RouterMLP parameters: {sum(p.numel() for p in router.parameters()):,}")

    dataset = CompletionPairDataset(model, train_tokens, n_pairs=CFG.n_train_pairs)
    loader = DataLoader(
        dataset, batch_size=CFG.router_batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(key_projector.parameters()),
        lr=CFG.router_lr,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.router_train_epochs * len(loader)
    )

    all_losses = []
    router.train()
    key_projector.train()

    for epoch in range(CFG.router_train_epochs):
        epoch_losses = []
        for anchors, positives in tqdm(loader, desc=f"Epoch {epoch+1}/{CFG.router_train_epochs}"):
            queries = router(anchors)
            keys = F.normalize(key_projector(positives), dim=-1)
            loss = mnrl_loss(queries, keys)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())
        all_losses.extend(epoch_losses)
        print(f"  Epoch {epoch+1}: mean loss = {np.mean(epoch_losses):.4f}")

    torch.save(
        {"router": router.state_dict(), "key_projector": key_projector.state_dict()},
        CFG.router_checkpoint,
    )
    print(f"Router checkpoint saved → {CFG.router_checkpoint}")
    return router, key_projector


def load_router(model) -> Tuple[RouterMLP, nn.Linear]:
    """Load router + key_projector from checkpoint."""
    hidden_dim = model.config.hidden_size
    router = RouterMLP(in_dim=hidden_dim, hidden_dim=CFG.router_hidden_dim, out_dim=CFG.router_out_dim)
    key_projector = nn.Linear(hidden_dim, CFG.router_out_dim, bias=False)
    ckpt = torch.load(CFG.router_checkpoint, weights_only=True)
    router.load_state_dict(ckpt["router"])
    key_projector.load_state_dict(ckpt["key_projector"])
    router.eval()
    key_projector.eval()
    print(f"Router loaded from {CFG.router_checkpoint}")
    return router, key_projector


def project_completion_index(key_projector: nn.Linear,
                              completion_index: torch.Tensor,
                              device: torch.device = None) -> torch.Tensor:
    """Project [N, d] completion embeddings → [N, d_r] normalised.

    Args:
        key_projector: trained linear projection layer.
        completion_index: [N, d] float32 embeddings (CPU).
        device: if provided, move result to this device (e.g. DEVICE for GPU-native router).
                Default None = leave on CPU (backward compatible).
    """
    key_projector.eval()
    chunk_size = 512
    proj_chunks = []
    with torch.no_grad():
        for i in range(0, len(completion_index), chunk_size):
            chunk = completion_index[i : i + chunk_size]
            proj_chunks.append(F.normalize(key_projector(chunk), dim=-1))
    result = torch.cat(proj_chunks, dim=0)  # [N, d_r]
    return result.to(device) if device is not None else result
