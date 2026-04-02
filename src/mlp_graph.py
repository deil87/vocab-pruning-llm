"""Build (and cache) the MLP transition graph."""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.config import CFG
from src.model_utils import DEVICE


def build_mlp_transition_graph(model, lm_head_weight: torch.Tensor):
    """
    For each token i, pass lm_head_weight[i] through all MLP layers as a
    residual stream probe, then project back to vocabulary logits and keep the
    top-M successors.

    Returns:
      graph_edges   [V, M]  int32
      graph_weights [V, M]  float32
    """
    V = model.config.vocab_size
    M = CFG.graph_edges_per_node
    graph_edges = torch.zeros(V, M, dtype=torch.int32)
    graph_weights = torch.zeros(V, M, dtype=torch.float32)

    mlp_layers = [layer.mlp for layer in model.model.layers]
    n_layers = len(mlp_layers)
    chunk_size = 256

    print(f"Building MLP transition graph: V={V:,}, M={M}, layers={n_layers}")
    print(f"Processing in chunks of {chunk_size}…")

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, V, chunk_size), desc="Graph construction"):
            end = min(start + chunk_size, V)
            h = lm_head_weight[start:end].to(DEVICE).to(model.dtype)  # [B, d]

            for mlp in mlp_layers:
                h = h + mlp(h)

            logits = h.float().to(model.lm_head.weight.device) @ model.lm_head.weight.float().T  # [B, V]
            top = logits.topk(M, dim=1)
            probs = F.softmax(top.values, dim=-1)

            graph_edges[start:end] = top.indices.int().cpu()
            graph_weights[start:end] = probs.cpu()

    return graph_edges, graph_weights


def load_or_build_graph(model, lm_head_weight: torch.Tensor):
    """Load cached graph or build and save it."""
    if CFG.graph_path.exists():
        print(f"Loading cached graph from {CFG.graph_path}")
        ckpt = torch.load(CFG.graph_path, weights_only=True)
        graph_edges = ckpt["edges"]
        graph_weights = ckpt["weights"]
    else:
        graph_edges, graph_weights = build_mlp_transition_graph(model, lm_head_weight)
        torch.save({"edges": graph_edges, "weights": graph_weights}, CFG.graph_path)
        print(f"Graph saved → {CFG.graph_path}")

    print(f"Graph: edges {graph_edges.shape}, weights {graph_weights.shape}")
    sz = CFG.graph_path.stat().st_size / 1e6
    print(f"Graph size on disk: {sz:.1f} MB")
    return graph_edges, graph_weights
