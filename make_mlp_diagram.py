"""Generate MLP Transition Graph Router diagram as PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Colour palette ─────────────────────────────────────────────────────────────
C_OFFLINE_BG  = "#1a1a2e"
C_ONLINE_BG   = "#16213e"
C_OFFLINE_HDR = "#e94560"
C_ONLINE_HDR  = "#0f9b8e"
C_BOX_OFFLINE = "#2d2d5e"
C_BOX_ONLINE  = "#1e4d4a"
C_BOX_EDGE_O  = "#e94560"
C_BOX_EDGE_N  = "#0f9b8e"
C_TEXT        = "#f0f0f0"
C_SUBTEXT     = "#b0b8c8"
C_ARROW       = "#aaaacc"
C_ARROW_DASH  = "#888888"
C_ACCENT      = "#f5a623"
C_PAD         = "#7c5cbf"

fig = plt.figure(figsize=(18, 13), facecolor="#0d0d1a")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")

# ── Helper: rounded box ────────────────────────────────────────────────────────
def box(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.25, alpha=1.0):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
                       zorder=3)
    ax.add_patch(b)

def label(ax, x, y, text, size=9, color=C_TEXT, ha="center", va="center",
          weight="normal", style="normal"):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, fontstyle=style, zorder=5,
            fontfamily="monospace")

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, ls="-",
          arrowstyle="->", mutation_scale=14):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, linestyle=ls,
                                mutation_scale=mutation_scale),
                zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
ax.text(9, 12.4, "MLP Transition Graph Router",
        fontsize=17, color=C_TEXT, ha="center", va="center",
        fontweight="bold", zorder=5)
ax.text(9, 12.0,
        "Vocabulary pruning via offline MLP residual probing + online 1-hop graph walk",
        fontsize=10, color=C_SUBTEXT, ha="center", va="center", zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# OFFLINE panel (left)
# ══════════════════════════════════════════════════════════════════════════════
# Panel background
box(ax, 0.3, 0.4, 7.4, 11.2, C_OFFLINE_BG, C_BOX_EDGE_O, lw=2, radius=0.4)
ax.text(4.0, 11.25, "//  OFFLINE  —  Build Graph (once)",
        fontsize=11, color=C_OFFLINE_HDR, ha="center", va="center",
        fontweight="bold", zorder=5)
ax.text(4.0, 10.85, "cached to  mlp_transition_graph.pt  (31 MB)",
        fontsize=8, color=C_SUBTEXT, ha="center", va="center", zorder=5,
        fontstyle="italic")

# Node 1 — lm_head.weight
box(ax, 0.9, 9.6, 6.2, 0.85, C_BOX_OFFLINE, C_BOX_EDGE_O, lw=1.5)
label(ax, 4.0, 10.10, "lm_head.weight   [128 256 × 2048]", size=9, weight="bold")
label(ax, 4.0,  9.78, "row i  =  output embedding of token i", size=8, color=C_SUBTEXT)

arrow(ax, 4.0, 9.60, 4.0, 9.10)

# Node 2 — chunk
box(ax, 0.9, 8.35, 6.2, 0.65, C_BOX_OFFLINE, C_BOX_EDGE_O)
label(ax, 4.0, 8.675, "Iterate over all V = 128 256 tokens  (chunks of 256)", size=8.5)

arrow(ax, 4.0, 8.35, 4.0, 7.85)

# Node 3 — MLP residual probe
box(ax, 0.9, 6.85, 6.2, 0.90, C_BOX_OFFLINE, C_BOX_EDGE_O)
label(ax, 4.0, 7.38, "Residual MLP probe  (16 layers)", size=9, weight="bold")
label(ax, 4.0, 7.05, "h = lm_head.weight[i]", size=8, color=C_SUBTEXT)
label(ax, 4.0, 6.76, "for each layer:   h = h + MLP(h)", size=8, color=C_ACCENT)

arrow(ax, 4.0, 6.85, 4.0, 6.30)

# Node 4 — project back
box(ax, 0.9, 5.45, 6.2, 0.75, C_BOX_OFFLINE, C_BOX_EDGE_O)
label(ax, 4.0, 5.91, "Project back to vocabulary", size=9, weight="bold")
label(ax, 4.0, 5.60, "logits = h · lm_head.weightᵀ   →   [128 256]", size=8, color=C_SUBTEXT)

arrow(ax, 4.0, 5.45, 4.0, 4.90)

# Node 5 — topk successors
box(ax, 0.9, 4.10, 6.2, 0.70, C_BOX_OFFLINE, C_BOX_EDGE_O)
label(ax, 4.0, 4.55, "topk( M = 32 )  successors per token", size=9, weight="bold")
label(ax, 4.0, 4.23, "tokens most likely to follow token i  (by MLP logits)", size=8, color=C_SUBTEXT)

arrow(ax, 4.0, 4.10, 4.0, 3.55)

# Node 6 — graph stored
box(ax, 0.9, 2.60, 6.2, 0.85, C_BOX_OFFLINE, C_BOX_EDGE_O, lw=2)
label(ax, 4.0, 3.13, "graph_edges    [128 256 × 32]   int32", size=9, weight="bold")
label(ax, 4.0, 2.82, "graph_weights  [128 256 × 32]   float32", size=8.5, color=C_SUBTEXT)

# Loopback annotation
ax.annotate("", xy=(0.65, 8.675), xytext=(0.65, 9.60),
            arrowprops=dict(arrowstyle="->", color=C_OFFLINE_HDR,
                            lw=1.2, linestyle="--", mutation_scale=10),
            zorder=4)
ax.text(0.45, 9.1, "∀i", fontsize=8, color=C_OFFLINE_HDR,
        ha="center", rotation=90, zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# ONLINE panel (right)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 9.9, 0.4, 7.8, 11.2, C_ONLINE_BG, C_BOX_EDGE_N, lw=2, radius=0.4)
ax.text(13.8, 11.25, ">>  ONLINE  —  Route at Each Decoding Step",
        fontsize=11, color=C_ONLINE_HDR, ha="center", va="center",
        fontweight="bold", zorder=5)

# Node A — hidden state
box(ax, 10.4, 9.6, 6.8, 0.85, C_BOX_ONLINE, C_BOX_EDGE_N, lw=1.5)
label(ax, 13.8, 10.10, "hidden state   h_T   [2048]", size=9, weight="bold")
label(ax, 13.8,  9.78, "last-token hidden state from LLM forward pass", size=8, color=C_SUBTEXT)

arrow(ax, 13.8, 9.60, 13.8, 9.10)

# Node B — cosine sim
box(ax, 10.4, 8.25, 6.8, 0.75, C_BOX_ONLINE, C_BOX_EDGE_N)
label(ax, 13.8, 8.72, "Cosine similarity  vs  lm_head_norm", size=9, weight="bold")
label(ax, 13.8, 8.42, "h_T · lm_head_norm.T   →   scores [128 256]", size=8, color=C_SUBTEXT)

arrow(ax, 13.8, 8.25, 13.8, 7.75)

# Node C — anchor tokens
box(ax, 10.4, 6.95, 6.8, 0.70, C_BOX_ONLINE, C_BOX_EDGE_N)
label(ax, 13.8, 7.41, "Top  A = 16  anchor tokens", size=9, weight="bold")
label(ax, 13.8, 7.10, "vocabulary tokens most similar to h_T", size=8, color=C_SUBTEXT)

arrow(ax, 13.8, 6.95, 13.8, 6.40)

# Node D — graph walk
box(ax, 10.4, 5.50, 6.8, 0.80, C_BOX_ONLINE, C_BOX_EDGE_N)
label(ax, 13.8, 5.99, "1-hop graph walk", size=9, weight="bold")
label(ax, 13.8, 5.70, "graph_edges[ anchor_ids ]   →   neighbour_ids", size=8, color=C_ACCENT)
label(ax, 13.8, 5.43, "16 anchors × 32 edges  =  up to 512 neighbours", size=8, color=C_SUBTEXT)

arrow(ax, 13.8, 5.50, 13.8, 4.95)

# Node E — union
box(ax, 10.4, 4.10, 6.8, 0.75, C_BOX_ONLINE, C_BOX_EDGE_N)
label(ax, 13.8, 4.57, "Union of anchors + neighbours", size=9, weight="bold")
label(ax, 13.8, 4.27, "≤ 528 candidate tokens", size=8, color=C_SUBTEXT)

arrow(ax, 13.8, 4.10, 13.8, 3.55)

# Node F — pad decision (diamond-ish wider box)
box(ax, 10.4, 2.70, 6.8, 0.75, C_PAD+"33", C_PAD, lw=1.8)
label(ax, 13.8, 3.17, "shortlist size  ≥  k ?", size=9, weight="bold", color="#e0d0ff")
label(ax, 13.8, 2.87, "k ∈ {512, 1000, 2000, 5000, 10000}", size=8, color=C_SUBTEXT)

# YES branch — straight down
arrow(ax, 13.8, 2.70, 13.8, 2.15, color=C_ONLINE_HDR)
ax.text(14.15, 2.42, "yes", fontsize=8, color=C_ONLINE_HDR, zorder=5)

# NO branch — right side pad box
box(ax, 10.4, 1.45, 6.8, 0.65, C_BOX_ONLINE, C_PAD, lw=1.5)
label(ax, 13.8, 1.78, "Pad with cosine-nearest tokens  until size = k", size=8.5, color="#e0d0ff")
# Arrow: NO from left side of decision box down to pad box
arrow(ax, 10.4, 3.075, 9.8, 3.075, color=C_PAD, lw=1.3)
arrow(ax, 9.8, 3.075, 9.8, 1.775, color=C_PAD, lw=1.3)
arrow(ax, 9.8, 1.775, 10.4, 1.775, color=C_PAD, lw=1.3)
ax.text(9.55, 2.42, "no", fontsize=8, color=C_PAD, ha="center", rotation=90, zorder=5)
# Pad box merges back down
arrow(ax, 13.8, 1.45, 13.8, 1.05, color=C_ONLINE_HDR)

# Node G — lm_head pruned
box(ax, 10.4, 0.55, 6.8, 0.80, C_BOX_ONLINE, C_BOX_EDGE_N, lw=2)
label(ax, 13.8, 1.05, "lm_head matmul  over shortlist only", size=9, weight="bold")
label(ax, 13.8, 0.75, "[k × 2048]  instead of  [128 256 × 2048]", size=8, color=C_ACCENT)

# ══════════════════════════════════════════════════════════════════════════════
# Cross-panel dashed arrow: graph → walk
# ══════════════════════════════════════════════════════════════════════════════
# From right edge of graph node to left edge of graph-walk node
ax.annotate("", xy=(10.4, 5.90), xytext=(7.1, 3.025),
            arrowprops=dict(arrowstyle="->", color=C_ARROW_DASH,
                            lw=1.4, linestyle="dashed",
                            connectionstyle="arc3,rad=-0.25",
                            mutation_scale=12),
            zorder=4)
ax.text(8.75, 5.0, "loaded\nat startup", fontsize=7.5, color=C_ARROW_DASH,
        ha="center", va="center", fontstyle="italic", zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# Speedup annotation box
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 10.4, 0.0, 6.8, 0.48, "#1a1a1a", C_ACCENT, lw=1.5, radius=0.2, alpha=0.85)
ax.text(13.8, 0.24,
        "Speedup: lm_head FLOPs reduced by  ×(128256 / k)  |  k=10000 → ×12.8",
        fontsize=8, color=C_ACCENT, ha="center", va="center", zorder=5)

plt.tight_layout(pad=0)
out = "paper/mlp_graph_diagram.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
