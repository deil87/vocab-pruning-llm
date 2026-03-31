"""Generate MLP Transition Graph Router — graph structure diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import torch
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
GRAPH_PATH = "results/mlp_transition_graph.pt"
MODEL_NAME  = "meta-llama/Llama-3.2-1B-Instruct"
OUT_PATH    = "paper/mlp_graph_structure.png"
N_NBRS      = 6   # neighbours per anchor shown in graph panel

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = "#0d0d1a"
PANEL_BG    = "#12122a"
C_TEXT      = "#f0f0f0"
C_SUBTEXT   = "#8899aa"
C_GRID      = "#1e1e3a"

GROUP_COLS = {
    "paris":  {"anchor": "#0fd9c0", "nbr": "#0a7a6e", "edge": "#0fd9c0"},
    "neural": {"anchor": "#c07aff", "nbr": "#6a3d8f", "edge": "#c07aff"},
    "king":   {"anchor": "#f5a623", "nbr": "#8a5c10", "edge": "#f5a623"},
    "run":    {"anchor": "#ff5577", "nbr": "#8a2030", "edge": "#ff5577"},
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Loading graph…")
ckpt   = torch.load(GRAPH_PATH, weights_only=True)
edges  = ckpt["edges"]    # [V, 32] int32
weights = ckpt["weights"] # [V, 32] float32

def get_neighbours(token_str, n=N_NBRS):
    tid  = tok.encode(token_str, add_special_tokens=False)[0]
    nbr_ids = edges[tid].tolist()[:n]
    nbr_wts  = weights[tid].tolist()[:n]
    nbr_strs = [tok.decode([i]) for i in nbr_ids]
    return tid, nbr_ids, nbr_strs, nbr_wts

# Real data
anchors_data = {
    "paris":  get_neighbours(" Paris"),
    "neural": get_neighbours(" neural"),
    "king":   get_neighbours(" king"),
    "run":    get_neighbours(" run"),
}

# Print for verification
for name, (tid, nbr_ids, nbr_strs, nbr_wts) in anchors_data.items():
    print(f"  {name:8s} id={tid}: {list(zip(nbr_strs, [round(w,3) for w in nbr_wts]))}")

# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 19), facecolor=BG)

# GridSpec: top-left (graph), top-right (schematic), bottom (pipeline)
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2, figure=fig,
              left=0.02, right=0.98, top=0.90, bottom=0.03,
              hspace=0.10, wspace=0.07,
              height_ratios=[1.8, 0.82],
              width_ratios=[1.55, 1.0])

ax_graph    = fig.add_subplot(gs[0, 0])  # left top:   graph
ax_schema   = fig.add_subplot(gs[0, 1])  # right top:  data structure
ax_pipeline = fig.add_subplot(gs[1, :])  # bottom full: inference pipeline

for ax in [ax_graph, ax_schema, ax_pipeline]:
    ax.set_facecolor(PANEL_BG)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
        spine.set_linewidth(1.2)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Graph Structure
# ══════════════════════════════════════════════════════════════════════════════
ax_graph.set_title("Token Neighbourhood Graph  (4 anchor tokens × 6 neighbours each)",
                   fontsize=11, color=C_TEXT, pad=8, fontfamily="monospace")

# Build networkx graph
G = nx.Graph()
node_colour = {}
node_size   = {}
node_group  = {}
edge_weights = {}

# Anchors at true corners — symmetric ±2.0 grid, each owns its quadrant
anchor_pos = {
    " Paris":  np.array([-2.0,  2.0]),   # upper-left
    " neural": np.array([ 2.0,  2.0]),   # upper-right
    " king":   np.array([-2.0, -2.0]),   # lower-left
    " run":    np.array([ 2.0, -2.0]),   # lower-right
}
# Fan base angles: point toward the nearest corner of the panel
anchor_base_angle = {
    " Paris":  np.radians(135),   # NW
    " neural": np.radians(45),    # NE
    " king":   np.radians(225),   # SW
    " run":    np.radians(315),   # SE
}
anchor_name_map = {
    " Paris": "paris", " neural": "neural",
    " king": "king",   " run": "run",
}

pos = {}
for anchor_str, anchor_name in anchor_name_map.items():
    tid, nbr_ids, nbr_strs, nbr_wts = anchors_data[anchor_name]
    G.add_node(anchor_str, kind="anchor", group=anchor_name)
    pos[anchor_str] = anchor_pos[anchor_str].copy()
    node_colour[anchor_str] = GROUP_COLS[anchor_name]["anchor"]
    node_size[anchor_str]   = 900
    node_group[anchor_str]  = anchor_name

    # Place neighbours in a fan around the anchor
    ap = anchor_pos[anchor_str]
    base_angle = anchor_base_angle[anchor_str]
    spread = np.pi * 0.55          # 99° fan — stays within quadrant
    for i, (ns, nw) in enumerate(zip(nbr_strs, nbr_wts)):
        # ASCII-safe label (replace non-ASCII chars)
        ns_safe = ns.encode("ascii", "replace").decode("ascii").replace("?", "*")
        node_id = f"{ns_safe}_{anchor_name}"  # unique id
        angle = base_angle - spread/2 + (spread * i / max(N_NBRS-1, 1))
        dist  = 1.45 + 0.20 * (i % 2)  # stagger odd/even radially
        nx_pos = ap + dist * np.array([np.cos(angle), np.sin(angle)])
        G.add_node(node_id, kind="nbr", group=anchor_name, label=ns_safe)
        pos[node_id] = nx_pos
        node_colour[node_id] = GROUP_COLS[anchor_name]["nbr"]
        node_size[node_id]   = 320
        node_group[node_id]  = anchor_name

        G.add_edge(anchor_str, node_id, weight=nw, group=anchor_name)
        edge_weights[(anchor_str, node_id)] = nw

# ── Draw edges first (behind nodes) ──────────────────────────────────────────
for (u, v), w in edge_weights.items():
    grp  = G[u][v]["group"]
    col  = GROUP_COLS[grp]["edge"]
    # weight range ~0.03–0.055; normalise to lw 0.8–3.5
    lw   = 0.8 + (w - 0.030) / (0.055 - 0.030) * 2.7
    alpha = 0.35 + (w - 0.030) / (0.055 - 0.030) * 0.55
    xu, yu = pos[u]
    xv, yv = pos[v]
    ax_graph.plot([xu, xv], [yu, yv],
                  color=col, lw=lw, alpha=alpha, zorder=1, solid_capstyle="round")

# ── Draw nodes ────────────────────────────────────────────────────────────────
for node in G.nodes():
    x, y = pos[node]
    kind = G.nodes[node]["kind"]
    col  = node_colour[node]
    grp  = node_group[node]
    sz   = node_size[node]
    r    = np.sqrt(sz) / 2400

    if kind == "anchor":
        # Outer glow ring
        ring = Circle((x, y), r * 1.65, color=col, alpha=0.18, zorder=2)
        ax_graph.add_patch(ring)
        ring2 = Circle((x, y), r * 1.35, color=col, alpha=0.25, zorder=2)
        ax_graph.add_patch(ring2)
        # Main circle
        circ = Circle((x, y), r * 1.05, color=col, zorder=3, ec="white", lw=1.2)
        ax_graph.add_patch(circ)
        # Label above
        label_str = G.nodes[node].get("label", node)
        ax_graph.text(x, y + r*1.25, repr(node), fontsize=8.5,
                      ha="center", va="bottom", color="white", fontweight="bold",
                      fontfamily="monospace", zorder=5,
                      bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=col,
                                lw=1.0, alpha=0.85))
    else:
        circ = Circle((x, y), r * 1.1, color=col, zorder=3, ec=col, lw=0.5, alpha=0.88)
        ax_graph.add_patch(circ)
        label_str = G.nodes[node].get("label", node)
        ax_graph.text(x, y + r * 1.5, repr(label_str), fontsize=8,
                      ha="center", va="bottom", color=col,
                      fontfamily="monospace", zorder=5, alpha=0.95,
                      bbox=dict(boxstyle="round,pad=0.15", fc=BG,
                                ec=col, lw=0.6, alpha=0.75))

# ── Edge weight labels on top-2 edges per anchor ─────────────────────────────
labelled = {grp: 0 for grp in GROUP_COLS}
for (u, v), w in sorted(edge_weights.items(), key=lambda x: -x[1]):
    grp = G[u][v]["group"]
    if labelled[grp] >= 2:
        continue
    xu, yu = pos[u]
    xv, yv = pos[v]
    mx, my = (xu+xv)/2, (yu+yv)/2
    col = GROUP_COLS[grp]["edge"]
    ax_graph.text(mx, my, f"{w:.3f}", fontsize=6.5, ha="center", va="center",
                  color=col, fontfamily="monospace", zorder=6, alpha=0.9,
                  bbox=dict(boxstyle="round,pad=0.1", fc=BG, ec="none", alpha=0.7))
    labelled[grp] += 1

# ── Semantic / morphological callout annotations ──────────────────────────────
# Paris → France (semantic) — callout offset points inward/right
france_id = " France_paris"
if france_id in pos:
    fx, fy = pos[france_id]
    ax_graph.annotate("semantic\nneighbour",
                      xy=(fx, fy), xytext=(fx + 0.55, fy - 0.45),
                      fontsize=7, color="#0fd9c0", ha="center",
                      arrowprops=dict(arrowstyle="->", color="#0fd9c0", lw=0.9),
                      fontfamily="monospace", zorder=7)

# king → queen (semantic) — callout offset points up/right (away from legend)
queen_id = " queen_king"
if queen_id in pos:
    qx, qy = pos[queen_id]
    ax_graph.annotate("semantic\ncrossover",
                      xy=(qx, qy), xytext=(qx + 0.6, qy + 0.5),
                      fontsize=7, color="#f5a623", ha="center",
                      arrowprops=dict(arrowstyle="->", color="#f5a623", lw=0.9),
                      fontfamily="monospace", zorder=7)

# neural → neurons (morphological) — callout offset points inward/left
neurons_id = " neurons_neural"
if neurons_id in pos:
    nx_, ny = pos[neurons_id]
    ax_graph.annotate("morphological\nvariant",
                      xy=(nx_, ny), xytext=(nx_ - 0.6, ny - 0.45),
                      fontsize=7, color="#c07aff", ha="center",
                      arrowprops=dict(arrowstyle="->", color="#c07aff", lw=0.9),
                      fontfamily="monospace", zorder=7)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    Line2D([0],[0], marker='o', color='none', markerfacecolor=GROUP_COLS[g]["anchor"],
           markersize=9, label=lbl)
    for g, lbl in [("paris","' Paris' group"),("neural","' neural' group"),
                   ("king","' king' group"),("run","' run' group")]
]
legend_items += [
    Line2D([0],[0], marker='o', color='none', markerfacecolor="#555577",
           markersize=6, label="neighbour node"),
    Line2D([0],[0], color="#888899", lw=2.5, label="edge  (width = weight)"),
]
ax_graph.legend(handles=legend_items, loc="lower center",
                ncol=3, fontsize=7.5, framealpha=0.5,
                facecolor=BG, edgecolor="#333355", labelcolor=C_TEXT,
                bbox_to_anchor=(0.5, -0.01))

ax_graph.set_xlim(-4.2, 4.2)
ax_graph.set_ylim(-4.2, 4.2)
ax_graph.set_aspect("equal")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Data Structure Schematic
# ══════════════════════════════════════════════════════════════════════════════
ax_schema.set_title("Graph as Data Structure", fontsize=11, color=C_TEXT,
                    pad=8, fontfamily="monospace")
ax_schema.set_xlim(0, 10); ax_schema.set_ylim(0, 10)

def sbox(ax, x, y, w, h, fc, ec, lw=1.0, alpha=1.0, radius=0.15):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=3, transform=ax.transData)
    ax.add_patch(b)

def stxt(ax, x, y, t, size=8.5, color=C_TEXT, ha="center", va="center",
         weight="normal", mono=True):
    ff = "monospace" if mono else "sans-serif"
    ax.text(x, y, t, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, fontfamily=ff, zorder=5, transform=ax.transData)

# Header
sbox(ax_schema, 0.4, 8.8, 9.2, 0.85, "#1a1a3a", "#4444aa", lw=1.5)
stxt(ax_schema, 5.0, 9.22, "graph_edges    [128 256  ×  32]   int32", size=9, weight="bold")
stxt(ax_schema, 5.0, 8.95, "graph_weights  [128 256  ×  32]   float32", size=8, color=C_SUBTEXT)

# Matrix schematic
ROW_H = 0.44
COL_W = 0.62
N_COLS_SHOW = 8
N_ROWS_SHOW = 10
MX = 0.5   # matrix left x
MY = 2.2   # matrix bottom y

# Column headers
for j in range(N_COLS_SHOW):
    stxt(ax_schema, MX + (j+0.5)*COL_W, MY + N_ROWS_SHOW*ROW_H + 0.25,
         f"[{j}]", size=6.5, color=C_SUBTEXT)
stxt(ax_schema, MX + (N_COLS_SHOW+0.5)*COL_W, MY + N_ROWS_SHOW*ROW_H + 0.25,
     "...", size=7, color=C_SUBTEXT)

# Rows: show row 0, 1, 2, ..., ellipsis, Paris row, ..., last row
paris_id, paris_nbr_ids, paris_nbr_strs, paris_nbr_wts = anchors_data["paris"]
neural_id = anchors_data["neural"][0]
king_id   = anchors_data["king"][0]
run_id    = anchors_data["run"][0]

# rows to show: first 4, "...", paris_id row, "...", last row
show_rows = [
    (0,       "token 0",        "#1e1e3e", "#333366", None),
    (1,       "token 1",        "#1e1e3e", "#333366", None),
    (2,       "token 2",        "#1e1e3e", "#333366", None),
    (None,    "...",             "#141428", "#222244", None),
    (king_id, f"tok {king_id}  (' king')",  "#1e2a1e", "#2a5a2a", "king"),
    (None,    "...",             "#141428", "#222244", None),
    (paris_id,f"tok {paris_id}  (' Paris')", "#1a2a2a", "#1a6a5a", "paris"),
    (None,    "...",             "#141428", "#222244", None),
    (neural_id,f"tok {neural_id}  (' neural')","#221a2a","#5a2a7a","neural"),
    (None,    "...",             "#141428", "#222244", None),
]

# Row labels on left
stxt(ax_schema, 0.25, MY + N_ROWS_SHOW*ROW_H + 0.25, "row", size=6.5, color=C_SUBTEXT)

for ri, (tid, row_label, fc, ec, grp) in enumerate(show_rows):
    y0 = MY + (N_ROWS_SHOW - 1 - ri) * ROW_H

    # Row label
    col_label = GROUP_COLS[grp]["anchor"] if grp else C_SUBTEXT
    stxt(ax_schema, 0.22, y0 + ROW_H*0.5, row_label,
         size=6, color=col_label, ha="right")

    if tid is None:
        # Ellipsis row
        sbox(ax_schema, MX, y0, N_COLS_SHOW*COL_W + 0.3, ROW_H*0.6, fc, ec, lw=0.5)
        stxt(ax_schema, MX + (N_COLS_SHOW*COL_W + 0.3)/2, y0 + ROW_H*0.3,
             "...  ...  ...  ...  ...  ...  ...", size=8, color=C_SUBTEXT)
    else:
        nbr_ids_row = edges[tid].tolist()
        nbr_wts_row = weights[tid].tolist()
        for j in range(N_COLS_SHOW):
            sbox(ax_schema, MX + j*COL_W, y0, COL_W*0.92, ROW_H*0.85, fc, ec, lw=0.8)
            cell_val = str(nbr_ids_row[j])
            stxt(ax_schema, MX + (j+0.46)*COL_W, y0 + ROW_H*0.42,
                 cell_val, size=6.2, color=C_TEXT if grp is None else GROUP_COLS[grp]["anchor"])
        # "..." at end
        sbox(ax_schema, MX + N_COLS_SHOW*COL_W, y0, COL_W*0.5, ROW_H*0.85,
             fc, ec, lw=0.5)
        stxt(ax_schema, MX + (N_COLS_SHOW+0.25)*COL_W, y0+ROW_H*0.42,
             "...", size=7, color=C_SUBTEXT)

# Callout: Paris row → token strings
paris_row_y = MY + (N_ROWS_SHOW - 1 - 6) * ROW_H + ROW_H*0.42
callout_x   = MX + N_COLS_SHOW*COL_W + 1.0
stxt(ax_schema, callout_x + 0.1, paris_row_y + 1.6,
     "Decoded neighbours:", size=7.5, color="#0fd9c0", ha="left", weight="bold")
for j, (ns, nw) in enumerate(zip(paris_nbr_strs[:6], paris_nbr_wts[:6])):
    y = paris_row_y + 1.2 - j * 0.36
    stxt(ax_schema, callout_x + 0.1, y,
         f"[{j}]  {repr(ns):18s}  {nw:.3f}", size=7, color="#0fd9c0", ha="left")

# Arrow from paris row to callout
ax_schema.annotate("",
    xy=(callout_x - 0.05, paris_row_y + 0.5),
    xytext=(MX + N_COLS_SHOW*COL_W + 0.35, paris_row_y + ROW_H*0.42),
    arrowprops=dict(arrowstyle="->", color="#0fd9c0", lw=1.2,
                    connectionstyle="arc3,rad=-0.3"),
    zorder=6, transform=ax_schema.transData)

# Stats at bottom
sbox(ax_schema, 0.4, 0.15, 9.2, 1.7, "#0d1a1a", "#0f9b8e", lw=1.3)
stxt(ax_schema, 5.0, 1.6,  "Total edges:    128 256 × 32  =  4 100 992  edges", size=8.5, weight="bold")
stxt(ax_schema, 5.0, 1.25, "Memory:         graph_edges   ~  494 MB  →  stored as int32", size=8, color=C_SUBTEXT)
stxt(ax_schema, 5.0, 0.90, "                graph_weights ~  494 MB  →  stored as float32", size=8, color=C_SUBTEXT)
stxt(ax_schema, 5.0, 0.52, "On-disk (mlp_transition_graph.pt):  31 MB  (compressed)", size=8, color="#0fd9c0")
stxt(ax_schema, 5.0, 0.25, "Build time: ~10 min on P100  |  loaded in <1 s at inference", size=7.5, color=C_SUBTEXT)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Inference Pipeline
# ══════════════════════════════════════════════════════════════════════════════
ax_pipeline.set_title("Online Inference: Routing at Each Decoding Step",
                      fontsize=11, color=C_TEXT, pad=6, fontfamily="monospace")
ax_pipeline.set_xlim(0, 20); ax_pipeline.set_ylim(0, 5.2)

def pbox(x, y, w, h, fc, ec, lw=1.5, radius=0.2):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
                       transform=ax_pipeline.transData)
    ax_pipeline.add_patch(b)

def ptxt(x, y, t, size=8.5, color=C_TEXT, ha="center", va="center", weight="normal"):
    ax_pipeline.text(x, y, t, fontsize=size, color=color, ha=ha, va=va,
                     fontweight=weight, fontfamily="monospace", zorder=5,
                     transform=ax_pipeline.transData)

def parrow(x1, y1, x2, y2, col="#aaaacc", lw=1.8):
    ax_pipeline.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                         mutation_scale=14),
                         zorder=4, transform=ax_pipeline.transData)

# Step boxes (x_left, label_top, label_bot, fc, ec, group_col)
steps = [
    (0.25, 2.3, 1.9, "hidden state\nh_T  [2048]",
     "last-token repr\nfrom LLM fwd pass", "#1a1a3a", "#6666cc", "#aaaaff"),
    (2.6, 2.3, 1.9, "Cosine\nSimilarity",
     "h_T · lm_head_norm.T\n→ scores [128 256]", "#1a2a1a", "#4a8a4a", "#88cc88"),
    (4.95, 2.3, 1.9, "Top-16\nAnchor Tokens",
     "16 vocab tokens\nmost similar to h_T", "#2a1a1a", "#8a4a2a", "#ffaa66"),
    (7.3,  2.3, 1.9, "1-hop\nGraph Walk",
     "graph_edges[anchors]\n→ 16×32 neighbours", "#1a1a2a", "#5a4a8a", "#bb99ff"),
    (9.65, 2.3, 1.9, "Union\nShortlist",
     "anchors + neighbours\n≤ 528 candidates", "#1a2a2a", "#2a7a7a", "#55ddcc"),
    (12.0, 2.3, 1.9, "lm_head\n[k × 2048]",
     "matmul over k tokens\ninstead of 128 256", "#1a2a1a", "#2a6a2a", "#66dd66"),
    (14.35,2.3, 1.9, "Next Token\nLogits [k]",
     "argmax / sample\n→ next token", "#2a1a2a", "#6a2a6a", "#dd88ff"),
]
BOX_W = 2.15; BOX_H = 1.55
for i, (bx, by, _, top_lbl, bot_lbl, fc, ec, tc) in enumerate(steps):
    pbox(bx, by, BOX_W, BOX_H, fc, ec)
    lines = top_lbl.split("\n")
    ptxt(bx + BOX_W/2, by + BOX_H - 0.38, lines[0], size=8.5, weight="bold", color=tc)
    if len(lines) > 1:
        ptxt(bx + BOX_W/2, by + BOX_H - 0.7, lines[1], size=7.5, color=tc)
    for j, bl in enumerate(bot_lbl.split("\n")):
        ptxt(bx + BOX_W/2, by + 0.55 - j*0.32, bl, size=7, color=C_SUBTEXT)
    # Arrow to next
    if i < len(steps) - 1:
        parrow(bx + BOX_W, by + BOX_H/2,
               steps[i+1][0], steps[i+1][2] + BOX_H/2,
               col=tc, lw=1.6)

# ── Vocab reduction bar chart ─────────────────────────────────────────────────
bar_x = 16.8; bar_y_base = 0.45; bar_w = 0.55

# Full vocab bar
full_h = 3.8
pbox(bar_x, bar_y_base, bar_w, full_h, "#2a2a4a", "#5555aa", lw=1.2)
ptxt(bar_x + bar_w/2, bar_y_base + full_h + 0.2,
     "128 256", size=7.5, color="#aaaaff")
ptxt(bar_x + bar_w/2, bar_y_base - 0.22, "full\nvocab", size=7, color=C_SUBTEXT)

# k=10000 bar
k_vals  = [10000, 5000, 2000]
k_cols  = ["#2a7a2a", "#5a7a2a", "#7a5a2a"]
k_ecols = ["#55cc55", "#99cc33", "#cc8833"]
for ki, (k, kc, ke) in enumerate(zip(k_vals, k_cols, k_ecols)):
    bx2 = bar_x + bar_w + 0.15 + ki * (bar_w + 0.1)
    kh  = full_h * k / 128256
    pbox(bx2, bar_y_base, bar_w, kh, kc, ke, lw=1.2)
    ptxt(bx2 + bar_w/2, bar_y_base + kh + 0.2, f"{k//1000}k", size=7.5, color=ke)
    ratio = 128256 / k
    ptxt(bx2 + bar_w/2, bar_y_base - 0.22, f"×{ratio:.0f}\nfewer", size=6.5, color=ke)

ptxt(bar_x + (bar_w + 0.1)*2 + bar_w/2, bar_y_base + full_h + 0.55,
     "lm_head FLOPs  (k=shortlist size)", size=8, color=C_TEXT, weight="bold")

# Brace-style arrow from "Union Shortlist" box down to bar chart
ax_pipeline.annotate("",
    xy=(bar_x + bar_w/2 + 0.5, bar_y_base + full_h * 10000/128256 + 0.15),
    xytext=(steps[4][0] + BOX_W/2, steps[4][2]),
    arrowprops=dict(arrowstyle="->", color="#55ddcc", lw=1.2, linestyle="dashed",
                    connectionstyle="arc3,rad=0.3"),
    zorder=4, transform=ax_pipeline.transData)

# ── Timing annotation ──────────────────────────────────────────────────────────
pbox(0.25, 0.1, 11.5, 1.0, "#0d1a0d", "#2a6a2a", lw=1.2)
ptxt(0.95, 0.72, "Graph walk overhead:", size=8, color=C_SUBTEXT, ha="left")
ptxt(0.95, 0.42, "cosine topk + 1-hop walk + union  ~  0.5 ms on CUDA", size=8,
     color="#66dd66", ha="left", weight="bold")
ptxt(0.95, 0.18, "lm_head saving at k=10 000:  full=6.2 ms  →  pruned~0.5 ms  (×12 speedup)",
     size=7.5, color=C_SUBTEXT, ha="left")

# ── Master title ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.965,
         "MLP Transition Graph Router — Structure & Inference",
         fontsize=15, color=C_TEXT, ha="center", fontweight="bold",
         fontfamily="monospace")
fig.text(0.5, 0.948,
         "128 256 tokens  ×  32 MLP-derived neighbours  =  4.1 M edges  |  "
         "Llama-3.2-1B-Instruct  |  WikiText-2",
         fontsize=9, color=C_SUBTEXT, ha="center", fontfamily="monospace")

# ── Save ───────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved → {OUT_PATH}")
