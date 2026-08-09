"""Report charts — matplotlib renders for the colourful analysis report.

All functions render PNGs into a report directory and are safe on empty /
tiny inputs (they still produce a valid figure). Charts follow the same
"frosted glass" palette as the desktop UI.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MINT = "#3fbf8f"
SKY = "#5aa7e8"
TEAL = "#35c7c2"
AMBER = "#e0a03d"
RED = "#e0566b"
NAVY = "#1e3b33"
MUTED = "#6b857d"
LIGHT = "#eef5f1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#8a97a5",
    "axes.labelcolor": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "text.color": NAVY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

SOURCE_COLORS = {"bank": MINT, "cdr": SKY, "social": AMBER}


def _guard(ax):
    """Tidy spines and tight layout for a finished axes."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def render_timeline(result, out_dir: str | Path, name: str = "timeline.png") -> Path:
    """Event volume over time, stacked by source."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df = result.unified
    fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=160)
    if df is None or df.empty or "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        ax.text(0.5, 0.5, "no timestamped events to plot", ha="center", va="center",
                fontsize=11, color=MUTED, transform=ax.transAxes)
        ax.axis("off")
    else:
        ts = df.dropna(subset=["timestamp"]).copy()
        ts["date"] = pd.to_datetime(ts["timestamp"]).dt.floor("D")
        pivot = ts.pivot_table(index="date", columns="source", values="entity_id",
                               aggfunc="count", fill_value=0)
        pivot = pivot.resample("D").sum().fillna(0)
        colors = [SOURCE_COLORS.get(c, SKY) for c in pivot.columns]
        pivot.plot.area(ax=ax, color=colors, alpha=0.75)
        ax.set_ylabel("records")
        ax.set_title("When the activity happened", fontsize=12, fontweight="bold", color=NAVY, pad=8)
        _guard(ax)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_model_scores(result, out_dir: str | Path, name: str = "models.png") -> Path:
    """Per-model peak/coverage bars — all model results at a glance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    scores = result.model_scores or {}
    fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=160)
    if not scores:
        ax.text(0.5, 0.5, "no model scores to plot", ha="center", va="center",
                fontsize=11, color=MUTED, transform=ax.transAxes)
        ax.axis("off")
    else:
        models = sorted(scores)
        peaks = [max(scores[m].values()) if scores[m] else 0.0 for m in models]
        flagged = [sum(1 for v in scores[m].values() if v > 0.6) for m in models]
        x = np.arange(len(models))
        ax.bar(x - 0.2, peaks, 0.4, label="peak score", color=TEAL)
        ax.bar(x + 0.2, flagged, 0.4, label="entities flagged (>0.6)", color=AMBER)
        ax.set_xticks(x, [m.replace("_", "\n") for m in models], fontsize=7.5)
        ax.set_ylim(0, max(1.0, max(peaks + [0.0]) * 1.15))
        ax.set_title("Model results — peak score and flagged-entity count per model",
                     fontsize=12, fontweight="bold", color=NAVY, pad=8)
        _guard(ax)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_person_attention(result, out_dir: str | Path, name: str = "people.png") -> Path:
    """Who stands out — person names vs number of reasons found.

    Plain-language rendering: bar width = how many concrete reasons we
    found for watching this person, coloured by how strongly they stand out.
    """
    from .insights import build_insights

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig, ax = plt.subplots(figsize=(9.6, 4.0), dpi=160)
    top = build_insights(result, top_k=10)
    if not top:
        ax.text(0.5, 0.5, "no people stood out this run", ha="center", va="center",
                fontsize=11, color=MUTED, transform=ax.transAxes)
        ax.axis("off")
    else:
        labels = [x["name"] for x in reversed(top)]
        vals = [max(1, len(x["evidence"])) for x in reversed(top)]
        colors = [RED if x["risk"] == "HIGH" else (AMBER if x["risk"] == "MEDIUM" else SKY)
                  for x in reversed(top)]
        ax.barh(np.arange(len(vals)), vals, color=colors, alpha=0.9)
        ax.set_yticks(np.arange(len(vals)), labels, fontsize=8.5)
        ax.set_xlim(0, max(vals) * 1.2)
        ax.set_xlabel("signals we found")
        ax.set_title("People who stand out the most", fontsize=12, fontweight="bold",
                     color=NAVY, pad=8)
        _guard(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_sources(result, out_dir: str | Path, name: str = "sources.png") -> Path:
    """Per-source row / entity counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    per_source = getattr(result, "per_source", None) or []
    fig, ax = plt.subplots(figsize=(9.6, 3.0), dpi=160)
    if not per_source:
        ax.text(0.5, 0.5, "no source files loaded", ha="center", va="center",
                fontsize=11, color=MUTED, transform=ax.transAxes)
        ax.axis("off")
    else:
        names = [rs.source for rs in per_source]
        rows = [rs.n_rows for rs in per_source]
        ents = [rs.n_entities for rs in per_source]
        x = np.arange(len(names))
        ax.bar(x - 0.2, rows, 0.4, label="records", color=SKY)
        ax.bar(x + 0.2, ents, 0.4, label="people", color=MINT)
        ax.set_xticks(x, names, fontsize=10)
        ax.set_title("Where the records came from",
                     fontsize=12, fontweight="bold", color=NAVY, pad=8)
        _guard(ax)
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _entity_display_name(result, entity_id: str) -> str:
    """Short human label: the best known name, else the id."""
    from .insights import display_name
    return display_name(result, entity_id)


def render_ego_graph(result, out_dir: str | Path, top_k: int = 3,
                     name: str = "network.png") -> Path:
    """Ego-graph of the top people — who each flagged person talks to.

    One panel per top person: the person is the red hub, each neighbour is a
    dot coloured by the source of the interaction, dot size scales with how
    much contact there was, and thick lines mean heavy flow. Everything is
    visible in plain terms — no model jargon.
    """
    import networkx as nx

    from .insights import build_insights

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    top = build_insights(result, top_k=top_k)
    if not top:
        fig, ax = plt.subplots(figsize=(9.6, 3.2), dpi=160)
        ax.text(0.5, 0.5, "no flagged people to map", ha="center", va="center",
                fontsize=11, color=MUTED, transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    df = result.unified.dropna(subset=["entity_id", "counterparty_id"]).copy()
    df["entity_id"] = df["entity_id"].astype(str)
    df["counterparty_id"] = df["counterparty_id"].astype(str)
    wgt = df.groupby(["entity_id", "counterparty_id", "source"]).size()
    n_panels = len(top)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.6, 3.2 * nrows), dpi=160)
    axes = np.atleast_1d(axes).ravel()

    for ax, it in zip(axes, top):
        eid = str(it["entity_id"])
        sub = wgt.loc[eid] if eid in wgt.index.get_level_values(0) else None
        G = nx.Graph()
        G.add_node(eid, weight=1.0)
        edge_color, node_sizes = [], []
        if sub is not None:
            for (cp, src), w in sub.items():
                if cp == eid or w <= 0:
                    continue
                G.add_edge(eid, cp, weight=float(w), source=str(src))
        if G.number_of_edges() == 0:
            ax.text(0.5, 0.5, "no contacts recorded", ha="center", va="center",
                    fontsize=10, color=MUTED, transform=ax.transAxes)
            ax.axis("off")
            ax.set_title(_entity_display_name(result, eid), fontsize=9,
                         fontweight="bold", color=NAVY)
            continue
        pos = nx.spring_layout(G, seed=7, k=0.9 / np.sqrt(max(G.number_of_nodes(), 1)))
        ews = [G[u][v]["weight"] for u, v in G.edges()]
        elw = 1.5 + 4.0 * (np.asarray(ews) - min(ews)) / max(1e-9, max(ews) - min(ews))
        nx.draw_networkx_edges(G, pos, ax=ax, width=elw, edge_color=MUTED, alpha=0.55)
        for node in G.nodes():
            if node == eid:
                node_sizes.append(560)
                continue
            counts = {}
            for nbr in G.neighbors(node):
                if nbr != eid:
                    counts[nbr] = counts.get(nbr, 0) + 1
            edge_w = G[eid][node]["weight"]
            source = G[eid][node]["source"]
            color = SOURCE_COLORS.get(source, SKY)
            node_sizes.append(min(380, 40 + 18 * int(edge_w)))
            ax.scatter([pos[node][0]], [pos[node][1]], s=node_sizes[-1],
                       color=color, alpha=0.85, edgecolors="white", linewidths=0.8,
                       zorder=3)
            ax.annotate(node, pos[node], fontsize=6, color=NAVY, alpha=0.85,
                        ha="center", va="center", zorder=4)
        ax.scatter([pos[eid][0]], [pos[eid][1]], s=node_sizes[0], color=RED,
                   alpha=0.95, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate("flagged", pos[eid], fontsize=7, fontweight="bold", color=RED,
                    ha="center", va="center", zorder=6)
        ax.set_title(_entity_display_name(result, eid), fontsize=9.5,
                     fontweight="bold", color=NAVY)
        ax.axis("off")
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle("Who the flagged people are connected to",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_entity_timeline(result, out_dir: str | Path, entity_id: str,
                           name: str | None = None) -> Path:
    """Per-person drill-down: one person's activity over time.

    Each event is a dot — colour = which source it came from, dot size =
    how much money moved (when an amount exists). Below the dots a thin bar
    shows how many events happened each day, so quiet and frantic stretches
    both show up at a glance.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"entity_{entity_id}.png"
    path = out_dir / name
    eid = str(entity_id)
    df = result.unified[result.unified["entity_id"].astype(str) == eid].copy()
    fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=160)
    if df.empty or "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        ax.text(0.5, 0.5, "no events recorded for this person", ha="center",
                va="center", fontsize=11, color=MUTED, transform=ax.transAxes)
        ax.axis("off")
    else:
        sub = df.dropna(subset=["timestamp"]).copy()
        sub["t"] = pd.to_datetime(sub["timestamp"])
        sub = sub.sort_values("t")
        amt = pd.to_numeric(sub["amount"], errors="coerce")
        size = 12 + 44 * np.clip(amt.fillna(0).abs() / max(1.0, amt.abs().max()), 0, 1)
        colors = [SOURCE_COLORS.get(s, SKY) for s in sub["source"]]
        ax.scatter(sub["t"], np.zeros(len(sub)) + 0.5, s=size, color=colors,
                   alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3)
        daily = sub.groupby(sub["t"].dt.floor("D")).size()
        ax.bar(daily.index, daily.values, width=0.8, color=MINT, alpha=0.35,
               zorder=2)
        ax.set_ylim(0, 1.6)
        ax.set_yticks([])
        ax.set_xlabel("time")
        ax.set_title(f"Every event for {_entity_display_name(result, eid)}",
                     fontsize=12, fontweight="bold", color=NAVY, pad=8)
        _guard(ax)
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                              color=SOURCE_COLORS.get(s, SKY), label=s)
                   for s in sorted(set(sub["source"]))]
        ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right")
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_location_map(result, out_dir: str | Path, name: str = "map.png") -> Path:
    """Activity map — where flagged activity actually happened.

    Locations (branches / terminals / towers) are laid out like a map: each
    dot is a place, its size is how much activity happened there, and dots
    turn red when a flagged person was involved. Plain geography-style view
    with no model jargon.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df = result.unified.dropna(subset=["location"]).copy()
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=160)
    if df.empty:
        ax.text(0.5, 0.5, "no location data in these files", ha="center",
                va="center", fontsize=11, color=MUTED, transform=ax.transAxes)
        ax.axis("off")
    else:
        from .insights import build_insights
        flagged = {str(it["entity_id"]) for it in build_insights(result, top_k=8)}
        df["loc"] = df["location"].astype(str)
        agg = df.groupby("loc").agg(
            n=("entity_id", "size"),
            flagged=("entity_id", lambda s: int(s.astype(str).isin(flagged).any())),
        ).reset_index()
        agg = agg.sort_values("n", ascending=False).head(200).reset_index(drop=True)
        # deterministic pseudo-geography from the location id
        seed = 11
        rng = np.random.default_rng(seed)
        hashes = rng.uniform(0, 1, size=len(agg))
        xs = hashes * agg["n"].rank(pct=True) * 1.6
        ys = rng.uniform(0, 1, size=len(agg))
        sizes = 30 + 140 * (agg["n"] / agg["n"].max())
        for _, row in agg.iterrows():
            ax.scatter(xs[row.name], ys[row.name], s=sizes[row.name],
                       color=RED if row["flagged"] else TEAL, alpha=0.75,
                       edgecolors="white", linewidths=0.7)
        flagged_locs = agg[agg["flagged"] == 1]
        for _, row in flagged_locs.head(10).iterrows():
            ax.annotate(row["loc"], (xs[row.name], ys[row.name]), fontsize=6.5,
                        color=RED, fontweight="bold", ha="center", va="center")
        ax.scatter([], [], s=80, color=TEAL, label="busy places", alpha=0.75)
        ax.scatter([], [], s=80, color=RED, label="places a flagged person used",
                   alpha=0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("The activity map — where things happened",
                     fontsize=12, fontweight="bold", color=NAVY, pad=8)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
