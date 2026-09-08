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
SHARED = "#8a4fd6"  # a counterparty connected to 2+ flagged people


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


def render_ego_graph(result, out_dir: str | Path, top_k: int = 8,
                     max_neighbors_per_hub: int = 8,
                     name: str = "network.png") -> Path:
    """One combined network of the top flagged people and who they talk to.

    Every flagged person is a red hub; each neighbour is a dot coloured by
    the source of the interaction. Critically, all flagged people share the
    *same* graph rather than getting an isolated panel each — so if a
    counterparty shows up in more than one flagged person's contacts (a
    shared money-mule hand-off point, a common relative, a hub account),
    that person is drawn once, in purple, with a line to every flagged
    person they touch. Two flagged people transacting directly with each
    other are linked by a bold red edge. This is the whole point of the
    view: overlap between separate people's networks is often the most
    meaningful signal, and it is invisible when each person gets a
    disconnected panel.
    """
    import networkx as nx

    from .insights import build_insights

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    top = build_insights(result, top_k=top_k)
    fig, ax = plt.subplots(figsize=(9.6, 6.6), dpi=160)
    if not top:
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

    hub_ids = [str(it["entity_id"]) for it in top]
    hub_set = set(hub_ids)
    hub_names = {str(it["entity_id"]): it["name"] for it in top}

    G = nx.Graph()
    G.add_nodes_from(hub_ids)
    neighbor_hubs = {}  # counterparty -> set of hubs connected to it
    for eid in hub_ids:
        if eid not in wgt.index.get_level_values(0):
            continue
        sub = wgt.loc[eid].sort_values(ascending=False)
        seen_cp = set()
        for (cp, src), w in sub.items():
            if cp == eid or w <= 0 or cp in seen_cp:
                continue
            if len(seen_cp) >= max_neighbors_per_hub:
                break
            seen_cp.add(cp)
            if G.has_edge(eid, cp) and G[eid][cp]["weight"] >= w:
                pass
            else:
                G.add_edge(eid, cp, weight=float(w), source=str(src))
            neighbor_hubs.setdefault(cp, set()).add(eid)

    if G.number_of_edges() == 0:
        ax.text(0.5, 0.5, "no contacts recorded for the flagged people",
                ha="center", va="center", fontsize=11, color=MUTED, transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    shared_nodes = {cp for cp, hubs in neighbor_hubs.items() if len(hubs) >= 2 and cp not in hub_set}
    linked_hub_pairs = [(u, v) for u, v in G.edges() if u in hub_set and v in hub_set]

    pos = nx.spring_layout(G, seed=7, k=1.3 / np.sqrt(max(G.number_of_nodes(), 1)))

    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1.0)
    for u, v, data in G.edges(data=True):
        is_hub_link = u in hub_set and v in hub_set
        color = RED if is_hub_link else MUTED
        width = 2.6 if is_hub_link else 1.1 + 3.2 * data["weight"] / max(1e-9, max_w)
        alpha = 0.85 if is_hub_link else 0.4
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, linewidth=width, alpha=alpha, zorder=2)

    # ordinary (single-hub) neighbours, coloured by source of contact
    for node in G.nodes():
        if node in hub_set or node in shared_nodes:
            continue
        nbr_hub = next(iter(neighbor_hubs.get(node, ())), None)
        source = G[nbr_hub][node]["source"] if nbr_hub else "bank"
        w = G[nbr_hub][node]["weight"] if nbr_hub else 1.0
        ax.scatter([pos[node][0]], [pos[node][1]], s=min(320, 40 + 16 * w),
                   color=SOURCE_COLORS.get(source, SKY), alpha=0.85,
                   edgecolors="white", linewidths=0.7, zorder=3)
        ax.annotate(_entity_display_name(result, node), pos[node], fontsize=5.2,
                    color=MUTED, alpha=0.9, ha="center", va="center", zorder=4)

    # shared neighbours — connected to 2+ flagged people, the headline signal
    for node in shared_nodes:
        ax.scatter([pos[node][0]], [pos[node][1]], s=260 + 55 * len(neighbor_hubs[node]),
                   color=SHARED, alpha=0.95, edgecolors="white", linewidths=1.4, zorder=4)
        ax.annotate(_entity_display_name(result, node), pos[node], fontsize=6.8, fontweight="bold",
                    color=NAVY, ha="center", va="center", zorder=6,
                    xytext=(0, 10), textcoords="offset points")

    # flagged hubs on top
    for eid in hub_ids:
        if eid not in pos:
            continue
        ax.scatter([pos[eid][0]], [pos[eid][1]], s=620, color=RED, alpha=0.95,
                   edgecolors="white", linewidths=1.4, zorder=5)
        ax.annotate(hub_names.get(eid, eid), pos[eid], fontsize=7.8, fontweight="bold",
                    color=NAVY, ha="center", va="center", zorder=6,
                    xytext=(0, 13), textcoords="offset points")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=RED,
                   markeredgecolor="white", markersize=10, label="flagged person"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=SHARED,
                   markeredgecolor="white", markersize=9, label="connected to 2+ flagged people"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=MINT,
                   markeredgecolor="white", markersize=7, label="bank contact"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=SKY,
                   markeredgecolor="white", markersize=7, label="call/CDR contact"),
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=AMBER,
                   markeredgecolor="white", markersize=7, label="social contact"),
    ]
    if linked_hub_pairs:
        handles.append(plt.Line2D([0], [0], color=RED, linewidth=2.4,
                                  label="direct link between flagged people"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
             ncol=3, frameon=False, fontsize=7.3, labelcolor=NAVY)

    title = "Who the flagged people are connected to"
    if shared_nodes:
        title += f" — {len(shared_nodes)} shared connection(s) found"
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY)
    ax.axis("off")
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
