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
