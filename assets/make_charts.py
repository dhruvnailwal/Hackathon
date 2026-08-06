import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT, exist_ok=True)

NAVY = "#0f2537"
TEAL = "#1f8a8c"
AMBER = "#e0a03d"
RED = "#c94f4f"
GREEN = "#3d9c63"
GREY = "#9aa5b1"
LIGHT = "#eef3f7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#8a97a5",
    "axes.labelcolor": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "text.color": NAVY,
    "figure.facecolor": "white",
})


def heatmap(path):
    rows = ["A/B Schema + Entity Res", "C Time correlation", "D Network correlation", "E Stat / ML outliers", "H Benford / structuring"]
    cols = ["Bank\nonly", "CDR\nonly", "Social\nonly", "Bank\n+CDR", "Bank\n+Social", "CDR\n+Social", "All\n3 sources"]
    # 2 = full, 1 = partial/degraded, 0 = blocked/weak
    data = np.array([
        [2, 2, 2, 2, 2, 2, 2],
        [1, 1, 0, 2, 2, 2, 2],
        [2, 2, 0, 2, 1, 1, 2],
        [2, 1, 1, 2, 2, 2, 2],
        [2, 0, 0, 2, 2, 0, 2],
    ])
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
    cmap = ["#c94f4f", "#e0a03d", "#3d9c63"]
    im = ax.imshow(data, cmap=matplotlib.colors.ListedColormap(cmap), aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            label = {2: "FULL", 1: "DEGRADED", 0: "BLOCKED"}[v]
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if v == 0 else "white",
                    fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(cols)), cols, fontsize=8.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=8.5)
    ax.set_title("Model activation per data-source combination", fontsize=12, color=NAVY, fontweight="bold", pad=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def coverage(path):
    models = ["C Time corr.", "D Network", "E Stat/ML", "Fusion ensemble"]
    recall = [0.62, 0.70, 0.68, 0.92]
    precision = [0.41, 0.55, 0.52, 0.71]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=200)
    bars = ax.bar(x - 0.2, recall, 0.35, label="Recall@10", color=TEAL)
    ax.bar(x + 0.2, precision, 0.35, label="Precision@10", color=AMBER)
    for b, v in zip(bars, recall):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.set_xticks(x, models, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Expected detection performance — single models vs fused ensemble\n(illustrative pre-evaluation targets, H1)", fontsize=11, color=NAVY, fontweight="bold", pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def flow(path):
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=9.5, weight="normal", tc=NAVY):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                           linewidth=1.4, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, color=NAVY):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                            linewidth=1.6, color=color)
        ax.add_patch(a)

    box(0.3, 4.7, 1.7, 0.9, "CDR / Bank /\nSocial CSVs", fc="#dbe7f0")
    box(0.3, 2.7, 1.7, 1.1, "Stage A\nDynamic schema\ndetection", fc="#dbe7f0")
    box(0.3, 0.7, 1.7, 1.1, "Stage B\nEntity\nresolution", fc="#dbe7f0")
    box(3.0, 2.2, 2.2, 1.9, "Unified long\nformat dataframe\n(entity x event x\nsource)", fc="#cfe8e6", ec=TEAL, weight="bold")
    box(6.4, 4.7, 3.1, 0.9, "C  Time correlation\n(merge_asof)", fc="#eaf3ea", ec=GREEN)
    box(6.4, 2.8, 3.1, 0.9, "D  Network correlation\n(graph models)", fc="#eaf3ea", ec=GREEN)
    box(6.4, 0.9, 3.1, 0.9, "E  Stat / ML outliers\n(IF / EIF / rules)", fc="#eaf3ea", ec=GREEN)
    arrow(2.0, 5.15, 3.0, 4.0)
    arrow(2.0, 3.25, 3.0, 3.4)
    arrow(2.0, 1.25, 3.0, 2.9)
    arrow(5.2, 3.15, 6.4, 5.15)
    arrow(5.2, 3.15, 6.4, 3.25)
    arrow(5.2, 3.15, 6.4, 1.35)
    box(6.4, 5.6, 3.1, 0.0, "")
    box(8.6, 5.6, 0.0, 0.0, "")
    box(0.3, 5.8, 9.3, 0.0, "")
    # sufficiency loop
    box(3.0, 0.15, 2.2, 1.1, "Sufficiency engine\n(SUPPORTED /\nDEGRADED / BLOCKED)", fc="#fdeede", ec=AMBER, fs=8.5)
    arrow(3.0, 0.7, 2.0, 0.7, color=AMBER)
    arrow(5.2, 0.7, 6.4, 0.7, color=AMBER)
    box(6.4, 5.9, 3.1, 0.0, "")
    ax.text(8.05, 5.95, "F  Fusion → ranking   |   G  Explanation   |   Dashboard/API", ha="center",
            fontsize=9.5, fontweight="bold", color=NAVY)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def gantt(path):
    tasks = [
        ("0-2", "Schema freeze + aliases + storylines", 0, 2, TEAL),
        ("2-8", "Synthetic generator + surprise injector", 2, 6, TEAL),
        ("8-13", "Stage A schema detection / B entity resolution", 8, 5, TEAL),
        ("13-16", "Unified pipeline + sufficiency engine", 13, 3, AMBER),
        ("16-24", "Parallel: C time / D network / E stat-ML / dashboard", 16, 8, AMBER),
        ("24-28", "Fusion + explanations + E2E wiring", 24, 4, TEAL),
        ("28-30", "Validation vs answer key", 28, 2, GREEN),
        ("30-34", "Polish + demo prep", 30, 4, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.4), dpi=200)
    for i, (name, label, start, dur, color) in enumerate(tasks[::-1]):
        ax.barh(i, dur, left=start, height=0.55, color=color, alpha=0.9)
        ax.text(start + 0.2, i, label, va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 34)
    ax.set_xticks(range(0, 35, 2))
    ax.set_xlabel("Hours", fontsize=9)
    ax.set_title("Execution roadmap — parallel tracks unlock the team", fontsize=12, color=NAVY, fontweight="bold", pad=8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sufficiency_flow(path):
    fig, ax = plt.subplots(figsize=(8.8, 2.8), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    linewidth=1.4, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=NAVY, fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=NAVY))

    box(0.2, 1.0, 2.4, 1.0, "Resolved field\nslots per model", "#dbe7f0", NAVY, 8.5)
    box(3.4, 1.0, 2.2, 1.0, "required vs\navailable?", "#fdeede", AMBER, 8.5)
    box(6.6, 2.0, 3.0, 0.8, "SUPPORTED → run model", "#eaf3ea", GREEN, 9)
    box(6.6, 1.0, 3.0, 0.8, "DEGRADED → run + ⚠ warn\nwhich features were dropped", "#fdeede", AMBER, 8.5)
    box(6.6, 0.0, 3.0, 0.8, "BLOCKED → skip + tell user\nexactly what to add", "#fbe8e8", RED, 8.5)
    arrow(2.6, 1.5, 3.4, 1.5)
    arrow(5.6, 1.5, 6.6, 2.4)
    arrow(5.6, 1.5, 6.6, 1.4)
    arrow(5.6, 1.5, 6.6, 0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


heatmap(os.path.join(OUT, "activation_heatmap.png"))
coverage(os.path.join(OUT, "coverage.png"))
flow(os.path.join(OUT, "architecture.png"))
gantt(os.path.join(OUT, "roadmap.png"))
sufficiency_flow(os.path.join(OUT, "sufficiency.png"))
print("charts done:", os.listdir(OUT))
