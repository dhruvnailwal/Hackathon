"""Real-data charts for the 4-slide pitch deck.

All performance numbers are copied from the deterministic evaluation snapshot
(results/current_baseline/eval.md + eval.py output): recall@10 / precision@10,
H1 standalone per-model checks, and the leave-one-model-out ablation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def aml_scale(path):
    """UN scale fact + FATF seizure gap: the problem in two bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200)
    # left: laundered share of global GDP
    gdp = 105.0  # trillion USD, illustrative
    lo, hi = 0.8, 2.0
    ax1.barh(0, gdp, height=0.5, color=LIGHT, edgecolor=NAVY)
    ax1.barh(0, hi, height=0.5, left=0, color=RED, alpha=0.85)
    ax1.barh(0, lo, height=0.5, left=0, color=NAVY)
    ax1.text(hi + 1.5, 0, f"$0.8\u20132.0T laundered\nper year (2\u20135% of global GDP)",
             va="center", ha="left", fontsize=9.5, color=NAVY, fontweight="bold")
    ax1.text(0.3, 0, "$", va="center", ha="left", fontsize=8, color="white", fontweight="bold")
    ax1.set_xlim(0, 120)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_xticklabels(["$0", "$20T", "$40T", "$60T", "$80T", "$100T"], fontsize=8)
    ax1.set_yticks([])
    ax1.set_title("Scale: laundered money vs world GDP", fontsize=10.5, color=NAVY, fontweight="bold")
    for s in ["top", "right", "left"]:
        ax1.spines[s].set_visible(False)
    # right: seized vs laundered
    seized, laundered = 1.0, 100.0
    ax2.barh([1, 0], [laundered, seized], height=0.5, color=[GREY, GREEN])
    for i, (v, lab) in enumerate([(seized, "< 1% of proceeds"), (laundered, "every year")]):
        ax2.text(v + 1.0, i, lab, va="center", fontsize=9.5, color=NAVY, fontweight="bold")
    ax2.set_xlim(0, 118)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Deterrence gap: what is seized (FATF)", fontsize=10.5, color=NAVY, fontweight="bold")
    for s in ["top", "right", "left", "bottom"]:
        ax2.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def schema_chaos(path):
    """Column-name chaos: why no fixed parser can work."""
    cols = ["Attribute", "Bank file A", "Bank file B", "CDR file", "Social export"]
    rows = [
        ["timestamp", "txn_date", "posted_at", "call_time", "created_utc"],
        ["counterparty", "account_id", "beneficiary", "msisdn", "@mentions"],
        ["value", "debit / credit", "amount", "\u2014", "\u2014"],
        ["location", "branch_code", "\u2014", "tower_id", "geo_tag"],
    ]
    fig, ax = plt.subplots(figsize=(9.6, 3.1), dpi=200)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#8a97a5")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#f4f8fa")
            cell.get_text().set_color(NAVY)
        if c == 0 and r > 0:
            cell.get_text().set_fontweight("bold")
    ax.set_title("One logical event, four export formats \u2014 a hardcoded parser dies on the first new file",
                 fontsize=10.5, color=NAVY, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def eval_fusion(path):
    """Real measured: recall@10 / precision@10 per single model vs fused ensemble."""
    models = ["time", "network", "statml", "benford", "structuring", "behavioral", "chain", "FUSED"]
    recall = [0.143, 0.333, 0.143, 0.190, 0.286, 0.286, 0.143, 0.429]
    precision = [None, None, None, None, None, None, None, 0.900]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9.6, 3.6), dpi=200)
    ax.bar(x[:7], recall[:7], 0.62, color="#b8c6cf", label="recall@10 (single model)")
    ax.bar(7, recall[7], 0.62, color=TEAL, label="recall@10 (fused ensemble)")
    ax.bar(7.35, 0.900, 0.2, color=AMBER, label="precision@10 (fused)")
    for i, v in enumerate(recall[:7]):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=NAVY)
    ax.text(7, 0.429 + 0.02, "0.43", ha="center", fontsize=10, fontweight="bold", color=TEAL)
    ax.text(7.55, 0.90 + 0.02, "0.90", ha="center", fontsize=10, fontweight="bold", color=AMBER)
    ax.annotate("+29% recall over the\nbest single model",
                xy=(7, 0.429), xytext=(4.1, 0.80),
                fontsize=9.5, color=NAVY, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    ax.set_xticks(x, models, fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall@10  (21 truth anomalies, top-10 view)", fontsize=9)
    ax.set_title("Measured (deterministic eval snapshot): no single model beats the ensemble (H1)",
                 fontsize=11, color=NAVY, fontweight="bold", pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ablation(path):
    """Leave-one-model-out recall@10: every family adds signal."""
    models = ["full ensemble", "\u2212time corr.", "\u2212network", "\u2212statml",
              "\u2212benford", "\u2212structuring", "\u2212behavioral", "\u2212chain"]
    vals = [0.429, 0.333, 0.238, 0.333, 0.333, 0.333, 0.333, 0.333]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    colors = [TEAL] + [AMBER] * 7
    bars = ax.barh(range(len(models)), vals, height=0.58, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_yticks(range(len(models)), models, fontsize=9.5)
    ax.set_xlim(0, 0.55)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_title("Ablation: recall@10 when one family is removed", fontsize=10.5,
                 color=NAVY, fontweight="bold", pad=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def model_zoo(path):
    """Candidates behind one interface: the model zoo per family."""
    fams = ["time\ncorr.", "network", "stat / ML", "benford", "struct.", "behavior", "chain", "fusion"]
    counts = [3, 8, 13, 3, 2, 1, 1, 5]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    bars = ax.bar(range(len(fams)), counts, 0.62, color=TEAL)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v), ha="center",
                fontsize=10, fontweight="bold", color=NAVY)
    ax.set_xticks(range(len(fams)), fams, fontsize=9)
    ax.set_ylim(0, 15)
    ax.set_title(f"Model zoo: {sum(counts)} candidates, one interface, best kept",
                 fontsize=10.5, color=NAVY, fontweight="bold", pad=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def targets(path):
    """Proposed outcome: targets only (no measured values)."""
    labels = ["recall@10 \u00b7 predefined storylines",
              "recall@10 \u00b7 surprise hold-out",
              "precision@10 \u00b7 false-alert control",
              "fusion vs best single model",
              "schema adoption \u00b7 renamed files"]
    vals = [0.95, 0.80, 0.60, 0.15, 1.00]
    fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=200)
    colors = [TEAL, TEAL, AMBER, AMBER, GREEN]
    bars = ax.barh(range(len(vals)), vals, height=0.55, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2,
                f"\u2265 {v:.2f}" if v < 1.0 else "100%",
                va="center", fontsize=10, fontweight="bold", color=NAVY)
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=8.5)
    ax.set_xlabel("normalized target (recall / precision / lift / adoption)", fontsize=8.5)
    ax.set_title("Proposed outcome \u2014 every target is proven by the answer-key harness",
                 fontsize=11, color=NAVY, fontweight="bold", pad=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def real_world_validation(path):
    """Real (not illustrative) results from scripts/real_dataset_samld.py, run
    TWICE at different fraud-prevalence levels — because a single
    "precision@10 = 1.00" number doesn't survive scrutiny on its own. Both
    validation samples are still enriched well above SAML-D's true 0.104%
    fraud rate (the volume gate forces this — see the script's module
    docstring), so neither panel is a real-world deployment precision claim.
    The point is the middle panel: precision degrades gracefully as
    prevalence drops, rather than collapsing — evidence of real ranking
    signal, not a lucky fluke at one enriched setting."""
    ks = [5, 10, 20, 50, 100, 200, 500, 1000]
    curated = [1.00, 1.00, 0.95, 0.70, 0.59, 0.565, 0.372, 0.264]   # 21.7% prevalence
    harder = [1.00, 0.80, 0.50, 0.38, 0.27, 0.225, 0.168, 0.154]    # 8.8% prevalence
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.6, 3.3), dpi=200,
                                          gridspec_kw={"width_ratios": [1.4, 1.4, 1.0]})

    x = np.arange(len(ks))
    ax1.plot(x, curated, marker="o", color=TEAL, linewidth=2, label="curated (21.7% prevalence)")
    ax1.plot(x, harder, marker="o", color=AMBER, linewidth=2, label="harder (8.8% prevalence)")
    ax1.set_xticks(x, [str(k) for k in ks], fontsize=8.5)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel("top-k ranked entities", fontsize=9)
    ax1.set_title("precision@k — degrades gracefully,\nnot a lucky fluke", fontsize=10,
                  color=NAVY, fontweight="bold")
    ax1.legend(fontsize=7.5, loc="upper right", frameon=False)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    # zoomed callout on the k=10 comparison
    labels = ["curated\n21.7% prev.", "harder\n8.8% prev."]
    vals10 = [curated[1], harder[1]]
    bars = ax2.bar(labels, vals10, color=[TEAL, AMBER], width=0.55)
    for b, v in zip(bars, vals10):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha="center", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("precision@10, side by side", fontsize=10, color=NAVY, fontweight="bold")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    # prevalence context: why the numbers differ, and how far both still
    # are from SAML-D's real, full-dataset base rate
    prev_labels = ["curated\nsample", "harder\nsample", "SAML-D\ntrue rate"]
    prev_vals = [21.7, 8.8, 0.104]
    bars3 = ax3.bar(prev_labels, prev_vals, color=[TEAL, AMBER, GREY], width=0.55)
    ax3.set_yscale("log")
    for b, v in zip(bars3, prev_vals):
        ax3.text(b.get_x() + b.get_width() / 2, v * 1.3, f"{v:g}%",
                 ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax3.set_title("fraud prevalence\n(log scale)", fontsize=10, color=NAVY, fontweight="bold")
    for s in ["top", "right"]:
        ax3.spines[s].set_visible(False)

    fig.suptitle("Validated blind against SAML-D — an external dataset never used in development",
                 fontsize=10.5, color=NAVY, y=1.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


aml_scale(os.path.join(OUT, "deck_aml_scale.png"))
schema_chaos(os.path.join(OUT, "deck_schema_chaos.png"))
eval_fusion(os.path.join(OUT, "deck_eval_fusion.png"))
ablation(os.path.join(OUT, "deck_ablation.png"))
model_zoo(os.path.join(OUT, "deck_model_zoo.png"))
targets(os.path.join(OUT, "deck_targets.png"))
real_world_validation(os.path.join(OUT, "deck_real_world_validation.png"))
print("deck charts done:", [f for f in os.listdir(OUT) if f.startswith("deck_")])
