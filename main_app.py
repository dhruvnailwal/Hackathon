"""Anomaly Lens - minimalist PySide6 desktop dashboard.

Frosted-glass (glassmorphism) panels over a light green / light blue
crystalline background. Runs the AML multi-model anomaly pipeline and renders:
  1. pipeline summary stats
  2. per-model sufficiency verdicts (SUPPORTED / DEGRADED / BLOCKED)
  3. fused ranked insights with explanations

Run:
    python main_app.py                  # desktop window
    python main_app.py --shot ui.png    # render to PNG and exit
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QRadialGradient, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsBlurEffect,
    QGraphicsDropShadowEffect,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------
MINT = "#3fbf8f"
SKY = "#5aa7e8"
TEAL = "#35c7c2"
INK = "#1e3b33"
MUTED = "#6b857d"
WHITE = "#ffffff"

STATUS_COLORS = {
    "SUPPORTED": MINT,
    "DEGRADED": "#e6a23c",
    "BLOCKED": "#e0566b",
}

QSS = """
QWidget#Root {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #e9f8f1, stop:0.5 #e6f1f9, stop:1 #dcedf7);
}
QLabel { color: #1e3b33; background: transparent; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: rgba(255,255,255,60); width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: rgba(120,180,160,110); border-radius: 4px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }

#Brand { font-size: 30px; font-weight: 800; letter-spacing: 0.5px; }
#Tag   { font-size: 13px; color: #5f7a72; }

#GlassPanel {
    background: rgba(255, 255, 255, 0.42);
    border: 1px solid rgba(255, 255, 255, 0.65);
    border-radius: 22px;
}
#GlassPanelStrong {
    background: rgba(255, 255, 255, 0.58);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-radius: 22px;
}

#SectionTitle { font-size: 15px; font-weight: 700; color: #25483f; }
#SectionSub   { font-size: 12px; color: #6b857d; }

QPushButton#RunBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #3fbf8f, stop:1 #5aa7e8);
    color: white; border: none; border-radius: 14px;
    font-weight: 700; padding: 10px 22px; font-size: 13px;
}
QPushButton#RunBtn:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4fd0a2, stop:1 #6cb8f2); }
QPushButton#RunBtn:pressed { background: #2eae83; }

#SrcChip {
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(255,255,255,0.75);
    border-radius: 12px;
    color: #2a4a41; font-weight: 600; font-size: 12px; padding: 6px 12px;
}

#StatValue { font-size: 26px; font-weight: 800; color: #1e3b33; }
#StatLabel { font-size: 11px; color: #6b857d; font-weight: 600; }

#StatusChip { color: white; font-weight: 800; font-size: 10px;
              padding: 3px 10px; border-radius: 9px; }
#VerdictModel { font-weight: 700; font-size: 13px; color: #25483f; }
#VerdictReason { font-size: 12px; color: #5f7a72; }

#RankGem { color: rgba(255,255,255,235); font-size: 20px; font-weight: 800;
           border-radius: 18px; }
#CardEntity { font-size: 20px; font-weight: 800; color: #1c3a31; }
#CardMeta   { font-size: 11px; color: #6b857d; }
#CardScore  { font-size: 26px; font-weight: 800; }
#ModelChip { background: rgba(63,191,143,40); border: 1px solid rgba(63,191,143,90);
             border-radius: 10px; color: #26785b; font-size: 10px; font-weight: 700;
             padding: 3px 9px; }
#Explain   { font-size: 12px; color: #3d5a51; }
"""


class PipelineRunner(QThread):
    """Run the anomaly pipeline off the UI thread."""
    done = Signal(object)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self.paths = paths

    def run(self):
        from aml.pipeline import Pipeline
        self.done.emit(Pipeline().run(self.paths, with_models=True))


class FrostBlob(QLabel):
    """A soft radial colour blob that fakes out-of-focus background depth."""

    def __init__(self, color, size, alpha, parent=None):
        super().__init__(parent)
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        grad = QRadialGradient(size / 2, size / 2, size / 2)
        grad.setColorAt(0.0, QColor(color))
        grad.setColorAt(1.0, QColor(color))
        col = QColor(color)
        col.setAlpha(alpha)
        grad.setColorAt(0.0, col)
        col2 = QColor(color)
        col2.setAlpha(0)
        grad.setColorAt(1.0, col2)
        p.setBrush(grad)
        p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, size, size)
        p.end()
        self.setPixmap(pix)
        self.setAttribute(Qt.WA_TranslucentBackground)


def make_blur_effect(widget, radius=40):
    eff = QGraphicsBlurEffect(widget)
    eff.setBlurRadius(radius)
    widget.setGraphicsEffect(eff)


def make_shadow(widget, blur=26, dy=8, alpha=50):
    sh = QGraphicsDropShadowEffect(widget)
    sh.setBlurRadius(blur)
    sh.setOffset(0, dy)
    sh.setColor(QColor(60, 120, 105, alpha))
    widget.setGraphicsEffect(sh)


class Glass(QFrame):
    """A frosted glass panel: translucent white with a white hairline edge."""

    def __init__(self, strong=False, radius=22, parent=None):
        super().__init__(parent)
        self.setObjectName("GlassPanelStrong" if strong else "GlassPanel")
        self._radius = radius
        self.setStyleSheet(self.styleSheet())

    def setContent(self, widget):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 20)
        lay.setSpacing(14)
        lay.addWidget(widget)


def section_title(text, sub=""):
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    t = QLabel(text)
    t.setObjectName("SectionTitle")
    lay.addWidget(t)
    if sub:
        s = QLabel(sub)
        s.setObjectName("SectionSub")
        lay.addWidget(s)
    return box


def make_source_chips(per_source):
    row = QWidget()
    from PySide6.QtWidgets import QHBoxLayout
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    for rs in per_source:
        chip = QLabel(f"{rs.source.upper()}  {rs.n_rows:,}")
        chip.setObjectName("SrcChip")
        lay.addWidget(chip)
    return row


def make_stat(label, value):
    box = QFrame()
    box.setObjectName("GlassPanel")
    lay = QVBoxLayout(box)
    lay.setContentsMargins(16, 12, 16, 12)
    v = QLabel(str(value))
    v.setObjectName("StatValue")
    l = QLabel(label)
    l.setObjectName("StatLabel")
    lay.addWidget(v)
    lay.addWidget(l)
    make_shadow(box, 18, 4, 40)
    return box


def make_verdict_row(model, status, reason):
    row = QWidget()
    from PySide6.QtWidgets import QHBoxLayout
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 2, 0, 2)
    lay.setSpacing(12)
    m = QLabel(model)
    m.setObjectName("VerdictModel")
    m.setMinimumWidth(190)
    lay.addWidget(m)
    chip = QLabel(status)
    chip.setObjectName("StatusChip")
    chip.setStyleSheet(f"#StatusChip {{ background: {STATUS_COLORS.get(status, MUTED)}; }}")
    lay.addWidget(chip)
    r = QLabel(reason)
    r.setObjectName("VerdictReason")
    r.setWordWrap(True)
    lay.addWidget(r, 1)
    return row


def make_insight_card(rank, item):
    """A crystallised ranked-insight card."""
    from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
    card = QFrame()
    card.setObjectName("GlassPanelStrong")
    make_shadow(card, 22, 6, 46)
    lay = QHBoxLayout(card)
    lay.setContentsMargins(18, 16, 18, 16)
    lay.setSpacing(18)

    # rank gem (crystal facet)
    gem = QLabel(f"{rank}")
    gem.setObjectName("RankGem")
    gem.setFixedSize(38, 38)
    gem.setAlignment(Qt.AlignCenter)
    gem.setStyleSheet(
        f"#RankGem {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
        f"stop:0 {MINT}, stop:1 {SKY}); }}"
    )
    lay.addWidget(gem)

    # middle: entity + meta + explanation
    mid = QWidget()
    v = QVBoxLayout(mid)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)
    e = QLabel(item.get("entity_id", ""))
    e.setObjectName("CardEntity")
    v.addWidget(e)
    meta = QLabel(f"{item.get('n_events', 0):,} events   models: "
                  + ", ".join(item.get("models_fired", [])))
    meta.setObjectName("CardMeta")
    v.addWidget(meta)
    expl = QLabel(item.get("explanation", ""))
    expl.setObjectName("Explain")
    expl.setWordWrap(True)
    v.addWidget(expl)
    lay.addWidget(mid, 1)

    # right: score
    score = QLabel(f"{item.get('score', 0):.2f}")
    score.setObjectName("CardScore")
    score.setStyleSheet(f"#CardScore {{ color: {TEAL}; }}")
    score.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lay.addWidget(score)
    return card


def build_dashboard(result, window=None):
    """Lay out every section from a PipelineResult."""
    from PySide6.QtWidgets import QHBoxLayout, QGridLayout

    body = QVBoxLayout()
    body.setSpacing(18)

    # ---- header ----------------------------------------------------------
    head = QHBoxLayout()
    brand = QVBoxLayout()
    t = QLabel("Anomaly Lens")
    t.setObjectName("Brand")
    tag = QLabel("multi-source anomaly detection · bank + CDR + social")
    tag.setObjectName("Tag")
    brand.addWidget(t)
    brand.addWidget(tag)
    head.addLayout(brand, 1)
    head.addWidget(make_source_chips(result.per_source))
    btn = QPushButton("Run analysis")
    btn.setObjectName("RunBtn")
    head.addWidget(btn)
    body.addLayout(head)

    # ---- stats -------------------------------------------------------------
    unified = result.unified
    n_entities = unified["entity_id"].nunique() if not unified.empty else 0
    n_cross = getattr(result.entity_map, "n_cross_source", 0)
    blocked = sum(1 for v in result.sufficiency.values() if v.status == "BLOCKED")
    stats = QHBoxLayout()
    stats.setSpacing(14)
    stats.addWidget(make_stat("events", len(unified)))
    stats.addWidget(make_stat("entities", n_entities))
    stats.addWidget(make_stat("sources", len(result.per_source)))
    stats.addWidget(make_stat("cross-source links", n_cross))
    stats.addWidget(make_stat("blocked models", blocked))
    stats.addStretch(1)
    body.addLayout(stats)

    # ---- sufficiency ---------------------------------------------------------
    suff = Glass(strong=True)
    sv = suff.layout() if suff.layout() else QVBoxLayout(suff)
    sv.setContentsMargins(22, 18, 22, 18)
    sv.setSpacing(8)
    sv.addWidget(section_title("Data sufficiency",
                               "per-model verdict — the pipeline only runs what the data supports"))
    for v in result.sufficiency.values():
        sv.addWidget(make_verdict_row(v.model, v.status, v.reason))
    body.addWidget(suff)

    # ---- ranked insights ----------------------------------------------------
    ins = QVBoxLayout()
    ins.setSpacing(14)
    ins.addWidget(section_title("Ranked insights",
                                "fused across models — highest risk first"))
    expl = result.explanations
    for i, item in enumerate(result.rankings[:10], 1):
        item = dict(item)
        item["explanation"] = expl.get(item["entity_id"], "")
        ins.addWidget(make_insight_card(i, item))
    body.addLayout(ins)
    body.addStretch(1)

    # wire re-run button
    from PySide6.QtCore import QObject
    btn.clicked.connect(lambda: _rerun(btn, result, window))
    return body


def _sources_dir() -> str:
    from aml.config import PipelineConfig
    from aml.loaders import collect_files
    cfg = PipelineConfig.from_yaml(str(ROOT / "config" / "config.yaml"))
    return collect_files([str(ROOT / (cfg.sources_dir or "data/sources"))])


def _rerun(btn, result, window=None):
    paths = _sources_dir()
    btn.setEnabled(False)
    btn.setText("running…")
    th = PipelineRunner(paths, btn)
    _RERUN_THREADS.append(th)
    th.done.connect(lambda res: _apply_refresh(res, btn, window))
    th.start()


_RERUN_THREADS = []


def _apply_refresh(res, btn, window=None):
    from PySide6.QtWidgets import QMessageBox
    btn.setText("Run analysis")
    btn.setEnabled(True)
    if window is not None and hasattr(window, "populate"):
        window.populate(res)
    else:
        QMessageBox.information(btn, "Anomaly Lens", "Analysis finished.")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anomaly Lens")
        self.resize(1120, 780)
        self.setObjectName("Root")

        # background frost blobs
        b1 = FrostBlob(MINT, 460, 60, self)
        b1.move(-120, -120)
        make_blur_effect(b1)
        b2 = FrostBlob(SKY, 520, 55, self)
        b2.move(760, 420)
        make_blur_effect(b2)
        b3 = FrostBlob("#bfe3d3", 320, 80, self)
        b3.move(900, 0)
        make_blur_effect(b3)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.content.setObjectName("ScrollContent")
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(28, 22, 28, 30)
        self.scroll.setWidget(self.content)
        self.root_lay = QVBoxLayout(self)
        self.root_lay.setContentsMargins(0, 0, 0, 0)
        self.root_lay.addWidget(self.scroll)

    def populate(self, result):
        # clear placeholders and add the dashboard
        while self.body.count():
            it = self.body.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self.body.addLayout(build_dashboard(result, self))
        self.content.setStyleSheet(
            "QWidget#ScrollContent { background: transparent; }"
        )


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)

    # run pipeline
    paths = _sources_dir()
    from aml.pipeline import Pipeline
    result = Pipeline().run(paths, with_models=True)

    win = MainWindow()
    win.resize(1120, 780)
    win.populate(result)
    win.show()

    if args.shot:
        out = ROOT / args.shot
        win.grab().save(str(out))
        print("saved", out)
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())