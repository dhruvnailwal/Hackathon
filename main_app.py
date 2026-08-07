"""Anomaly Lens — PySide6 desktop analyst.

A drag-and-drop window for financial / comms source files (bank CSV,
CDR, social, JSON, JSONL, prose logs…). Press **Analyze** and every
available model runs over the uploaded set; the colourful report opens in
a full-window webview and is archived automatically under the user's app
data directory (so history survives restarts *and* uninstalls).

Screens:
  Home    — drop zone + file chips + Analyze
  Report  — QWebEngineView report preview (fits the window) + Download
            MD / PDF + History
  History — archived runs, open / delete

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

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
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

QPushButton#PrimaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
              stop:0 #3fbf8f, stop:1 #5aa7e8);
    color: white; border: none; border-radius: 14px;
    font-weight: 700; padding: 10px 24px; font-size: 13px;
}
QPushButton#PrimaryBtn:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
              stop:0 #4fd0a2, stop:1 #6cb8f2); }
QPushButton#PrimaryBtn:disabled { background: #9fb8ae; }

QPushButton#GhostBtn {
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(255,255,255,0.8); border-radius: 14px;
    color: #2a4a41; font-weight: 700; padding: 10px 20px; font-size: 13px;
}
QPushButton#GhostBtn:hover { background: rgba(255,255,255,0.8); }

#DropZone {
    background: rgba(255, 255, 255, 0.30);
    border: 2px dashed #8fb8c2;
    border-radius: 22px;
}
#DropZone[drag="true"] { background: rgba(63, 191, 143, 0.10);
                         border-color: #3fbf8f; }
#DropHint { font-size: 17px; font-weight: 700; color: #2a4a41; }
#DropSub  { font-size: 12px; color: #6b857d; }

#SrcChip { background: rgba(255,255,255,0.55);
           border: 1px solid rgba(255,255,255,0.75); border-radius: 12px;
           color: #2a4a41; font-weight: 600; font-size: 12px; padding: 6px 12px; }

#StatValue { font-size: 26px; font-weight: 800; color: #1e3b33; }
#StatLabel { font-size: 11px; color: #6b857d; font-weight: 600; }

#WebBar { background: rgba(255,255,255,0.55); border: none; border-radius: 0;
          border-bottom: 1px solid rgba(255,255,255,0.9); }
"""


# ---------------------------------------------------------------------------
# pipeline runner (off the UI thread)
# ---------------------------------------------------------------------------
class PipelineRunner(QThread):
    done = Signal(object)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self.paths = paths

    def run(self):
        from aml.pipeline import Pipeline
        try:
            res = Pipeline().run(self.paths, with_models=True)
        except Exception as exc:  # surface load errors instead of dying silently
            res = exc
        self.done.emit(res)


# ---------------------------------------------------------------------------
# shared chrome
# ---------------------------------------------------------------------------
def make_shadow(widget, blur=26, dy=8, alpha=50):
    sh = QGraphicsDropShadowEffect(widget)
    sh.setBlurRadius(blur)
    sh.setOffset(0, dy)
    sh.setColor(QColor(60, 120, 105, alpha))
    widget.setGraphicsEffect(sh)


def header_row(window) -> QFrame:
    bar = QFrame()
    bar.setObjectName("GlassPanelStrong")
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(22, 14, 22, 14)
    lay.setSpacing(10)
    brand = QLabel("Anomaly Lens")
    brand.setObjectName("Brand")
    lay.addWidget(brand)
    rot = QLabel("multi-source anomaly detection · drag-and-drop analyst")
    rot.setObjectName("Tag")
    lay.addWidget(rot, 1)
    return bar


def make_striped_table(rows):  # helper kept for screenshots / plain views
    return rows


# ---------------------------------------------------------------------------
# drop zone
# ---------------------------------------------------------------------------
class DropZone(QFrame):
    dropped = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setProperty("drag", False)
        self.setAcceptDrops(True)
        self.setMinimumHeight(220)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(6)
        t = QLabel("Drag & drop source files here")
        t.setObjectName("DropHint")
        t.setAlignment(Qt.AlignCenter)
        sub = QLabel("CSV · TSV · JSON · JSONL · TXT · LOG — bank, CDR, social, prose")
        sub.setObjectName("DropSub")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(t)
        lay.addWidget(sub)
        self.tip = sub

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("drag", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("drag", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event):
        from aml.loaders import is_supported
        self.setProperty("drag", False)
        urls = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        ok = [p for p in urls if is_supported(p)]
        if not ok:
            QMessageBox.information(self, "Anomaly Lens",
                                    "None of the dropped files have a supported "
                                    "extension (CSV/TSV/JSON/TXT/LOG).")
            return
        self.dropped.emit(ok)


# ---------------------------------------------------------------------------
# home page
# ---------------------------------------------------------------------------
class HomePage(QWidget):
    """Empty state: no hardcoded demo numbers — files appear only after the user drops them."""
    analyze_clicked = Signal(object)
    history_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.files: list = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 24, 40, 30)
        lay.setSpacing(20)
        lay.addWidget(header_row(self))

        self._files: list = []

        self.drop = DropZone(self)
        self.drop.dropped.connect(self._add_files)
        lay.addWidget(self.drop, 1)

        self.chip_row = QWidget(self)
        self.chip_lay = QHBoxLayout(self.chip_row)
        self.chip_lay.setContentsMargins(0, 0, 0, 0)
        self.chip_lay.setSpacing(8)
        self.chip_lay.addStretch(1)
        lay.addWidget(self.chip_row)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.btn_history = QPushButton("History")
        self.btn_history.setObjectName("GhostBtn")
        self.btn_history.clicked.connect(self.history_requested)
        actions.addWidget(self.btn_history)
        self.btn_browse = QPushButton("Browse files…")
        self.btn_browse.setObjectName("GhostBtn")
        self.btn_browse.clicked.connect(self._browse)
        actions.addWidget(self.btn_browse)
        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.setObjectName("PrimaryBtn")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(lambda: self.analyze_clicked.emit(list(self._files)))
        actions.addWidget(self.btn_analyze)
        actions.addStretch(1)
        lay.addLayout(actions)

        self.status = QLabel("No files loaded yet — drop or browse to begin.")
        self.status.setObjectName("Tag")
        lay.addWidget(self.status)

    def dropzone(self):
        return self

    def _add_files(self, paths):
        from aml.loaders import collect_files
        self._files = collect_files(list(self._files) + list(paths))
        self._render_chips()
        self.btn_analyze.setEnabled(len(self._files) > 0)
        self.status.setText(f"{len(self._files)} file(s) ready to analyze.")

    def _browse(self):
        from aml.loaders import SUPPORTED_EXTENSIONS
        flt = "Data files (" + " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS) + ")"
        paths, _ = QFileDialog.getOpenFileNames(self, "Open source files", "", flt)
        if paths:
            self._add_files(paths)

    def _render_chips(self):
        while self.chip_lay.count():
            it = self.chip_lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self.chip_lay.addStretch(1)
        for p in self._files[:8]:
            chip = QLabel(Path(p).name)
            chip.setObjectName("SrcChip")
            chip.setToolTip(p)
            self.chip_lay.addWidget(chip)
        if len(self._files) > 8:
            more = QLabel(f"+{len(self._files) - 8}")
            more.setObjectName("SrcChip")
            self.chip_lay.addWidget(more)
        self.chip_lay.addStretch(1)


# ---------------------------------------------------------------------------
# history dialog
# ---------------------------------------------------------------------------
class HistoryDialog(QDialog):
    def __init__(self, runs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anomaly Lens — History")
        self.resize(680, 460)
        self.runs = runs
        lay = QVBoxLayout(self)
        title = QLabel("Previous analyses")
        title.setObjectName("SectionTitle")
        lay.addWidget(title)
        self.listw = QListWidget(self)
        for r in runs:
            summ = r.get("summary", {})
            top = r.get("top_insight")
            line = (f"{r.get('created_at', r.get('run_id', ''))}  "
                    f"{summ.get('events', 0):,} events · "
                    f"{summ.get('entities', 0):,} entities · "
                    f"{summ.get('sources', 0)} sources · "
                    f"top: {top or '-'} ({r.get('top_score', '-')})")
            item = QListWidgetItem(line)
            item.setData(Qt.UserRole, r.get("run_id"))
            self.listw.addItem(item)
        lay.addWidget(self.listw, 1)
        btns = QHBoxLayout()
        self.btn_open = QPushButton("Open selected")
        self.btn_delete = QPushButton("Delete")
        self.btn_cancel = QPushButton("Close")
        for b in (self.btn_open, self.btn_delete, self.btn_cancel):
            b.setObjectName("GhostBtn")
        self.btn_open.setObjectName("PrimaryBtn")
        self.btn_open.clicked.connect(self.accept)
        self.btn_delete.clicked.connect(self._delete)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_open)
        btns.addWidget(self.btn_delete)
        btns.addStretch(1)
        btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)

    def selected_run_id(self):
        it = self.listw.currentItem()
        return it.data(Qt.UserRole) if it else None

    def rerun_list(self):
        return self.runs

    def _delete(self):
        rid = self.selected_run_id()
        if rid:
            from aml import storage
            storage.delete_run(rid)
            self.runs = storage.list_runs()
            self.listw.clear()
            for r in self.runs:
                summ = r.get("summary", {})
                line = (f"{r.get('created_at', r.get('run_id', ''))}  "
                        f"{summ.get('events', 0):,} events · "
                        f"{summ.get('entities', 0):,} entities · "
                        f"{summ.get('sources', 0)} sources")
                item = QListWidgetItem(line)
                item.setData(Qt.UserRole, r.get("run_id"))
                self.listw.addItem(item)


# ---------------------------------------------------------------------------
# report viewer (webview fills the window)
# ---------------------------------------------------------------------------
class ReportView(QWidget):
    """Shows the generated report HTML in a webview that tracks the window."""
    home_requested = Signal()
    history_requested = Signal()
    export_requested = Signal(str)  # "md" | "pdf"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        bar = QFrame()
        bar.setObjectName("GlassPanelStrong")
        blay = QHBoxLayout(bar)
        blay.setContentsMargins(18, 10, 18, 10)
        blay.setSpacing(8)
        title = QLabel("Analysis report")
        title.setObjectName("SectionTitle")
        blay.addWidget(title)
        blay.addStretch(1)
        md = QPushButton("Download MD")
        pdf = QPushButton("Download PDF")
        hist = QPushButton("History")
        home = QPushButton("← Home")
        for b in (md, pdf, hist, home):
            b.setObjectName("GhostBtn")
        md.clicked.connect(lambda: self.export_requested.emit("md"))
        pdf.clicked.connect(lambda: self.export_requested.emit("pdf"))
        hist.clicked.connect(self.history_requested)
        home.clicked.connect(self.home_requested)
        blay.addWidget(md)
        blay.addWidget(pdf)
        blay.addWidget(hist)
        blay.addWidget(home)
        lay.addWidget(bar)

        self.web = self._make_webview()
        # 1 = stretch — the webview always fills the available window size
        lay.addWidget(self.web, 1)

    def _make_webview(self):
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView
            web = QWebEngineView(self)
        except ImportError:  # QtWebEngine not bundled — degrade to a label
            web = QLabel("QtWebEngine is not installed; open report.md / report.html manually.")
            web.setObjectName("DropSub")
            return web
        web.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return web

    def show_html(self, html_path: str):
        self._html_path = html_path
        if hasattr(self.web, "load"):
            self.web.load(QUrl.fromLocalFile(html_path))
        else:
            self.web.setText(f"report saved at: {html_path}")

    def current_file(self, kind: str):
        return getattr(self, "_html_path", None)


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anomaly Lens")
        self.resize(1180, 800)
        self.setObjectName("root")

        self.storage = None
        self._current_report: dict | None = None
        self.root_lay = QVBoxLayout(self)
        self.root_lay.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget(self)
        self.home = HomePage(self)
        self.report = ReportView(self)
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.report)
        self.root_lay.addWidget(self.stack, 1)

        self.home.analyze_clicked.connect(self._analyze_files)
        self.home.history_requested.connect(self._open_history)
        self.report.home_requested.connect(lambda: self.stack.setCurrentIndex(0))
        self.report.history_requested.connect(self._open_history)
        self.report.export_requested.connect(self._export)

    # -- actions -----------------------------------------------------------
    def _analyze_files(self, paths):
        self._last_files = list(paths)
        self.home.btn_analyze.setEnabled(False)
        self.home.btn_analyze.setText("Analyzing…")
        th = PipelineRunner(paths, self)
        self._th = th
        th.done.connect(self._rendered)
        th.start()

    def _rendered(self, result):
        self.home.btn_analyze.setEnabled(True)
        self.home.btn_analyze.setText("Analyze")
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Anomaly Lens", f"Analysis failed:\n{result}")
            return
        self._show_result(result)

    def _show_result(self, result):
        from aml.config import PipelineConfig
        from aml import storage
        cfg = PipelineConfig.from_yaml(str(ROOT / "config" / "config.yaml"))
        paths = getattr(self, "_last_files", None) or []
        meta = storage.save_run(result, cfg, paths, label="desktop")
        self._open_run(meta["run_id"])

    @property
    def dropzone(self):
        return self.home

    def populate(self, result):
        """API kept for tests / programmatic re-population."""
        self._show_result(result)

    def _open_run(self, run_id):
        from aml import storage
        meta = storage.load_run(run_id)
        if not meta:
            QMessageBox.warning(self, "Anomaly Lens", f"Run {run_id} not found.")
            return
        html = meta.get("files", {}).get("html")
        if html and Path(html).exists():
            self.report.show_html(html)
        else:
            self.report.web.setText(f"report missing for run {run_id}")
        self.stack.setCurrentWidget(self.report)

    def _open_history(self):
        from aml import storage
        runs = storage.list_runs()
        dlg = HistoryDialog(runs, self)
        if dlg.exec() == QDialog.Accepted:
            rid = dlg.selected_run_id()
            if rid:
                self._open_run(rid)

    def _export(self, kind):
        html = self.report.current_file("html")
        if not html:
            return
        run_dir = Path(html).parent
        src = run_dir / ("report.md" if kind == "md" else "report.pdf")
        if not src.exists():
            QMessageBox.warning(self, "Anomaly Lens", f"{src.name} not available.")
            return
        out, _ = QFileDialog.getSaveFileName(self, f"Save {kind.upper()}",
                                             str(ROOT / f"anomaly_report.{kind}"),
                                             f"{kind.upper()} files (*.{kind})")
        if out:
            from shutil import copyfile
            copyfile(src, out)
            QMessageBox.information(self, "Anomaly Lens", f"Saved to\n{out}")


def _apply_refresh(result, btn, window=None):
    """Compat helper used by the tests: reset the button and (re)populate
    the window with the finished pipeline result."""
    from PySide6.QtWidgets import QMessageBox
    btn.setText("Analyze")
    btn.setEnabled(True)
    if window is not None and hasattr(window, "populate"):
        window.populate(result)
    else:
        QMessageBox.information(btn, "Anomaly Lens", "Analysis finished.")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)

    win = MainWindow()
    win.show()

    if args.shot:
        # offline smoke: analyze the shipped sample set and capture the report view
        from aml.loaders import collect_files
        from aml.config import PipelineConfig
        from aml import storage
        from aml.pipeline import Pipeline
        import time
        cfg = PipelineConfig.from_yaml(str(ROOT / "config" / "config.yaml"))
        paths = collect_files([str(ROOT / (cfg.sources_dir or "data/sources"))])
        res = Pipeline().run(paths, with_models=True)
        meta = storage.save_run(res, cfg, paths, label="ui-screenshot")
        win._open_run(meta["run_id"])
        win.resize(1120, 780)
        win.show()
        # give the webview a moment to paint the report before grabbing
        for _ in range(120):
            app.processEvents()
            time.sleep(0.05)
        out = ROOT / args.shot
        win.grab().save(str(out))
        print("saved", out)
        storage.delete_run(meta["run_id"])
        return 0
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())