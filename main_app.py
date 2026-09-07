"""TraceWeave — PySide6 desktop analyst.

A drag-and-drop window for financial / comms source files (bank CSV,
CDR, social, JSON, JSONL, prose logs…). Press **Analyze** and every
available model runs over the uploaded set; the colourful report opens in
a full-window webview and is archived automatically under the user's app
data directory (so history survives restarts *and* uninstalls).

Screens:
  Home    — drop zone + file list (removable per-file, clear all) + Analyze
  Report  — QWebEngineView report preview (fits the window) + Download
            MD / PDF + History
  History — archived runs, open / delete

Run:
    python main_app.py                  # desktop window
    python main_app.py --shot ui.png   # render to PNG and exit
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QPropertyAnimation, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# theme — green & white with crisp hover / pressed feedback on every button
# ---------------------------------------------------------------------------
MINT = "#2fae8f"            # primary
MINT_HOVER = "#3fc9a2"
MINT_PRESSED = "#229a79"
MINT_DARK = "#1e7f63"
SKY = "#5aa7e8"
TEAL = "#35c7c2"
AMBER = "#e0a03d"
RED = "#e0566b"
RED_HOVER = "#fde7ec"
RED_PRESSED = "#f9d8de"
INK = "#16342a"
MUTED = "#6f8d82"
BORDER = "#d9e8e0"
HINT = "#8fa89f"

QSS = """
* { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; outline: none; }

QWidget#Root {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f9fdfb, stop:0.5 #f0f9f4, stop:1 #e6f3ec);
}
QLabel { color: #16342a; background: transparent; }
QToolTip { background: #16342a; color: #fff; border: none; padding: 4px 8px; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: rgba(46,175,143,20); width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: rgba(46,175,143,90); border-radius: 4px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }

#LogoMark {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #4fd0a2, stop:1 #35a885);
    color: white; border-radius: 22px;
    font-size: 21px; font-weight: 900; letter-spacing: 1px;
}
#AppTitle { font-size: 26px; font-weight: 800; letter-spacing: 0.3px; color: #16342a; }
#AppTag   { font-size: 12px; color: #6d8d82; font-weight: 600; }

#GlassPanel {
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid rgba(217, 232, 224, 0.9);
    border-radius: 22px;
}
#GlassPanelStrong {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(47, 174, 143, 0.28);
    border-radius: 22px;
}

#SectionTitle { font-size: 15px; font-weight: 800; color: #173b2e; }
#SectionSub   { font-size: 12px; color: #6d8d82; }

#Badge {
    background: #e6f6ee; color: #1e7f63; border: 1px solid #bfe5d3;
    border-radius: 11px; font-weight: 800; font-size: 11px;
    padding: 3px 11px;
}

/* ---------------- buttons: every state (base / hover / pressed) ---------- */
QPushButton {
    border: none; border-radius: 13px; font-weight: 700; font-size: 13px;
    padding: 9px 20px; color: #16342a; background: transparent;
}
QPushButton:focus { outline: none; }

QPushButton#PrimaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
              stop:0 #41c89b, stop:1 #29a57e);
    color: white; font-weight: 800;
    border: 1px solid rgba(255,255,255,0.25);
}
QPushButton#PrimaryBtn:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
              stop:0 #52dcae, stop:1 #34b78c); }
QPushButton#PrimaryBtn:pressed { background: #229a74; }
QPushButton#PrimaryBtn:disabled { background: #cfe7dc; color: #f4faf7; }

QPushButton#GhostBtn {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid #d7e8df; color: #1e5a44;
}
QPushButton#GhostBtn:hover {
    background: #ffffff; border-color: #2fae8f;
}
QPushButton#GhostBtn:pressed {
    background: #e3f5ec; border-color: #2fae8f;
}
QPushButton#GhostBtn:disabled { color: #a9c2b9; background: rgba(255,255,255,0.5); }

QPushButton#AccentBtn {
    background: #e8f7f0;
    border: 1px solid #2fae8f; color: #1e6b52;
}
QPushButton#AccentBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
              stop:0 #4fd0a2, stop:1 #2fae8f);
    color: white;
}
QPushButton#AccentBtn:pressed {
    background: #229a74; color: white; border-color: #229a74;
}
QPushButton#AccentBtn:disabled {
    color: #9fbaae; background: #eef6f1; border-color: #cfe5d9;
}

QPushButton#DangerBtn {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid #eecdd3; color: #d0455e;
}
QPushButton#DangerBtn:hover { background: #fdeef1; border-color: #e0566b; }
QPushButton#DangerBtn:pressed { background: #fadcdf; border-color: #d0455e; }
QPushButton#DangerBtn:disabled { color: #e8b6bf; background: rgba(255,255,255,0.5);
                                 border-color: #f2e2e5; }

QPushButton#IconBtn {
    background: transparent; color: #9db7ac; padding: 4px 9px;
    border-radius: 9px; font-size: 15px; font-weight: 800;
    border: 1px solid transparent;
}
QPushButton#IconBtn:hover { background: #fde7ec; color: #d0455e; }
QPushButton#IconBtn:pressed { background: #fad5dc; color: #b53b50; }

/* ---------------- drop zone --------------- */
#DropZone {
    background: rgba(255, 255, 255, 0.5);
    border: 2px dashed #bcd9cc;
    border-radius: 24px;
}
#DropZone:hover { border-color: #7ccfae; background: rgba(255,255,255,0.65); }
#DropZone[drag="true"] {
    background: #eefaf4; border-color: #2fae8f;
}
#DropIcon { color: #2fae8f; font-size: 42px; font-weight: 400; }
#DropHint { font-size: 17px; font-weight: 800; color: #1e5a44; }
#DropSub  { font-size: 12px; color: #6d8d82; }

/* ---------------- file rows ---------------- */
#FileRow {
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid rgba(217, 232, 224, 0.9);
    border-radius: 14px;
}
#FileRow:hover {
    background: #ffffff;
    border-color: #9fdcc3;
}
#FileBadge {
    color: white; border-radius: 9px;
    font-size: 10px; font-weight: 800; letter-spacing: 0.4px;
}
#FileName { font-size: 13px; font-weight: 700; color: #173b2e; }
#FileMeta { font-size: 11px; color: #7c9a90; }
#ListEmpty { font-size: 12px; color: #8aa69c; padding: 14px 4px; }

/* ---------------- stats / misc ---------------- */
#StatValue { font-size: 24px; font-weight: 800; color: #1e3b33; }
#StatLabel { font-size: 11px; color: #6d8d82; font-weight: 700; }

#WebBar {
    background: rgba(255,255,255,0.8);
    border: none; border-bottom: 1px solid #e3efe9;
    border-bottom-left-radius: 0; border-bottom-right-radius: 0;
}

QListWidget {
    background: rgba(255,255,255,0.6);
    border: 1px solid #dfede6; border-radius: 14px; padding: 4px;
    font-size: 13px; color: #16342a;
}
QListWidget::item { padding: 8px; border-radius: 9px; }
QListWidget::item:hover { background: #f2faf6; }
QListWidget::item:selected { background: #e2f5eb; color: #16342a; }

/* ---------------- toast ---------------- */
#Toast {
    background: #1e4b3a;
    color: #eef9f3; border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px; font-size: 12.5px; font-weight: 600;
    padding: 11px 18px;
}
"""

ALLOWED_EXTENSIONS_HINT = "CSV · TSV · JSON · JSONL · TXT · LOG — bank, CDR, social, prose"

FORCE_COLORS = {"csv": MINT, "tsv": TEAL, "json": SKY, "jsonl": SKY, "txt": AMBER, "log": AMBER}

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
def make_shadow(widget, blur=24, dy=7, alpha=45):
    sh = QGraphicsDropShadowEffect(widget)
    sh.setBlurRadius(blur)
    sh.setOffset(0, dy)
    sh.setColor(QColor(46, 145, 110, alpha))
    widget.setGraphicsEffect(sh)


class Toast(QLabel):
    """Non-blocking bottom-right notification that fades in and out."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setWordWrap(True)
        self.setMaximumWidth(420)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade_out)
        self._anim = QPropertyAnimation(self, b"windowOpacity", self)

    def _reposition(self):
        p = self.parentWidget()
        if not p:
            return
        hint = self.sizeHint()
        self.resize(min(hint.width(), 420), hint.height())
        self.adjustSize()
        self.move(p.width() - self.width() - 26, p.height() - self.height() - 26)

    def notify(self, text: str, ms: int = 3800):
        self.setText(text)
        self.adjustSize()
        self._reposition()
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self._anim.stop()
        self._anim.setDuration(220)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()
        self._timer.start(ms)

    def _fade_out(self):
        self._anim.stop()
        self._anim.setDuration(420)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        try:
            self._anim.finished.disconnect(self.hide)
        except TypeError:
            pass
        self._anim.finished.connect(self.hide)
        self._anim.start()


def header_row(window) -> QFrame:
    bar = QFrame()
    bar.setObjectName("GlassPanelStrong")
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(22, 14, 22, 14)
    lay.setSpacing(14)

    mark = QLabel("AL")
    mark.setObjectName("LogoMark")
    mark.setFixedSize(44, 44)
    mark.setAlignment(Qt.AlignCenter)
    lay.addWidget(mark)

    brand = QVBoxLayout()
    brand.setSpacing(1)
    t = QLabel("TraceWeave")
    t.setObjectName("AppTitle")
    tag = QLabel("multi-source anomaly detection · drag-and-drop analyst")
    tag.setObjectName("AppTag")
    brand.addWidget(t)
    brand.addWidget(tag)
    lay.addLayout(brand, 1)
    return bar


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
        self.setMinimumHeight(230)
        self._toast_cb = None

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(6)
        icon = QLabel("◎")
        icon.setObjectName("DropIcon")
        icon.setAlignment(Qt.AlignCenter)
        t = QLabel("Drag & drop source files here")
        t.setObjectName("DropHint")
        t.setAlignment(Qt.AlignCenter)
        sub = QLabel(ALLOWED_EXTENSIONS_HINT)
        sub.setObjectName("DropSub")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(icon)
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
        bad = len(urls) - len(ok)
        if bad:
            (self._toast_cb or (lambda _t: None))(
                f"{bad} file(s) skipped — unsupported type. "
                f"Allowed: CSV · TSV · JSON · JSONL · TXT · LOG")
        if ok:
            self.dropped.emit(ok)


# ---------------------------------------------------------------------------
# file row widget (name, path, size, per-file remove)
# ---------------------------------------------------------------------------
class FileRow(QFrame):
    remove_requested = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setObjectName("FileRow")

        ext = Path(path).suffix.lower().lstrip(".") or "file"
        color = FORCE_COLORS.get(ext, MINT)
        label = ext.upper()[:4] or "FILE"

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 10, 8)
        lay.setSpacing(12)

        badge = QLabel(label)
        badge.setObjectName("FileBadge")
        badge.setStyleSheet(f"#FileBadge {{ background: {color}; }}")
        badge.setFixedSize(44, 30)
        badge.setAlignment(Qt.AlignCenter)
        lay.addWidget(badge)

        text = QVBoxLayout()
        text.setSpacing(1)
        name = QLabel(Path(path).name)
        name.setObjectName("FileName")
        stats = _file_stats(path)
        meta = QLabel(stats)
        meta.setObjectName("FileMeta")
        text.addWidget(name)
        text.addWidget(meta)
        lay.addLayout(text, 1)

        self.btn_remove = QPushButton("✕")
        self.btn_remove.setObjectName("IconBtn")
        self.btn_remove.setFixedSize(30, 30)
        self.btn_remove.setToolTip(f"Remove {Path(path).name}")
        self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(path))
        lay.addWidget(self.btn_remove)

    def emit_remove(self):
        self.remove_requested.emit(self.path)


def _file_stats(path: str) -> str:
    p = Path(path)
    size = p.stat().st_size if p.exists() else 0
    return f"{p.parent}  ·  {size / 1024:.0f} KB" if size else str(p)


# ---------------------------------------------------------------------------
# home page
# ---------------------------------------------------------------------------
class HomePage(QWidget):
    """Empty state: no hardcoded demo numbers — files appear only after the
    user drops or browses them, and every file can be removed again."""
    analyze_clicked = Signal(object)
    history_requested = Signal()
    toast_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files: list = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 22, 40, 28)
        lay.setSpacing(16)
        lay.addWidget(header_row(self))

        # ---- drop zone -----------------------------------------------------
        self.drop = DropZone(self)
        self.drop.dropped.connect(self._add_files)
        self.drop._toast_cb = self.toast_requested.emit
        lay.addWidget(self.drop, 1)

        # ---- file list card (per-file remove + clear all) -------------------
        self.files_card = QFrame()
        self.files_card.setObjectName("GlassPanel")
        fv = QVBoxLayout(self.files_card)
        fv.setContentsMargins(18, 14, 18, 16)
        fv.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(10)
        title = QLabel("Files to analyze")
        title.setObjectName("SectionTitle")
        head.addWidget(title)
        self.count_badge = QLabel("0")
        self.count_badge.setObjectName("Badge")
        head.addWidget(self.count_badge)
        head.addStretch(1)
        self.btn_clear = QPushButton("Clear all")
        self.btn_clear.setObjectName("DangerBtn")
        self.btn_clear.setEnabled(False)
        self.btn_clear.clicked.connect(self._clear_files)
        head.addWidget(self.btn_clear)
        fv.addLayout(head)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setMinimumHeight(96)
        self.scroll.setMaximumHeight(232)
        self.list_host = QWidget()
        self.list_lay = QVBoxLayout(self.list_host)
        self.list_lay.setContentsMargins(2, 2, 2, 2)
        self.list_lay.setSpacing(6)
        self.scroll.setWidget(self.list_host)
        fv.addWidget(self.scroll)
        lay.addWidget(self.files_card)

        # ---- legal basis / authorization (logged into every archived run
        # and printed into the exported report — optional, but "not
        # recorded" is shown honestly rather than left implicit) ------------
        legal_row = QHBoxLayout()
        legal_row.setSpacing(10)
        legal_label = QLabel("Legal basis / authorization:")
        legal_label.setObjectName("SectionSub")
        legal_row.addWidget(legal_label)
        self.txt_legal_basis = QLineEdit()
        self.txt_legal_basis.setPlaceholderText(
            "e.g. warrant #, internal case ref, FIU-IND request — optional")
        legal_row.addWidget(self.txt_legal_basis, 2)
        analyst_label = QLabel("Analyst:")
        analyst_label.setObjectName("SectionSub")
        legal_row.addWidget(analyst_label)
        self.txt_analyst = QLineEdit()
        self.txt_analyst.setPlaceholderText("your name")
        legal_row.addWidget(self.txt_analyst, 1)
        lay.addLayout(legal_row)

        # ---- actions --------------------------------------------------------
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
        actions.addStretch(1)
        self.btn_analyze = QPushButton("Analyze")
        self.btn_analyze.setObjectName("PrimaryBtn")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(lambda: self.analyze_clicked.emit(list(self._files)))
        actions.addWidget(self.btn_analyze)
        lay.addLayout(actions)

        self.status = QLabel("No files loaded yet — drop or browse to begin.")
        self.status.setObjectName("SectionSub")
        lay.addWidget(self.status)

    # -- public API kept for the tests --------------------------------------
    def dropzone(self):
        return self

    # -- file management -----------------------------------------------------
    def _add_files(self, paths):
        from aml.loaders import collect_files
        self._files = collect_files(list(self._files) + list(paths))
        self._render_files()
        self.status.setText(f"{len(self._files)} file(s) ready to analyze — "
                            "press Analyze when you are.")

    def _remove_file(self, path: str):
        self._files = [p for p in self._files if p != path]
        self._render_files()
        if self._files:
            self.status.setText(f"{len(self._files)} file(s) ready to analyze.")
        else:
            self.status.setText("All files removed — drop or browse to begin.")
        self.toast_requested.emit(f"Removed {Path(path).name}")

    def _clear_files(self):
        n = len(self._files)
        self._files = []
        self._render_files()
        self.status.setText("No files loaded yet — drop or browse to start.")
        self.toast_requested.emit(f"Cleared {n} file(s)")

    def _browse(self):
        from aml.loaders import SUPPORTED_EXTENSIONS
        flt = "Data files (" + " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS) + ")"
        paths, _ = QFileDialog.getOpenFileNames(self, "Open source files", "", flt)
        if paths:
            self._add_files(paths)

    def _render_files(self):
        while self.list_lay.count():
            it = self.list_lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        if not self._files:
            empty = QLabel("Nothing here yet — files you add will be listed "
                           "here and can be removed at any time.")
            empty.setObjectName("ListEmpty")
            self.list_lay.addWidget(empty)
        else:
            for p in self._files:
                row = FileRow(p)
                row.remove_requested.connect(self._remove_file)
                self.list_lay.addWidget(row)
            self.list_lay.addStretch(1)
        self.count_badge.setText(str(len(self._files)))
        self.btn_analyze.setEnabled(len(self._files) > 0)
        self.btn_clear.setEnabled(len(self._files) > 0)


# ---------------------------------------------------------------------------
# history dialog
# ---------------------------------------------------------------------------
class HistoryDialog(QDialog):
    def __init__(self, runs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TraceWeave — History")
        self.resize(720, 480)
        self.runs = runs
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 22, 22, 22)
        lay.setSpacing(14)
        title = QLabel("Previous analyses")
        title.setObjectName("SectionTitle")
        lay.addWidget(title)
        self.listw = QListWidget(self)
        self._fill(self.runs)
        lay.addWidget(self.listw, 1)
        self.verify_status = QLabel("")
        self.verify_status.setObjectName("SectionSub")
        self.verify_status.setWordWrap(True)
        lay.addWidget(self.verify_status)

        btns = QHBoxLayout()
        btns.setSpacing(10)
        self.btn_open = QPushButton("Open selected")
        self.btn_verify = QPushButton("Verify integrity")
        self.btn_delete = QPushButton("Delete")
        self.btn_cancel = QPushButton("Close")
        for b in (self.btn_verify, self.btn_delete, self.btn_cancel):
            b.setObjectName("GhostBtn")
        self.btn_open.setObjectName("PrimaryBtn")
        self.btn_delete.setObjectName("DangerBtn")
        self.btn_open.clicked.connect(self.accept)
        self.btn_verify.clicked.connect(self._verify)
        self.btn_delete.clicked.connect(self._delete)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_open)
        btns.addWidget(self.btn_verify)
        btns.addWidget(self.btn_delete)
        btns.addStretch(1)
        btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)

    def _fill(self, runs):
        self.listw.clear()
        for r in runs:
            summ = r.get("summary", {})
            top = r.get("top_insight")
            line = (f"{r.get('created_at', r.get('run_id', ''))}  "
                    f"{summ.get('events', 0):,} records · "
                    f"{summ.get('entities', 0):,} people · "
                    f"{summ.get('sources', 0)} sources · "
                    f"top: {top or '-'} ({r.get('top_score', '-')})")
            item = QListWidgetItem(line)
            item.setData(Qt.UserRole, r.get("run_id"))
            self.listw.addItem(item)

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
            self._fill(self.runs)

    def _verify(self):
        rid = self.selected_run_id()
        if not rid:
            self.verify_status.setText("Select a run first.")
            return
        from aml import storage
        v = storage.verify_run(rid)
        if v["status"] == "OK":
            self.verify_status.setText(
                f"Verified: {len(v['files'])} file(s) match their sealed hashes "
                f"(sealed {v.get('sealed_at', '?')}).")
        elif v["status"] in ("NO_MANIFEST", "MANIFEST_UNREADABLE"):
            self.verify_status.setText(f"Cannot verify: {v['status']}.")
        else:
            bad = [f for f, s in v["files"].items() if s != "ok"]
            self.verify_status.setText(
                f"TAMPERED — {len(bad)} file(s) do not match their sealed hash: "
                + ", ".join(bad[:5]) + ("…" if len(bad) > 5 else ""))


# ---------------------------------------------------------------------------
# report viewer (webview fills the window)
# ---------------------------------------------------------------------------
class ReportView(QWidget):
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
        sub = QLabel("your plain-language findings — everything else stays "
                  "in report.json")
        sub.setObjectName("SectionSub")
        blay.addWidget(sub)
        blay.addStretch(1)
        md = QPushButton("Download MD")
        pdf = QPushButton("Download PDF")
        hist = QPushButton("History")
        home = QPushButton("← Home")
        for b in (md, pdf, hist, home):
            b.setObjectName("AccentBtn")
        md.setObjectName("PrimaryBtn")
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
            web = QLabel("QtWebEngine is not installed; open report.md "
                       "or report.html manually.")
            web.setObjectName("SectionSub")
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
        self.setWindowTitle("TraceWeave")
        self.resize(1200, 820)
        self.setObjectName("Root")

        self.storage = None
        self._current_report: dict | None = None
        self._toast = Toast(self)
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
        self.home.toast_requested.connect(self.toast)
        self.report.home_requested.connect(lambda: self.stack.setCurrentIndex(0))
        self.report.history_requested.connect(self._open_history)
        self.report.export_requested.connect(self._export)

    # -- toast ---------------------------------------------------------------
    def toast(self, text: str, ms: int = 3800):
        self._toast.notify(text, ms)

    # -- actions -------------------------------------------------------------
    def _analyze_files(self, paths):
        self._last_files = list(paths)
        self._last_legal_basis = self.home.txt_legal_basis.text().strip()
        self._last_analyst = self.home.txt_analyst.text().strip()
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
            QMessageBox.critical(self, "TraceWeave",
                                 f"Analysis failed:\n{result}")
            return
        self._show_result(result)

    def _show_result(self, result):
        from aml.config import PipelineConfig
        from aml import storage
        cfg = PipelineConfig.from_yaml(str(ROOT / "config" / "config.yaml"))
        paths = getattr(self, "_last_files", None) or []
        meta = storage.save_run(
            result, cfg, paths, label="desktop",
            legal_basis=getattr(self, "_last_legal_basis", ""),
            analyst=getattr(self, "_last_analyst", ""),
        )
        self._open_run(meta["run_id"])
        self.toast("Analysis complete — your report is ready.")

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
            QMessageBox.warning(self, "TraceWeave", f"Run {run_id} not found.")
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
            self.toast(f"{src.name} is not available for this run.", ms=4500)
            return
        out, _ = QFileDialog.getSaveFileName(self, f"Save {kind.upper()}",
                                             str(ROOT / f"anomaly_report.{kind}"),
                                             f"{kind.upper()} files (*.{kind})")
        if out:
            from shutil import copyfile
            copyfile(src, out)
            # non-blocking toast instead of a modal alert box
            self.toast(f"{kind.upper()} report saved — charts included.",
                       4500)


def _apply_refresh(result, btn, window=None):
    """Compat helper used by the tests: reset the button and (re)populate
    the window with the finished pipeline result."""
    from PySide6.QtWidgets import QMessageBox
    btn.setText("Analyze")
    btn.setEnabled(True)
    if window is not None and hasattr(window, "populate"):
        window.populate(result)
    else:
        QMessageBox.information(btn, "TraceWeave", "Analysis finished.")


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
        win.resize(1180, 800)
        win.show()
        # give the webview a moment to paint the report before grabbing
        for _ in range(120):
            app.processEvents()
            time.sleep(0.05)
        out = ROOT / args.shot
        win.grab().save(str(out))
        print("saved", out)
        # grab the home screen too for a before/after pair
        win.stack.setCurrentIndex(0)
        for _ in range(60):
            app.processEvents()
            time.sleep(0.05)
        shot = Path(args.shot)
        home_shot = ROOT / f"{shot.stem}_home{shot.suffix}"
        win.grab().save(str(home_shot))
        print("saved", home_shot)
        storage.delete_run(meta["run_id"])
        return 0
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())