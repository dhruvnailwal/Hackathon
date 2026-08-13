import os
import types

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from main_app import MainWindow, _apply_refresh


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


class SpyWindow(MainWindow):
    def __init__(self):
        super().__init__()
        self.populate_calls = []

    def populate(self, result):
        self.populate_calls.append(result)


def test_apply_refresh_repopulates_window(qt_app):
    window = SpyWindow()
    button = QPushButton("Analyze")
    result = types.SimpleNamespace(name="new-result")

    _apply_refresh(result, button, window)

    assert window.populate_calls == [result]
    assert button.text() == "Analyze"
    assert button.isEnabled() is True


def test_home_starts_empty_no_hardcoded_demo(qt_app):
    """The app must not ship hardcoded demo results — the analyze button
    stays disabled until the user drops/browses real files."""
    win = MainWindow()
    assert win.home.btn_analyze.isEnabled() is False
    assert win.home._files == []
    assert win.stack.currentWidget() is win.home


def test_home_has_dropzone_and_analyze(qt_app):
    win = MainWindow()
    assert hasattr(win.home, "drop")
    assert callable(win.home._add_files)
    assert win.home.btn_analyze.text() == "Analyze"


def test_analyze_flow_enables_and_persists(qt_app, synthetic_dir, tmp_path, monkeypatch):
    """Dropping supported files enables Analyze; pressing it archives a run
    under the user data dir and switches to the report webview."""
    import platformdirs
    from pathlib import Path
    from aml import storage
    monkeypatch.setattr(platformdirs, "user_data_dir",
                        lambda *a, **k: str(tmp_path / "userdata" / "TraceWeave"))
    root = storage.results_root()
    win = MainWindow()

    paths = [str(p) for p in sorted((synthetic_dir / "sources").glob("*"))]
    win.home._add_files(paths)
    assert win.home.btn_analyze.isEnabled() is True
    assert len(win.home._files) == len(paths)

    win.home.analyze_clicked.emit(list(win.home._files))
    # QThread finishes asynchronously; poll a few frames with processEvents.
    import time
    for _ in range(400):
        app = QApplication.instance()
        app.processEvents()
        if getattr(win, "_th", None) is None or not win._th.isRunning():
            break
        time.sleep(0.02)
    app.processEvents()

    runs = storage.list_runs()
    assert runs, "expected a persisted run"
    assert win.stack.currentWidget() is win.report
    assert getattr(win.report, "_html_path", None) is not None
    stored = storage.run_dir(runs[0]["run_id"])
    assert (stored / "report.html").exists()
    assert (stored / "report.pdf").exists()


def test_history_dialog_lists_runs(qt_app, tmp_path, monkeypatch):
    import platformdirs
    from main_app import HistoryDialog
    monkeypatch.setattr(platformdirs, "user_data_dir",
                        lambda *a, **k: str(tmp_path / "userdata" / "TraceWeave"))
    dlg = HistoryDialog([], qt_app.activeModalWidget() or None)
    assert dlg.listw.count() == 0
    dlg.runs = [{"run_id": "r1", "summary": {"events": 10}, "created_at": "t"}]
    dlg.rerun_list()
    assert dlg.selected_run_id() is None  # nothing selected yet


# ---------------------------------------------------------------------------
# file management: per-file remove + clear all (available post-analysis too)
# ---------------------------------------------------------------------------
def test_files_can_be_removed_individually(qt_app, synthetic_dir):
    win = MainWindow()
    paths = [str(p) for p in sorted((synthetic_dir / "sources").glob("*"))]
    win.home._add_files(paths)
    assert len(win.home._files) == len(paths)

    first = win.home._files[0]
    win.home._remove_file(first)
    assert first not in win.home._files
    assert len(win.home._files) == len(paths) - 1
    assert win.home.btn_analyze.isEnabled()

    # removing everything disables Analyze again
    for p in list(win.home._files):
        win.home._remove_file(p)
    assert win.home._files == []
    assert win.home.btn_analyze.isEnabled() is False
    assert win.home.btn_clear.isEnabled() is False


def test_clear_all_restores_empty_state(qt_app, synthetic_dir):
    win = MainWindow()
    paths = [str(p) for p in sorted((synthetic_dir / "sources").glob("*"))]
    win.home._add_files(paths)
    win.home._clear_files()
    assert win.home._files == []
    assert win.home.btn_analyze.isEnabled() is False
    assert "No files loaded yet" in win.home.status.text()


def test_every_file_row_has_a_remove_button(qt_app, synthetic_dir):
    from main_app import FileRow
    paths = [str(p) for p in sorted((synthetic_dir / "sources").glob("*"))]
    win = MainWindow()
    win.home._add_files(paths)
    rows = []
    for i in range(win.home.list_lay.count()):
        w = win.home.list_lay.itemAt(i).widget()
        if isinstance(w, FileRow):
            rows.append(w)
    assert len(rows) == len(paths)
    assert all(r.btn_remove is not None for r in rows)
    # clicking the row's ✕ removes exactly that file
    target = rows[0].path
    rows[0].btn_remove.click()
    assert target not in win.home._files


def test_toast_is_non_blocking(qt_app):
    win = MainWindow()
    win.toast("hello toast")
    toast = win._toast
    # the toast is an overlay label, not a modal dialog
    assert "hello toast" in toast.text()
    assert not any(isinstance(w, QMessageBox) for w in QApplication.topLevelWidgets())


def test_export_missing_report_uses_toast_not_alert(qt_app, tmp_path):
    win = MainWindow()
    # point the report view at a run folder with no generated report files
    win.report._html_path = str(tmp_path)
    win._export("md")
    assert not any(isinstance(w, QMessageBox) for w in QApplication.topLevelWidgets())
