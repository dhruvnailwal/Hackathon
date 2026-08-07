import os
import types

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QPushButton

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
                        lambda *a, **k: str(tmp_path / "userdata" / "AnomalyLens"))
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
                        lambda *a, **k: str(tmp_path / "userdata" / "AnomalyLens"))
    dlg = HistoryDialog([], qt_app.activeModalWidget() or None)
    assert dlg.listw.count() == 0
    dlg.runs = [{"run_id": "r1", "summary": {"events": 10}, "created_at": "t"}]
    dlg.rerun_list()
    assert dlg.selected_run_id() is None  # nothing selected yet
