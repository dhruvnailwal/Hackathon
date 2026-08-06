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
    button = QPushButton("Run analysis")
    result = types.SimpleNamespace(name="new-result")

    _apply_refresh(result, button, window)

    assert window.populate_calls == [result]
    assert button.text() == "Run analysis"
    assert button.isEnabled() is True
