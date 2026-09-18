"""
Pytest configuration and global fixtures for Typhoon ASR test suite.
Ensures session cache isolation across tests.
"""

import os

# Ensure headless Qt environment for CI and server runs
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from pathlib import Path
from typhoon_transcriber.config import LAST_SESSION_QUEUE_FILE
from typhoon_transcriber.transcriber.batch_queue import clear_last_session_queue


@pytest.fixture(autouse=True)
def isolate_session_cache(tmp_path, monkeypatch):
    """Ensure each test runs with an isolated clean last session cache."""
    test_session_file = tmp_path / "last_session_queue.json"
    monkeypatch.setattr("typhoon_transcriber.config.LAST_SESSION_QUEUE_FILE", test_session_file)
    monkeypatch.setattr("typhoon_transcriber.transcriber.batch_queue.LAST_SESSION_QUEUE_FILE", test_session_file)
    monkeypatch.setattr("typhoon_transcriber.ui.main_window.LAST_SESSION_QUEUE_FILE", test_session_file)
    clear_last_session_queue(test_session_file)
    yield test_session_file
    clear_last_session_queue(test_session_file)


@pytest.fixture(autouse=True)
def mock_modal_dialogs(monkeypatch):
    """Global headless CI guard: prevent any modal QMessageBox dialogs from blocking pytest."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

