"""
Headless UI and Widget Verification Tests using pytest-qt and PySide6.
"""

import pytest
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from typhoon_transcriber.ui.main_window import MainWindow
from typhoon_transcriber.ui.widgets.vu_meter import VUMeterWidget
from typhoon_transcriber.ui.widgets.download_dialog import ModelDownloadDialog
from typhoon_transcriber.models.onnx_engine import TyphoonONNXEngine
from typhoon_transcriber.models.model_manager import ModelManager


def test_vu_meter_widget(qtbot):
    """Verify VU meter level updates without error."""
    widget = VUMeterWidget()
    qtbot.addWidget(widget)

    widget.set_level(0.5)
    assert widget._level >= 0.35  # Smoothed level

    widget.set_level(1.0)
    assert widget._level == 1.0

    widget.set_level(0.0)
    assert widget._level < 1.0


def test_download_dialog_ui(qtbot):
    """Verify ModelDownloadDialog initialization and model combo options."""
    dialog = ModelDownloadDialog()
    qtbot.addWidget(dialog)

    assert dialog.combo_models.count() > 0
    assert dialog.btn_download.isEnabled()
    assert not dialog.btn_cancel.isEnabled()


def test_main_window_lifecycle_and_controls(qtbot):
    """Verify MainWindow layout, buttons, theme switching, and actions."""
    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    # Verify initial state
    assert window.btn_start.isEnabled()
    assert not window.btn_stop.isEnabled()
    assert window.device_combo.count() >= 0
    assert not window.mock_banner.isHidden()

    # Test text editing & clearing
    window.transcript_edit.setPlainText("สวัสดีครับ ทดสอบข้อความ")
    assert "สวัสดีครับ" in window.transcript_edit.toPlainText()

    # Test copy to clipboard
    window.copy_transcript_to_clipboard()
    clipboard_text = QApplication.clipboard().text()
    assert "สวัสดีครับ" in clipboard_text

    # Test font zoom in/out
    initial_font_size = window.font_size
    window._adjust_font_size(2)
    assert window.font_size == initial_font_size + 2
    window._adjust_font_size(-2)
    assert window.font_size == initial_font_size

    # Test theme toggle
    current_theme = window.is_dark_theme
    window.toggle_theme()
    assert window.is_dark_theme != current_theme

    # Test clear
    window.clear_transcript()
    assert window.transcript_edit.toPlainText() == ""

    # Clean close
    window.close()


def test_main_window_tabs_and_batch_table(qtbot, tmp_path):
    """Verify MainWindow tabs (Live vs Batch), table structure, and file ingestion."""
    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    # 1. Verify two tabs exist
    assert window.tabs.count() == 2
    assert "Live" in window.tabs.tabText(0)
    assert "Batch" in window.tabs.tabText(1)

    # 2. Verify batch table columns
    assert window.batch_table.columnCount() == 6
    assert window.batch_table.rowCount() == 0

    # 3. Create synthetic audio file and process batch
    wav_file = tmp_path / "test_audio.wav"
    wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    window.process_batch_paths([wav_file])

    # 4. Verify batch tab is activated and table populated
    assert window.tabs.currentIndex() == 1
    assert window.batch_table.rowCount() == 1
    assert window.batch_table.item(0, 1).text() == "test_audio.wav"
    assert window.batch_table.item(0, 4).text() == "Pending"

    # 5. Clear batch items
    window.clear_batch_items()
    assert window.batch_table.rowCount() == 0
    assert len(window.batch_items) == 0

    window.close()


def test_main_window_hardware_switching(qtbot):
    """Verify hardware acceleration combobox switches preference and updates UI badge."""
    mock_engine = TyphoonONNXEngine(use_mock=True, user_preference="CPU")
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    assert window.combo_hardware.count() == 2

    # Switch to GPU
    window.combo_hardware.setCurrentIndex(1)
    assert window.engine.user_preference == "GPU"
    assert window.lbl_provider_status.text() == window.engine.get_provider_status_text()

    # Switch back to CPU
    window.combo_hardware.setCurrentIndex(0)
    assert window.engine.user_preference == "CPU"
    assert window.lbl_provider_status.text() == "Running on CPU"

    window.close()


def test_main_window_batch_execution_flow(qtbot, tmp_path, monkeypatch):
    """Verify batch execution from the batch tab updates table rows and finishes cleanly."""
    import soundfile as sf
    from typhoon_transcriber.config import SAMPLE_RATE

    # Mock QMessageBox to prevent modal dialog popup in headless test
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    # Create dummy audio
    wav_file = tmp_path / "speech_test.wav"
    audio_data = np.zeros(SAMPLE_RATE, dtype=np.float32)
    sf.write(str(wav_file), audio_data, SAMPLE_RATE)

    # Ingest file
    window.process_batch_paths([wav_file])
    assert window.batch_table.rowCount() == 1

    # Set destination
    out_dir = tmp_path / "TranscriptsOut"
    window.txt_batch_dest.setText(str(out_dir))

    # Start batch
    window.start_batch_transcription()
    assert window.btn_batch_cancel.isEnabled()

    # Wait for completion
    qtbot.waitUntil(lambda: window.btn_batch_start.isEnabled(), timeout=5000)

    # Verify table row status updated to Completed
    assert window.batch_table.item(0, 4).text() == "Completed"
    assert (out_dir / "speech_test.txt").exists()

    window.close()


def test_live_and_batch_pause_resume(qtbot):
    """Verify pause and resume button behaviors in Live and Batch modes."""
    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    # 1. Verify default theme is Light
    assert not window.is_dark_theme

    # 2. Verify Live Pause initial state
    assert not window.btn_pause.isEnabled()
    assert window.btn_pause.text() == "⏸ Pause"

    # Start live transcription
    window.start_live_transcription()
    qtbot.waitUntil(lambda: window.streaming_worker is not None and window.streaming_worker.isRunning(), timeout=2000)
    assert window.btn_pause.isEnabled()
    assert window.btn_pause.text() == "⏸ Pause"
    assert not window.streaming_worker.is_paused()

    # Toggle Pause
    window.toggle_pause_live_transcription()
    assert window.streaming_worker.is_paused()
    assert window.btn_pause.text() == "▶ Resume"

    # Toggle Resume
    window.toggle_pause_live_transcription()
    assert not window.streaming_worker.is_paused()
    assert window.btn_pause.text() == "⏸ Pause"

    # Stop live transcription
    window.stop_live_transcription()
    assert not window.btn_pause.isEnabled()
    assert window.btn_pause.text() == "⏸ Pause"

    # 3. Verify Batch Pause initial state
    assert not window.btn_batch_pause.isEnabled()
    assert window.btn_batch_pause.text() == "⏸ Pause Batch"

    window.close()


def test_batch_config_dialog_prompt(qtbot, tmp_path, monkeypatch):
    """Verify BatchConfigDialog popup prompts for destination and updates batch settings."""
    from PySide6.QtWidgets import QDialog
    from typhoon_transcriber.ui.widgets.batch_dialog import BatchConfigDialog

    custom_dest = tmp_path / "MyCustomDestination"
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    # Mock dialog exec to simulate user configuring destination and clicking 'Add to Queue'
    def mock_exec(dialog_self):
        dialog_self.dest_dir = custom_dest
        dialog_self.selected_formats = [".txt", ".srt", ".vtt"]
        dialog_self.start_immediately = False
        return QDialog.Accepted

    monkeypatch.setattr(BatchConfigDialog, "exec", mock_exec)

    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    wav_file = tmp_path / "meeting.wav"
    wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    # Ingest with prompt_destination=True (as triggered by dropEvent and Add Files)
    window.process_batch_paths([wav_file], prompt_destination=True)

    assert window.batch_table.rowCount() == 1
    assert window.dest_batch_dir == custom_dest
    assert window.txt_batch_dest.text() == str(custom_dest)
    assert window.cb_batch_vtt.isChecked()

    window.close()




