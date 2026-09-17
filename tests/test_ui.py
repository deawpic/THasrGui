"""
Headless UI and Widget Verification Tests using pytest-qt and PySide6.
"""

import pytest
import numpy as np
from PySide6.QtCore import Qt, QItemSelectionModel
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog

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

    # Test font size zoom controls
    initial_font_size = window.font_size
    window.btn_zoom_in.click()
    assert window.font_size == initial_font_size + 2
    assert window.lbl_font_size.text() == f"{initial_font_size + 2} pt"

    window.btn_zoom_out.click()
    assert window.font_size == initial_font_size
    assert window.lbl_font_size.text() == f"{initial_font_size} pt"

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

    # 2. Verify batch table columns (5 columns without Relative Subfolder)
    assert window.batch_table.columnCount() == 5
    assert window.batch_table.rowCount() == 0

    # 3. Create synthetic audio file and process batch
    wav_file = tmp_path / "test_audio.wav"
    wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    window.process_batch_paths([wav_file])

    # 4. Verify batch tab is activated and table populated
    assert window.tabs.currentIndex() == 1
    assert window.batch_table.rowCount() == 1
    assert window.batch_table.item(0, 1).text() == "test_audio.wav"
    assert window.batch_table.item(0, 3).text() == "Pending"

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

    # Verify table row status updated to Completed (column 3)
    assert window.batch_table.item(0, 3).text() == "Completed"
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


def test_batch_table_multi_selection_and_removal(qtbot, tmp_path):
    """Verify QTableWidget ExtendedSelection enables multi-selection and batch removal."""
    from PySide6.QtWidgets import QTableWidget

    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    # 1. Verify selection configuration
    assert window.batch_table.selectionMode() == QTableWidget.ExtendedSelection
    assert window.batch_table.selectionBehavior() == QTableWidget.SelectRows

    # 2. Add 3 files
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    f3 = tmp_path / "c.wav"
    dummy_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    f1.write_bytes(dummy_bytes)
    f2.write_bytes(dummy_bytes)
    f3.write_bytes(dummy_bytes)

    window.process_batch_paths([f1, f2, f3])
    assert window.batch_table.rowCount() == 3

    # 3. Simulate multi-selection: select rows 0 and 2
    window.batch_table.selectionModel().select(
        window.batch_table.model().index(0, 0),
        QItemSelectionModel.Select | QItemSelectionModel.Rows,
    )
    window.batch_table.selectionModel().select(
        window.batch_table.model().index(2, 0),
        QItemSelectionModel.Select | QItemSelectionModel.Rows,
    )
    assert len(window.batch_table.selectionModel().selectedRows()) == 2

    # Remove selected items
    window.remove_selected_batch_items()

    # Verify only row 1 ("b.wav") remains
    assert window.batch_table.rowCount() == 1
    assert window.batch_table.item(0, 1).text() == "b.wav"

    window.close()


def test_batch_table_edit_custom_destination(qtbot, tmp_path, monkeypatch):
    """Verify right-click / editing custom output folder updates table and worker destination."""
    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    f1 = tmp_path / "song.wav"
    f1.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    window.process_batch_paths([f1])

    custom_out = tmp_path / "MyCustomFolder"
    custom_out.mkdir(parents=True, exist_ok=True)

    # Mock QFileDialog.getExistingDirectory to return custom_out
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(custom_out))

    # Trigger output destination edit for row 0
    window._edit_output_folder_for_rows([0])

    # Verify item and cell updated
    assert window.batch_items[0].custom_dest_dir == custom_out
    col4_text = window.batch_table.item(0, 4).text()
    assert "[Custom]" in col4_text
    assert str(custom_out) in col4_text

    window.close()


def test_batch_save_and_load_queue_dialogs(qtbot, tmp_path, monkeypatch):
    """Verify Save Queue and Load Queue buttons, JSON persistence, and status preservation."""
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    f1 = tmp_path / "file1.wav"
    f2 = tmp_path / "file2.wav"
    dummy = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    f1.write_bytes(dummy)
    f2.write_bytes(dummy)

    window.process_batch_paths([f1, f2])
    # Mark file1 as Completed
    window.batch_items[0].status = "Completed"
    window._refresh_batch_table()

    # Save Queue
    queue_json = tmp_path / "my_saved_queue.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(queue_json), "JSON"))
    window.save_queue_dialog()
    assert queue_json.exists()

    # Clear Queue
    window.clear_batch_items()
    assert window.batch_table.rowCount() == 0

    # Load Queue (mock getOpenFileNames)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args, **kwargs: ([str(queue_json)], "JSON"))
    window.load_queue_dialog()

    # Verify 2 items restored with statuses preserved
    assert window.batch_table.rowCount() == 2
    assert window.batch_items[0].status == "Completed"
    assert window.batch_items[1].status == "Pending"
    assert window.batch_table.item(0, 3).text() == "Completed"
    assert window.batch_table.item(1, 3).text() == "Pending"

    window.close()


def test_batch_load_multiple_queues_and_append(qtbot, tmp_path, monkeypatch):
    """Verify loading multiple queue files simultaneously and appending to existing queue."""
    from typhoon_transcriber.transcriber.batch_queue import save_batch_queue, BatchItem

    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)

    # 1. Prepare 2 separate queue JSON files
    q1_file = tmp_path / "queue_part1.json"
    q2_file = tmp_path / "queue_part2.json"

    f1 = tmp_path / "speech1.wav"
    f2 = tmp_path / "speech2.wav"
    f3 = tmp_path / "speech3.wav"
    f4 = tmp_path / "speech4.wav"
    dummy = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    for f in [f1, f2, f3, f4]:
        f.write_bytes(dummy)

    # Queue 1: speech1 (Completed) & speech2 (Pending)
    items1 = [
        BatchItem(source_file=f1, status="Completed"),
        BatchItem(source_file=f2, status="Pending"),
    ]
    save_batch_queue(q1_file, items1)

    # Queue 2: speech3 (Pending) & speech4 (Completed)
    items2 = [
        BatchItem(source_file=f3, status="Pending"),
        BatchItem(source_file=f4, status="Completed"),
    ]
    save_batch_queue(q2_file, items2)

    # 2. Launch MainWindow
    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    # 3. Load Queue 1 initially
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args, **kwargs: ([str(q1_file)], "JSON"))
    window.load_queue_dialog()
    assert window.batch_table.rowCount() == 2

    # 4. Load Queue 2 with Append mode
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args, **kwargs: ([str(q2_file)], "JSON"))
    window.load_queue_dialog(append_mode=True)

    # Verify all 4 items are in the queue appended together
    assert window.batch_table.rowCount() == 4
    filenames = [it.source_file.name for it in window.batch_items]
    assert filenames == ["speech1.wav", "speech2.wav", "speech3.wav", "speech4.wav"]

    # Verify Completed statuses are preserved
    assert window.batch_items[0].status == "Completed"
    assert window.batch_items[1].status == "Pending"
    assert window.batch_items[2].status == "Pending"
    assert window.batch_items[3].status == "Completed"

    # 5. Test loading multiple files simultaneously in one shot (q1 and q2 together)
    window.clear_batch_items()
    assert window.batch_table.rowCount() == 0

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(q1_file), str(q2_file)], "JSON"),
    )
    window.load_queue_dialog()
    assert window.batch_table.rowCount() == 4

    # 6. Test Replace mode
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *args, **kwargs: ([str(q1_file)], "JSON"))
    window.load_queue_dialog(append_mode=False)
    assert window.batch_table.rowCount() == 2

    window.close()


def test_last_session_auto_restore_on_launch(qtbot, tmp_path):
    """Verify application restores previous session and resets in-flight 'Processing' to 'Pending'."""
    from typhoon_transcriber.transcriber.batch_queue import save_last_session_queue, BatchItem

    test_session = tmp_path / "last_session_queue.json"
    f1 = tmp_path / "f1.wav"
    f2 = tmp_path / "f2.wav"
    dummy = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    f1.write_bytes(dummy)
    f2.write_bytes(dummy)

    # Simulate previous session that crashed while transcribing f2
    item1 = BatchItem(source_file=f1, status="Completed")
    item2 = BatchItem(source_file=f2, status="Processing")
    save_last_session_queue([item1, item2], session_file=test_session)

    # Launch MainWindow with this session file
    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine, session_file=test_session)
    qtbot.addWidget(window)
    window.show()

    # Verify items are restored
    assert window.batch_table.rowCount() == 2
    # Completed item stays Completed
    assert window.batch_items[0].status == "Completed"
    assert window.batch_table.item(0, 3).text() == "Completed"
    # Processing item recovered to Pending
    assert window.batch_items[1].status == "Pending"
    assert window.batch_table.item(1, 3).text() == "Pending"

    window.close()


def test_post_processing_dialog_and_header_button(qtbot):
    """Verify PostProcessingDialog contents, theme contrast (dark & light), clipboard copy, and header button."""
    from typhoon_transcriber.ui.widgets.post_processing_dialog import PostProcessingDialog

    # 1. Verify Dialog in Dark Theme
    dialog_dark = PostProcessingDialog(is_dark=True)
    qtbot.addWidget(dialog_dark)
    dialog_dark.show()

    assert dialog_dark.is_dark is True
    assert "#cdd6f4" in dialog_dark.desc_label.styleSheet()
    assert "#181825" in dialog_dark.txt_prompt.styleSheet()

    prompt_text = dialog_dark.txt_prompt.toPlainText()
    assert "ผู้เชี่ยวชาญด้านการตรวจสอบและแก้ไขคำผิดภาษาไทยจากข้อความเสียง (ASR)" in prompt_text
    assert "กฎเหล็กในการทำความสะอาดข้อความ" in prompt_text
    assert "Clean & Correct" in prompt_text

    # Test copy to clipboard
    dialog_dark._copy_prompt_to_clipboard()
    copied = QApplication.clipboard().text()
    assert "ผู้เชี่ยวชาญด้านการตรวจสอบและแก้ไขคำผิดภาษาไทยจากข้อความเสียง (ASR)" in copied
    assert "คัดลอกเรียบร้อย" in dialog_dark.btn_copy.text()
    dialog_dark.close()

    # 2. Verify Dialog in Light Theme (High Contrast & Legible)
    dialog_light = PostProcessingDialog(is_dark=False)
    qtbot.addWidget(dialog_light)
    dialog_light.show()

    assert dialog_light.is_dark is False
    # Verify dark charcoal text on light background (no invisible #cdd6f4)
    assert "#1d1d1f" in dialog_light.desc_label.styleSheet()
    assert "#cdd6f4" not in dialog_light.desc_label.styleSheet()
    # Verify crisp white editor for prompt with dark text
    assert "#ffffff" in dialog_light.txt_prompt.styleSheet()
    assert "#1d1d1f" in dialog_light.txt_prompt.styleSheet()
    # Verify crisp primary blue title and button
    assert "#0071e3" in dialog_light.title_label.styleSheet()
    assert "#0071e3" in dialog_light.btn_copy.styleSheet()
    dialog_light.close()

    # 3. Verify dynamic theme switching via apply_theme
    dialog_switch = PostProcessingDialog(is_dark=True)
    qtbot.addWidget(dialog_switch)
    dialog_switch.apply_theme(is_dark=False)
    assert dialog_switch.is_dark is False
    assert "#1d1d1f" in dialog_switch.desc_label.styleSheet()
    dialog_switch.close()

    # 4. Verify MainWindow header button and parent theme propagation
    mock_engine = TyphoonONNXEngine(use_mock=True)
    window = MainWindow(engine=mock_engine)
    qtbot.addWidget(window)
    window.show()

    assert hasattr(window, "btn_post_processing")
    assert window.btn_post_processing.isEnabled()
    assert "Post-Processing" in window.btn_post_processing.text()

    # Switch window to light theme and verify dialog detects light theme
    window.is_dark_theme = False
    window._apply_theme()
    child_dlg = PostProcessingDialog(parent=window)
    qtbot.addWidget(child_dlg)
    assert child_dlg.is_dark is False
    assert "#1d1d1f" in child_dlg.desc_label.styleSheet()
    child_dlg.close()

    # 5. Verify Font Size Zoom controls
    zoom_dlg = PostProcessingDialog(is_dark=False)
    qtbot.addWidget(zoom_dlg)
    initial_font_size = zoom_dlg.prompt_font_size
    assert initial_font_size == 15
    zoom_dlg.btn_zoom_in.click()
    assert zoom_dlg.prompt_font_size == initial_font_size + 1
    assert "16 pt" in zoom_dlg.lbl_font_size.text()
    assert "16px" in zoom_dlg.txt_prompt.styleSheet()
    zoom_dlg.btn_zoom_out.click()
    assert zoom_dlg.prompt_font_size == initial_font_size
    assert "15 pt" in zoom_dlg.lbl_font_size.text()
    zoom_dlg.close()

    window.close()






