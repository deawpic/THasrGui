"""
PySide6 Desktop GUI Main Window for Typhoon ASR Desktop Transcriber.
Supports Real-Time Live Streaming, Drag-and-Drop, and Multi-File/Folder Batch Transcription with Mirroring.
Includes Tabbed Navigation (Live vs Batch Table) and Cross-Platform Hardware Acceleration (GPU/DirectML/CUDA/ROCm/OpenVINO vs CPU).
"""

from pathlib import Path
from typing import Optional, List, Union
import logging
import psutil
import os
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QTextEdit,
    QProgressBar,
    QLabel,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QApplication,
    QFrame,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QGroupBox,
    QDialog,
)
from PySide6.QtCore import Qt, Slot, QTimer, QUrl
from PySide6.QtGui import (
    QKeySequence,
    QShortcut,
    QClipboard,
    QFont,
    QDragEnterEvent,
    QDropEvent,
    QColor,
    QDesktopServices,
)

from ..config import APP_NAME, APP_VERSION, DEFAULT_CACHE_DIR, SUPPORTED_AUDIO_EXTENSIONS
from ..models.onnx_engine import TyphoonONNXEngine
from ..models.model_manager import ModelManager
from ..audio.device_manager import AudioDeviceManager, AudioDevice
from ..transcriber.streaming_worker import ASRStreamingWorker
from ..transcriber.batch_queue import BatchItem, scan_files_and_folders, BatchQueueWorker
from ..transcriber.exporter import TranscriptSegment
from .styles import DARK_THEME_QSS, LIGHT_THEME_QSS
from .widgets.vu_meter import VUMeterWidget
from .widgets.export_dialog import ExportDialog
from .widgets.download_dialog import ModelDownloadDialog
from .widgets.batch_dialog import BatchConfigDialog

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main PySide6 Desktop Application Window.
    Features:
    - Tabbed interface separating Live Audio Transcription and Batch File Transcription.
    - Batch Transcription tab with a rich table interface and folder mirroring.
    - Cross-platform hardware acceleration control (CPU vs GPU: DirectML/CUDA/ROCm/OpenVINO)
      with automatic CPU fallback and live provider status display.
    - Drag-and-Drop file and folder ingestion.
    """

    def __init__(self, engine: Optional[TyphoonONNXEngine] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1020, 720)
        self.setMinimumSize(700, 520)
        self.setAcceptDrops(True)  # Enable Drag and Drop

        # Initialize core components
        self.model_manager = ModelManager(cache_dir=DEFAULT_CACHE_DIR)
        self.device_manager = AudioDeviceManager()

        if engine is not None:
            self.engine = engine
        else:
            if self.model_manager.is_model_installed():
                model_p = self.model_manager.get_model_path()
                dec_p = self.model_manager.get_decoder_path()
                tok_p = self.model_manager.get_tokenizer_path()
                self.engine = TyphoonONNXEngine(
                    model_path=model_p,
                    decoder_path=dec_p,
                    tokenizer_path=tok_p,
                    use_mock=False,
                )
            else:
                self.engine = TyphoonONNXEngine(use_mock=True)
                QTimer.singleShot(400, self._auto_check_models)

        self.streaming_worker: Optional[ASRStreamingWorker] = None
        self.batch_worker: Optional[BatchQueueWorker] = None
        self.batch_items: List[BatchItem] = []
        self.dest_batch_dir: Path = Path.home() / "Transcripts"
        self.current_segments: List[TranscriptSegment] = []
        self.is_dark_theme = False
        self.font_size = 16

        self._init_ui()
        self._init_shortcuts()
        self._refresh_audio_devices()
        self._update_mock_banner()
        self._update_hardware_status()
        self._apply_theme()

        # Status timer to display RAM footprint and active engine provider
        self.ram_timer = QTimer(self)
        self.ram_timer.timeout.connect(self._update_ram_status)
        self.ram_timer.start(3000)

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Mock Mode Notification Banner
        self.mock_banner = QFrame(self)
        self.mock_banner.setStyleSheet(
            "background-color: #f9e2af; border-radius: 6px; padding: 6px;"
        )
        banner_layout = QHBoxLayout(self.mock_banner)
        banner_layout.setContentsMargins(8, 4, 8, 4)

        self.lbl_banner = QLabel(
            "⚠️ <b>Mock Mode Active:</b> No local ONNX weights detected. Click 'Model Downloader' to install official Typhoon ASR models.",
            self,
        )
        self.lbl_banner.setStyleSheet("color: #1e1e2e; font-size: 13px;")

        self.btn_banner_dl = QPushButton("📥 Model Downloader", self)
        self.btn_banner_dl.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4; font-weight: bold; padding: 4px 10px;")
        self.btn_banner_dl.clicked.connect(self.open_model_download_dialog)

        banner_layout.addWidget(self.lbl_banner, stretch=1)
        banner_layout.addWidget(self.btn_banner_dl)
        main_layout.addWidget(self.mock_banner)

        # Global Header / Control Bar (Hardware Accelerator & Engine Status)
        header_bar = QHBoxLayout()
        header_bar.setSpacing(10)

        # Hardware Preference Selector (CPU vs GPU)
        lbl_hw = QLabel("⚡ Hardware / การประมวลผล:", self)
        lbl_hw.setStyleSheet("font-weight: bold;")
        self.combo_hardware = QComboBox(self)
        self.combo_hardware.addItem("🖥️ CPU", "CPU")
        self.combo_hardware.addItem("⚡ GPU (Auto-Detect)", "GPU")
        if self.engine.user_preference == "GPU":
            self.combo_hardware.setCurrentIndex(1)
        else:
            self.combo_hardware.setCurrentIndex(0)
        self.combo_hardware.currentIndexChanged.connect(self._on_hardware_changed)

        # Active Provider Status Badge (Requirement 5)
        self.lbl_provider_status = QLabel(self.engine.get_provider_status_text(), self)
        self.lbl_provider_status.setStyleSheet(
            "background-color: #313244; color: #89b4fa; border: 1px solid #45475a; "
            "border-radius: 6px; padding: 4px 10px; font-weight: bold;"
        )
        self.lbl_provider_status.setToolTip("Active ONNX Runtime Execution Provider reported by session.get_providers()")

        # Theme Toggle Button
        self.btn_theme = QPushButton("🌓 Theme", self)
        self.btn_theme.clicked.connect(self.toggle_theme)

        header_bar.addWidget(lbl_hw)
        header_bar.addWidget(self.combo_hardware)
        header_bar.addWidget(self.lbl_provider_status)
        header_bar.addStretch()
        header_bar.addWidget(self.btn_theme)
        main_layout.addLayout(header_bar)

        # Main Tab Widget separating Live and Batch
        self.tabs = QTabWidget(self)
        self._init_live_tab()
        self._init_batch_tab()
        main_layout.addWidget(self.tabs, stretch=1)

        # Status Bar
        self.statusBar().showMessage("Ready (100% Offline Mode) | Drag & Drop audio files/folders anytime")

    def _init_live_tab(self) -> None:
        """Initialize the Live Audio Transcription Tab."""
        self.tab_live = QWidget(self)
        layout = QVBoxLayout(self.tab_live)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Audio Controls Bar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        dev_label = QLabel("🎤 Audio Input:")
        dev_label.setStyleSheet("font-weight: bold;")
        self.device_combo = QComboBox(self)
        self.device_combo.setToolTip("Select Microphone or Desktop Loopback Audio Device")

        self.btn_refresh_dev = QPushButton("🔄", self)
        self.btn_refresh_dev.setToolTip("Refresh Audio Devices")
        self.btn_refresh_dev.setFixedWidth(36)
        self.btn_refresh_dev.clicked.connect(self._refresh_audio_devices)

        self.vu_meter = VUMeterWidget(self)

        self.btn_start = QPushButton("▶ Start Live (F9)", self)
        self.btn_start.setObjectName("start_btn")
        self.btn_start.clicked.connect(self.start_live_transcription)

        self.btn_pause = QPushButton("⏸ Pause", self)
        self.btn_pause.setObjectName("pause_btn")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause_live_transcription)

        self.btn_stop = QPushButton("⏹ Stop Live", self)
        self.btn_stop.setObjectName("stop_btn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_live_transcription)

        top_bar.addWidget(dev_label)
        top_bar.addWidget(self.device_combo, stretch=1)
        top_bar.addWidget(self.btn_refresh_dev)
        top_bar.addWidget(self.vu_meter)
        top_bar.addWidget(self.btn_start)
        top_bar.addWidget(self.btn_pause)
        top_bar.addWidget(self.btn_stop)
        layout.addLayout(top_bar)

        # Live Transcript Text Area
        self.transcript_edit = QTextEdit(self)
        self.transcript_edit.setPlaceholderText(
            "🎙️ Real-time transcription will appear here...\n\n"
            "💡 Click '▶ Start Live (F9)' to transcribe from your microphone or desktop loopback.\n"
            "💡 Or switch to the 'Batch Transcription' tab to transcribe multiple files and mirror folder structures!"
        )
        self.transcript_edit.setReadOnly(False)
        self.transcript_edit.setAcceptDrops(False)
        layout.addWidget(self.transcript_edit, stretch=1)

        # Live Bottom Action Bar
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(8)

        self.cb_autoscroll = QCheckBox("Auto-scroll", self)
        self.cb_autoscroll.setChecked(True)

        self.btn_copy = QPushButton("📋 Copy Text", self)
        self.btn_copy.clicked.connect(self.copy_transcript_to_clipboard)

        self.btn_clear = QPushButton("🗑 Clear", self)
        self.btn_clear.clicked.connect(self.clear_transcript)

        self.btn_zoom_in = QPushButton("A+", self)
        self.btn_zoom_in.setFixedWidth(40)
        self.btn_zoom_in.clicked.connect(lambda: self._adjust_font_size(2))

        self.btn_zoom_out = QPushButton("A-", self)
        self.btn_zoom_out.setFixedWidth(40)
        self.btn_zoom_out.clicked.connect(lambda: self._adjust_font_size(-2))

        self.btn_export = QPushButton("💾 Export Transcript...", self)
        self.btn_export.setStyleSheet("background-color: #89b4fa; color: #11111b; font-weight: bold;")
        self.btn_export.clicked.connect(self.open_export_dialog)

        bottom_bar.addWidget(self.cb_autoscroll)
        bottom_bar.addWidget(self.btn_copy)
        bottom_bar.addWidget(self.btn_clear)
        bottom_bar.addWidget(self.btn_zoom_in)
        bottom_bar.addWidget(self.btn_zoom_out)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.btn_export)
        layout.addLayout(bottom_bar)

        self.tabs.addTab(self.tab_live, "🎙️ Live Transcription (ถอดเสียงสด)")

    def _init_batch_tab(self) -> None:
        """Initialize the Batch File Transcription Tab with interactive table."""
        self.tab_batch = QWidget(self)
        layout = QVBoxLayout(self.tab_batch)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Batch Controls Toolbar
        batch_bar = QHBoxLayout()
        batch_bar.setSpacing(8)

        self.btn_batch_add_files = QPushButton("📁 Add Files...", self)
        self.btn_batch_add_files.setToolTip("Select one or multiple audio/video files for batch transcription")
        self.btn_batch_add_files.clicked.connect(self.add_files_dialog)

        self.btn_batch_add_folder = QPushButton("📂 Add Folder...", self)
        self.btn_batch_add_folder.setToolTip("Select a folder to recursively scan and mirror subfolders")
        self.btn_batch_add_folder.clicked.connect(self.add_folder_dialog)

        # Aliases for backward compatibility
        self.btn_add_files = self.btn_batch_add_files
        self.btn_add_folder = self.btn_batch_add_folder

        self.btn_batch_remove = QPushButton("➖ Remove Selected", self)
        self.btn_batch_remove.clicked.connect(self.remove_selected_batch_items)

        self.btn_batch_clear = QPushButton("🗑 Clear List", self)
        self.btn_batch_clear.clicked.connect(self.clear_batch_items)

        batch_bar.addWidget(self.btn_batch_add_files)
        batch_bar.addWidget(self.btn_batch_add_folder)
        batch_bar.addWidget(self.btn_batch_remove)
        batch_bar.addWidget(self.btn_batch_clear)
        batch_bar.addStretch()
        layout.addLayout(batch_bar)

        # Destination & Formats Settings Group
        settings_group = QGroupBox("📁 Destination & Export Settings / ตั้งค่าการบันทึกไฟล์", self)
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(6)

        # Dest path row
        dest_row = QHBoxLayout()
        dest_lbl = QLabel("Destination:")
        dest_lbl.setStyleSheet("font-weight: bold;")
        self.txt_batch_dest = QLineEdit(str(self.dest_batch_dir), self)
        self.btn_batch_browse = QPushButton("📂 Browse...", self)
        self.btn_batch_browse.clicked.connect(self._browse_batch_dest_folder)

        dest_row.addWidget(dest_lbl)
        dest_row.addWidget(self.txt_batch_dest, stretch=1)
        dest_row.addWidget(self.btn_batch_browse)
        settings_layout.addLayout(dest_row)

        # Format checkboxes row
        fmt_row = QHBoxLayout()
        self.cb_batch_txt = QCheckBox("Plain Text (.txt)", self)
        self.cb_batch_srt = QCheckBox("SubRip Subtitles (.srt)", self)
        self.cb_batch_vtt = QCheckBox("WebVTT Subtitles (.vtt)", self)
        self.cb_batch_json = QCheckBox("JSON with Timecodes (.json)", self)
        self.cb_batch_overwrite = QCheckBox("Overwrite existing", self)

        self.cb_batch_txt.setChecked(True)
        self.cb_batch_srt.setChecked(True)
        self.cb_batch_overwrite.setChecked(True)

        fmt_row.addWidget(self.cb_batch_txt)
        fmt_row.addWidget(self.cb_batch_srt)
        fmt_row.addWidget(self.cb_batch_vtt)
        fmt_row.addWidget(self.cb_batch_json)
        fmt_row.addStretch()
        fmt_row.addWidget(self.cb_batch_overwrite)
        settings_layout.addLayout(fmt_row)

        layout.addWidget(settings_group)

        # Batch Table Widget (Requirement: tab batch ทำเป็น ตาราง)
        self.batch_table = QTableWidget(0, 6, self)
        self.batch_table.setHorizontalHeaderLabels([
            "#", "File Name", "Relative Subfolder", "Size (MB)", "Status", "Mirrored Output / Details"
        ])
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.batch_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.batch_table.setAlternatingRowColors(True)
        layout.addWidget(self.batch_table, stretch=1)

        # Batch Progress Indicators
        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(4)

        self.lbl_batch_overall = QLabel("Overall Progress: 0 / 0 files (0%)", self)
        self.lbl_batch_overall.setStyleSheet("font-weight: bold; color: #89b4fa;")
        self.prog_batch_overall = QProgressBar(self)
        self.prog_batch_overall.setValue(0)

        self.lbl_batch_file = QLabel("Current File: Ready", self)
        self.lbl_batch_file.setStyleSheet("color: #cdd6f4;")
        self.prog_batch_file = QProgressBar(self)
        self.prog_batch_file.setValue(0)

        prog_layout.addWidget(self.lbl_batch_overall)
        prog_layout.addWidget(self.prog_batch_overall)
        prog_layout.addWidget(self.lbl_batch_file)
        prog_layout.addWidget(self.prog_batch_file)
        layout.addLayout(prog_layout)

        # Action Buttons
        act_row = QHBoxLayout()
        self.btn_batch_start = QPushButton("🚀 Start Batch Transcription", self)
        self.btn_batch_start.setStyleSheet("background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 8px 18px;")
        self.btn_batch_start.clicked.connect(self.start_batch_transcription)

        self.btn_batch_pause = QPushButton("⏸ Pause Batch", self)
        self.btn_batch_pause.setEnabled(False)
        self.btn_batch_pause.clicked.connect(self.toggle_pause_batch_transcription)

        self.btn_batch_cancel = QPushButton("⏹ Cancel Batch", self)
        self.btn_batch_cancel.setEnabled(False)
        self.btn_batch_cancel.clicked.connect(self.cancel_batch_transcription)

        self.btn_batch_open_folder = QPushButton("📂 Open Destination Folder", self)
        self.btn_batch_open_folder.setEnabled(False)
        self.btn_batch_open_folder.clicked.connect(self.open_batch_dest_folder)

        act_row.addWidget(self.btn_batch_start)
        act_row.addWidget(self.btn_batch_pause)
        act_row.addWidget(self.btn_batch_cancel)
        act_row.addStretch()
        act_row.addWidget(self.btn_batch_open_folder)
        layout.addLayout(act_row)

        self.tabs.addTab(self.tab_batch, "📁 Batch Transcription (ถอดเสียงเป็นชุด)")

    def _init_shortcuts(self) -> None:
        """Global window shortcuts for rapid control."""
        self.shortcut_toggle = QShortcut(QKeySequence("F9"), self)
        self.shortcut_toggle.activated.connect(self._toggle_live_transcription)

    def _update_mock_banner(self) -> None:
        """Show or hide mock notification banner based on engine status."""
        if getattr(self.engine, "use_mock", False):
            self.mock_banner.setVisible(True)
        else:
            self.mock_banner.setVisible(False)

    def _refresh_audio_devices(self) -> None:
        """Dynamically refresh audio input list."""
        self.device_combo.clear()
        devices = self.device_manager.refresh_devices()

        if not devices:
            self.device_combo.addItem("No audio devices detected", None)
            return

        for dev in devices:
            tag = " [Loopback]" if dev.is_loopback else " [Mic]"
            default_tag = " (Default)" if dev.is_default else ""
            display_name = f"{dev.name}{tag}{default_tag}"
            self.device_combo.addItem(display_name, dev.index)

        default_dev = self.device_manager.get_default_device()
        if default_dev:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == default_dev.index:
                    self.device_combo.setCurrentIndex(i)
                    break

    def _apply_theme(self) -> None:
        """Apply Dark or Light QSS stylesheet."""
        self.setStyleSheet(DARK_THEME_QSS if self.is_dark_theme else LIGHT_THEME_QSS)

    def toggle_theme(self) -> None:
        """Switch between dark and light themes."""
        self.is_dark_theme = not self.is_dark_theme
        self._apply_theme()

    def _adjust_font_size(self, delta: int) -> None:
        """Increase or decrease transcript display font size."""
        self.font_size = max(10, min(self.font_size + delta, 36))
        font = self.transcript_edit.font()
        font.setPointSize(self.font_size)
        self.transcript_edit.setFont(font)

    @Slot(int)
    def _on_hardware_changed(self, index: int) -> None:
        """Handle hardware selection switch (CPU vs GPU)."""
        pref = self.combo_hardware.currentData() or "CPU"
        logger.info("Switching execution provider preference to: %s", pref)
        self.engine.set_user_preference(pref)
        self._update_hardware_status()

    def _update_hardware_status(self) -> None:
        """Update hardware provider badge and status bar with active session provider."""
        status_text = self.engine.get_provider_status_text()
        self.lbl_provider_status.setText(status_text)
        self._update_ram_status()

    def _update_ram_status(self) -> None:
        """Display real-time RAM usage and active provider in status bar."""
        try:
            process = psutil.Process(os.getpid())
            ram_mb = process.memory_info().rss / (1024 * 1024)
            mode_tag = "MOCK Mode" if getattr(self.engine, "use_mock", False) else "ONNX Real-Time"
            provider_tag = self.engine.get_provider_status_text()
            self.statusBar().showMessage(
                f"Ready | Engine: {mode_tag} | {provider_tag} | RAM: {ram_mb:.1f} MB (Budget: <500 MB)"
            )
        except Exception:
            pass

    def _toggle_live_transcription(self) -> None:
        if self.streaming_worker and self.streaming_worker.isRunning():
            self.stop_live_transcription()
        else:
            self.start_live_transcription()

    # =========================================================================
    # DRAG AND DROP HANDLING
    # =========================================================================
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        paths = [Path(url.toLocalFile()) for url in urls if url.isLocalFile()]

        if not paths:
            event.ignore()
            return

        event.acceptProposedAction()
        self.process_batch_paths(paths, prompt_destination=True)

    def process_batch_paths(self, paths: List[Path], prompt_destination: bool = False) -> None:
        """Scan input paths and populate batch table. If prompt_destination is True, displays popup dialog."""
        items = scan_files_and_folders(paths)
        if not items:
            QMessageBox.warning(
                self,
                "No Audio Files Found",
                f"No supported audio or video files found in the specified path(s).\n\n"
                f"Supported extensions: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}",
            )
            return

        start_now = False
        if prompt_destination:
            dialog = BatchConfigDialog(
                items=items,
                engine=self.engine,
                initial_dest_dir=self.dest_batch_dir,
                parent=self,
            )
            if dialog.exec() != QDialog.Accepted:
                return

            self.dest_batch_dir = dialog.dest_dir
            self.txt_batch_dest.setText(str(self.dest_batch_dir))
            self.cb_batch_txt.setChecked(".txt" in dialog.selected_formats)
            self.cb_batch_srt.setChecked(".srt" in dialog.selected_formats)
            self.cb_batch_vtt.setChecked(".vtt" in dialog.selected_formats)
            self.cb_batch_json.setChecked(".json" in dialog.selected_formats)
            self.cb_batch_overwrite.setChecked(dialog.cb_overwrite.isChecked())
            start_now = getattr(dialog, "start_immediately", False)

        # Append new items avoiding duplicates
        existing_paths = {it.source_file.resolve() for it in self.batch_items}
        added_count = 0
        for it in items:
            if it.source_file.resolve() not in existing_paths:
                self.batch_items.append(it)
                existing_paths.add(it.source_file.resolve())
                added_count += 1

        self._refresh_batch_table()
        self.tabs.setCurrentIndex(1)  # Switch to Batch tab
        self.statusBar().showMessage(f"Added {added_count} audio file(s) to Batch queue.")

        if start_now:
            self.start_batch_transcription()

    # =========================================================================
    # BATCH OPERATIONS (TABLE & QUEUE)
    # =========================================================================
    @Slot()
    def add_files_dialog(self) -> None:
        """Select multiple audio/video files via file dialog."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio & Video Files for Batch Conversion",
            "",
            "Audio & Video Files (*.wav *.flac *.mp3 *.m4a *.mp4 *.aac *.ogg *.wma *.mkv *.webm *.opus);;All Files (*)",
        )
        if files:
            self.process_batch_paths([Path(f) for f in files], prompt_destination=True)

    @Slot()
    def add_folder_dialog(self) -> None:
        """Select a folder to recursively scan and mirror for batch conversion."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Root Folder to Recursively Transcribe & Mirror Subfolders",
            "",
        )
        if folder:
            self.process_batch_paths([Path(folder)], prompt_destination=True)

    def _browse_batch_dest_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder for Mirrored Transcripts",
            str(self.dest_batch_dir),
        )
        if folder:
            self.dest_batch_dir = Path(folder)
            self.txt_batch_dest.setText(str(self.dest_batch_dir))

    def _refresh_batch_table(self) -> None:
        """Update batch table rows to reflect self.batch_items."""
        self.batch_table.setRowCount(len(self.batch_items))
        for row, item in enumerate(self.batch_items):
            self.batch_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.batch_table.setItem(row, 1, QTableWidgetItem(item.source_file.name))

            rel_str = str(item.rel_path.parent) if item.rel_path.parent != Path(".") else ""
            self.batch_table.setItem(row, 2, QTableWidgetItem(rel_str))

            size_mb = item.source_file.stat().st_size / (1024 * 1024) if item.source_file.exists() else 0.0
            self.batch_table.setItem(row, 3, QTableWidgetItem(f"{size_mb:.2f} MB"))

            status_item = QTableWidgetItem(item.status)
            self._apply_status_color(status_item, item.status)
            self.batch_table.setItem(row, 4, status_item)

            detail_str = item.error_message or (
                str(self.dest_batch_dir / item.rel_path.parent / item.source_file.stem)
                if item.status == "Completed" else ""
            )
            self.batch_table.setItem(row, 5, QTableWidgetItem(detail_str))

    @staticmethod
    def _apply_status_color(table_item: QTableWidgetItem, status: str) -> None:
        if status == "Completed":
            table_item.setForeground(QColor("#a6e3a1"))  # Green
        elif status == "Processing":
            table_item.setForeground(QColor("#f9e2af"))  # Yellow
        elif status == "Failed":
            table_item.setForeground(QColor("#f38ba8"))  # Red
        else:
            table_item.setForeground(QColor("#a6adc8"))  # Gray / Pending

    def remove_selected_batch_items(self) -> None:
        """Remove selected rows from batch queue."""
        selected_rows = sorted(set(index.row() for index in self.batch_table.selectedIndexes()), reverse=True)
        if not selected_rows:
            return
        for r in selected_rows:
            if 0 <= r < len(self.batch_items):
                del self.batch_items[r]
        self._refresh_batch_table()
        self.statusBar().showMessage(f"Removed {len(selected_rows)} item(s) from batch queue.")

    def clear_batch_items(self) -> None:
        """Clear all items from batch queue."""
        if self.batch_worker and self.batch_worker.isRunning():
            QMessageBox.warning(self, "Batch Running", "Cannot clear queue while batch transcription is running.")
            return
        self.batch_items.clear()
        self._refresh_batch_table()
        self.prog_batch_overall.setValue(0)
        self.prog_batch_file.setValue(0)
        self.lbl_batch_overall.setText("Overall Progress: 0 / 0 files (0%)")
        self.lbl_batch_file.setText("Current File: Ready")
        self.statusBar().showMessage("Batch list cleared.")

    @Slot()
    def start_batch_transcription(self) -> None:
        """Start batch processing of all items in batch table."""
        if not self.batch_items:
            QMessageBox.information(
                self,
                "Batch Queue Empty",
                "No audio files in the batch queue.\nClick 'Add Files...' or 'Add Folder...', or Drag & Drop files.",
            )
            return

        dest_str = self.txt_batch_dest.text().strip()
        if not dest_str:
            self.dest_batch_dir = Path.home() / "Transcripts"
            self.txt_batch_dest.setText(str(self.dest_batch_dir))
        else:
            self.dest_batch_dir = Path(dest_str)

        self.dest_batch_dir.mkdir(parents=True, exist_ok=True)

        selected_formats = []
        if self.cb_batch_txt.isChecked():
            selected_formats.append(".txt")
        if self.cb_batch_srt.isChecked():
            selected_formats.append(".srt")
        if self.cb_batch_vtt.isChecked():
            selected_formats.append(".vtt")
        if self.cb_batch_json.isChecked():
            selected_formats.append(".json")

        if not selected_formats:
            QMessageBox.warning(self, "Format Selection", "Please select at least one output format (.txt, .srt, .vtt, or .json).")
            return

        # Reset item statuses
        for item in self.batch_items:
            item.status = "Pending"
            item.error_message = None
        self._refresh_batch_table()

        self.batch_worker = BatchQueueWorker(
            items=self.batch_items,
            engine=self.engine,
            dest_root_dir=self.dest_batch_dir,
            formats=selected_formats,
            overwrite=self.cb_batch_overwrite.isChecked(),
            parent=self,
        )

        self.batch_worker.overall_progress.connect(self._on_batch_overall_progress)
        self.batch_worker.file_progress.connect(self._on_batch_file_progress)
        self.batch_worker.item_status_changed.connect(self._on_batch_item_status_changed)
        self.batch_worker.batch_completed.connect(self._on_batch_completed)
        self.batch_worker.error_occurred.connect(self._on_batch_error)

        self.btn_batch_start.setEnabled(False)
        self.btn_batch_pause.setEnabled(True)
        self.btn_batch_pause.setText("⏸ Pause Batch")
        self.btn_batch_cancel.setEnabled(True)
        self.btn_batch_add_files.setEnabled(False)
        self.btn_batch_add_folder.setEnabled(False)
        self.btn_batch_remove.setEnabled(False)
        self.btn_batch_clear.setEnabled(False)

        self.batch_worker.start()
        self.statusBar().showMessage(f"Processing batch: {len(self.batch_items)} files...")

    @Slot()
    def toggle_pause_batch_transcription(self) -> None:
        """Toggle pause/resume state for batch file transcription."""
        if not self.batch_worker:
            return
        if self.batch_worker.is_paused():
            self.batch_worker.resume()
            self.btn_batch_pause.setText("⏸ Pause Batch")
            self.statusBar().showMessage("Batch transcription resumed.")
        else:
            self.batch_worker.pause()
            self.btn_batch_pause.setText("▶ Resume Batch")
            self.statusBar().showMessage("Batch transcription paused.")

    @Slot()
    def cancel_batch_transcription(self) -> None:
        """Cancel active batch transcription."""
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self.batch_worker = None

        self.btn_batch_start.setEnabled(True)
        self.btn_batch_pause.setEnabled(False)
        self.btn_batch_pause.setText("⏸ Pause Batch")
        self.btn_batch_cancel.setEnabled(False)
        self.btn_batch_add_files.setEnabled(True)
        self.btn_batch_add_folder.setEnabled(True)
        self.btn_batch_remove.setEnabled(True)
        self.btn_batch_clear.setEnabled(True)
        self.statusBar().showMessage("Batch transcription cancelled by user.")

    @Slot(int, int, str)
    def _on_batch_overall_progress(self, current_idx: int, total_count: int, total_eta_str: str) -> None:
        pct = int((current_idx / total_count) * 100) if total_count > 0 else 0
        self.prog_batch_overall.setValue(pct)
        self.lbl_batch_overall.setText(
            f"Overall Progress: {current_idx} / {total_count} files ({pct}%) — {total_eta_str}"
        )

    @Slot(str, int, str)
    def _on_batch_file_progress(self, filename: str, pct: int, file_eta_str: str) -> None:
        self.prog_batch_file.setValue(pct)
        self.lbl_batch_file.setText(f"Current File: {filename} ({pct}%) — {file_eta_str}")

    @Slot(int, str, str)
    def _on_batch_item_status_changed(self, idx: int, status: str, detail_info: str) -> None:
        if 0 <= idx < len(self.batch_items):
            self.batch_items[idx].status = status
            item_status = self.batch_table.item(idx, 4)
            if item_status:
                item_status.setText(status)
                self._apply_status_color(item_status, status)

            item_detail = self.batch_table.item(idx, 5)
            if item_detail and detail_info:
                item_detail.setText(detail_info)

    @Slot(int, int, float)
    def _on_batch_completed(self, success_count: int, fail_count: int, elapsed_sec: float) -> None:
        self.prog_batch_overall.setValue(100)
        self.prog_batch_file.setValue(100)
        self.lbl_batch_overall.setText(
            f"✅ Batch Completed: {success_count} Succeeded, {fail_count} Failed (Time: {elapsed_sec:.1f}s)"
        )
        self.lbl_batch_file.setText("All files processed.")

        self.btn_batch_start.setEnabled(True)
        self.btn_batch_pause.setEnabled(False)
        self.btn_batch_pause.setText("⏸ Pause Batch")
        self.btn_batch_cancel.setEnabled(False)
        self.btn_batch_add_files.setEnabled(True)
        self.btn_batch_add_folder.setEnabled(True)
        self.btn_batch_remove.setEnabled(True)
        self.btn_batch_clear.setEnabled(True)
        self.btn_batch_open_folder.setEnabled(True)

        self.statusBar().showMessage(f"Batch completed: {success_count} succeeded, {fail_count} failed ({elapsed_sec:.1f}s)")

        QMessageBox.information(
            self,
            "Batch Conversion Complete",
            f"Batch transcription completed successfully!\n\n"
            f"• Succeeded: {success_count} files\n"
            f"• Failed: {fail_count} files\n"
            f"• Output Directory: {self.dest_batch_dir}\n"
            f"• Total Elapsed: {elapsed_sec:.1f}s",
        )

    @Slot(str)
    def _on_batch_error(self, err_msg: str) -> None:
        self.statusBar().showMessage(f"Batch error: {err_msg}")
        QMessageBox.warning(self, "Batch Error", f"An error occurred during batch processing:\n{err_msg}")

    @Slot()
    def open_batch_dest_folder(self) -> None:
        """Open destination folder in native file manager."""
        if self.dest_batch_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.dest_batch_dir)))

    # =========================================================================
    # LIVE STREAMING
    # =========================================================================
    @Slot()
    def open_model_download_dialog(self, auto_start: bool = False) -> None:
        """Open Model Download & Installation Manager."""
        dialog = ModelDownloadDialog(model_manager=self.model_manager, auto_start=auto_start, parent=self)
        dialog.model_installed.connect(self._on_model_installed)
        dialog.exec()

    def _auto_check_models(self) -> None:
        """Automatically check model weights on startup and prompt download if missing."""
        if self.model_manager.is_model_installed():
            if getattr(self.engine, "use_mock", False):
                self._load_installed_model()
            self._update_mock_banner()
        else:
            reply = QMessageBox.question(
                self,
                "Auto-Download Typhoon ASR Model",
                "Typhoon ASR speech recognition model weights were not found locally.\n\n"
                "Would you like to automatically download official models now? (~480 MB)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.open_model_download_dialog(auto_start=True)

    def _load_installed_model(self) -> bool:
        """Attempt to load real model from local disk if available."""
        if not self.model_manager.is_model_installed():
            return False
        try:
            pref = self.combo_hardware.currentData() or "CPU"
            model_path = self.model_manager.get_model_path()
            decoder_path = self.model_manager.get_decoder_path()
            tokenizer_path = self.model_manager.get_tokenizer_path()
            logger.info("Auto-loading installed model: %s (decoder: %s, pref: %s)", model_path, decoder_path, pref)
            self.engine = TyphoonONNXEngine(
                model_path=model_path,
                decoder_path=decoder_path,
                tokenizer_path=tokenizer_path,
                use_mock=False,
                user_preference=pref,
            )
            self._update_mock_banner()
            self._update_hardware_status()
            logger.info("Successfully loaded Typhoon ASR model (%s)", self.engine.get_provider_status_text())
            return True
        except Exception as exc:
            logger.warning("Could not auto-load installed model: %s", exc)
            return False

    @Slot(str, str)
    def _on_model_installed(self, model_path: str, tokenizer_path: str) -> None:
        """Hot-reload ONNX Engine when new weights are installed."""
        try:
            pref = self.combo_hardware.currentData() or "CPU"
            logger.info("Hot-reloading ONNX Engine: model=%s, tokenizer=%s, preference=%s", model_path, tokenizer_path, pref)
            tok_p = tokenizer_path if tokenizer_path and Path(tokenizer_path).exists() else None
            dec_p = self.model_manager.get_decoder_path()
            self.engine = TyphoonONNXEngine(
                model_path=model_path,
                decoder_path=dec_p,
                tokenizer_path=tok_p,
                use_mock=False,
                user_preference=pref,
            )
            self._update_mock_banner()
            self._update_hardware_status()
            self.statusBar().showMessage("✅ Real Typhoon ASR Model loaded and active!")
            QMessageBox.information(
                self,
                "Engine Active",
                f"Typhoon ASR Model is now actively loaded for real transcription!\n\n{self.engine.get_provider_status_text()}",
            )
        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)
            QMessageBox.critical(self, "Load Error", f"Failed to initialize ONNX Engine:\n{exc}")

    @Slot()
    def start_live_transcription(self) -> None:
        """Start real-time audio streaming transcription worker."""
        if self.streaming_worker and self.streaming_worker.isRunning():
            return

        device_index = self.device_combo.currentData()
        self.streaming_worker = ASRStreamingWorker(
            engine=self.engine,
            device_index=device_index,
            parent=self,
        )

        self.streaming_worker.text_updated.connect(self._on_live_text_updated)
        self.streaming_worker.level_updated.connect(self.vu_meter.set_level)
        self.streaming_worker.status_changed.connect(lambda s: self.statusBar().showMessage(s))
        self.streaming_worker.error_occurred.connect(self._on_worker_error)

        self.streaming_worker.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸ Pause")
        self.btn_stop.setEnabled(True)
        self.device_combo.setEnabled(False)

    @Slot()
    def toggle_pause_live_transcription(self) -> None:
        """Toggle pause/resume state for real-time live transcription."""
        if not self.streaming_worker:
            return
        if self.streaming_worker.is_paused():
            self.streaming_worker.resume()
            self.btn_pause.setText("⏸ Pause")
            self.statusBar().showMessage("Live transcription resumed.")
        else:
            self.streaming_worker.pause()
            self.btn_pause.setText("▶ Resume")
            self.statusBar().showMessage("Live transcription paused.")

    @Slot()
    def stop_live_transcription(self) -> None:
        """Stop real-time audio streaming worker."""
        if self.streaming_worker:
            self.streaming_worker.stop()
            self.streaming_worker = None

        self.vu_meter.set_level(0.0)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Pause")
        self.btn_stop.setEnabled(False)
        self.device_combo.setEnabled(True)
        self.statusBar().showMessage("Live transcription stopped.")

    @Slot(str, str)
    def _on_live_text_updated(self, chunk_text: str, full_text: str) -> None:
        """Update live transcript in UI."""
        self.transcript_edit.setPlainText(full_text)
        if self.cb_autoscroll.isChecked():
            scrollbar = self.transcript_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @Slot(str)
    def _on_worker_error(self, err_msg: str) -> None:
        self.statusBar().showMessage(f"Error: {err_msg}")
        QMessageBox.warning(self, "Transcription Error", f"An error occurred:\n{err_msg}")
        self.stop_live_transcription()

    @Slot()
    def copy_transcript_to_clipboard(self) -> None:
        """Copy full transcript to system clipboard."""
        text = self.transcript_edit.toPlainText().strip()
        if not text:
            self.statusBar().showMessage("Nothing to copy.")
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(text, QClipboard.Clipboard)
        self.statusBar().showMessage("✅ Transcript copied to clipboard!")

    @Slot()
    def clear_transcript(self) -> None:
        """Clear transcript editor."""
        self.transcript_edit.clear()
        self.current_segments.clear()
        self.statusBar().showMessage("Transcript cleared.")

    @Slot()
    def open_export_dialog(self) -> None:
        """Open Export Dialog for saving transcript."""
        full_text = self.transcript_edit.toPlainText().strip()
        if not full_text:
            QMessageBox.information(self, "Export", "No transcript text available to export.")
            return
        dialog = ExportDialog(full_text=full_text, segments=self.current_segments, parent=self)
        dialog.exec()

    def closeEvent(self, event) -> None:
        """Clean shutdown of all worker threads on window close."""
        self.stop_live_transcription()
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
        if self.ram_timer:
            self.ram_timer.stop()
        event.accept()
