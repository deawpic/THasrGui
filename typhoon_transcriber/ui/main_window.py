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
    QMenu,
)
from PySide6.QtCore import Qt, Slot, QTimer, QUrl, QPoint
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

from ..config import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_CACHE_DIR,
    SUPPORTED_AUDIO_EXTENSIONS,
    LAST_SESSION_QUEUE_FILE,
)
from ..models.onnx_engine import TyphoonONNXEngine
from ..models.model_manager import ModelManager
from ..audio.device_manager import AudioDeviceManager, AudioDevice
from ..transcriber.streaming_worker import ASRStreamingWorker
from ..transcriber.batch_queue import (
    BatchItem,
    scan_files_and_folders,
    BatchQueueWorker,
    save_batch_queue,
    load_batch_queue,
    save_last_session_queue,
    load_last_session_queue,
    clear_last_session_queue,
)
from ..transcriber.exporter import TranscriptSegment
from .styles import DARK_THEME_QSS, LIGHT_THEME_QSS
from .widgets.vu_meter import VUMeterWidget
from .widgets.export_dialog import ExportDialog
from .widgets.download_dialog import ModelDownloadDialog
from .widgets.batch_dialog import BatchConfigDialog
from .widgets.post_processing_dialog import PostProcessingDialog

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

    def __init__(
        self,
        engine: Optional[TyphoonONNXEngine] = None,
        parent=None,
        session_file: Optional[Path] = None,
        restore_session: bool = True,
    ):
        super().__init__(parent)
        self.session_file = session_file or LAST_SESSION_QUEUE_FILE
        self.restore_session = restore_session
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
        self._restore_last_session_queue()

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

        # AI Post-Processing Advice & Prompt Button
        self.btn_post_processing = QPushButton("💡 AI Post-Processing", self)
        self.btn_post_processing.setToolTip("คำแนะนำและ Prompt สำหรับส่งข้อความให้ AI (Gemini/ChatGPT/Claude/Qwen) ขัดเกลาคำผิด")
        self.btn_post_processing.clicked.connect(self.open_post_processing_dialog)

        # Theme Toggle Button
        self.btn_theme = QPushButton("🌓 Theme", self)
        self.btn_theme.clicked.connect(self.toggle_theme)

        header_bar.addWidget(lbl_hw)
        header_bar.addWidget(self.combo_hardware)
        header_bar.addWidget(self.lbl_provider_status)
        header_bar.addStretch()
        header_bar.addWidget(self.btn_post_processing)
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
        self._apply_transcript_font()

        # Connect Ctrl+MouseWheel zooming on transcript_edit
        orig_wheel_event = self.transcript_edit.wheelEvent
        def _on_transcript_wheel(event):
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    self._adjust_font_size(2)
                elif delta < 0:
                    self._adjust_font_size(-2)
                event.accept()
            else:
                orig_wheel_event(event)
        self.transcript_edit.wheelEvent = _on_transcript_wheel

        layout.addWidget(self.transcript_edit, stretch=1)

        # Live Bottom Action Bar
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(8)

        self.cb_autoscroll = QCheckBox("Auto-scroll", self)
        self.cb_autoscroll.setChecked(True)

        self.btn_copy = QPushButton("📋 Copy Text", self)
        self.btn_copy.setToolTip("คัดลอกข้อความทั้งหมดไปยังคลิปบอร์ด (Copy transcript to clipboard)")
        self.btn_copy.clicked.connect(self.copy_transcript_to_clipboard)

        self.btn_clear = QPushButton("🗑 Clear", self)
        self.btn_clear.setToolTip("ล้างข้อความถอดเสียงทั้งหมดบนหน้าจอ (Clear all transcript text)")
        self.btn_clear.clicked.connect(self.clear_transcript)

        v_sep = QFrame(self)
        v_sep.setFrameShape(QFrame.VLine)
        v_sep.setFrameShadow(QFrame.Sunken)

        self.lbl_font_title = QLabel("🗚 ขนาดตัวอักษร:", self)

        self.btn_zoom_out = QPushButton("➖ เล็กลง", self)
        self.btn_zoom_out.setToolTip("ลดขนาดตัวอักษรข้อความ (Zoom Out -2pt) [Ctrl + -]")
        self.btn_zoom_out.clicked.connect(lambda: self._adjust_font_size(-2))

        self.lbl_font_size = QLabel(f"{self.font_size} pt", self)
        self.lbl_font_size.setFixedWidth(45)
        self.lbl_font_size.setAlignment(Qt.AlignCenter)
        self.lbl_font_size.setStyleSheet("font-weight: bold;")

        self.btn_zoom_in = QPushButton("➕ ใหญ่ขึ้น", self)
        self.btn_zoom_in.setToolTip("ขยายขนาดตัวอักษรข้อความ (Zoom In +2pt) [Ctrl + +]")
        self.btn_zoom_in.clicked.connect(lambda: self._adjust_font_size(2))

        self.btn_export = QPushButton("💾 Export Transcript...", self)
        self.btn_export.setToolTip("บันทึกส่งออกไฟล์ถอดเสียง (.txt, .srt, .vtt, .json)")
        self.btn_export.setStyleSheet("background-color: #89b4fa; color: #11111b; font-weight: bold;")
        self.btn_export.clicked.connect(self.open_export_dialog)

        bottom_bar.addWidget(self.cb_autoscroll)
        bottom_bar.addWidget(self.btn_copy)
        bottom_bar.addWidget(self.btn_clear)
        bottom_bar.addWidget(v_sep)
        bottom_bar.addWidget(self.lbl_font_title)
        bottom_bar.addWidget(self.btn_zoom_out)
        bottom_bar.addWidget(self.lbl_font_size)
        bottom_bar.addWidget(self.btn_zoom_in)
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
        self.btn_batch_remove.setToolTip("Remove selected rows from queue (Shift / Ctrl for multi-select)")
        self.btn_batch_remove.clicked.connect(self.remove_selected_batch_items)

        self.btn_batch_clear = QPushButton("🗑 Clear List", self)
        self.btn_batch_clear.setToolTip("Clear all items from batch queue")
        self.btn_batch_clear.clicked.connect(self.clear_batch_items)

        self.btn_batch_save_queue = QPushButton("💾 Save Queue...", self)
        self.btn_batch_save_queue.setToolTip("Save current queue & progress to a JSON file to resume later")
        self.btn_batch_save_queue.clicked.connect(self.save_queue_dialog)

        self.btn_batch_load_queue = QPushButton("📂 Load Queue...", self)
        self.btn_batch_load_queue.setToolTip("Load a saved queue to continue unfinished files")
        self.btn_batch_load_queue.clicked.connect(self.load_queue_dialog)

        batch_bar.addWidget(self.btn_batch_add_files)
        batch_bar.addWidget(self.btn_batch_add_folder)
        batch_bar.addWidget(self.btn_batch_remove)
        batch_bar.addWidget(self.btn_batch_clear)
        batch_bar.addWidget(self.btn_batch_save_queue)
        batch_bar.addWidget(self.btn_batch_load_queue)
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
        self.txt_batch_dest.textChanged.connect(self._on_batch_dest_text_changed)
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

        self.cb_batch_txt.toggled.connect(lambda: self._auto_save_session_queue())
        self.cb_batch_srt.toggled.connect(lambda: self._auto_save_session_queue())
        self.cb_batch_vtt.toggled.connect(lambda: self._auto_save_session_queue())
        self.cb_batch_json.toggled.connect(lambda: self._auto_save_session_queue())

        fmt_row.addWidget(self.cb_batch_txt)
        fmt_row.addWidget(self.cb_batch_srt)
        fmt_row.addWidget(self.cb_batch_vtt)
        fmt_row.addWidget(self.cb_batch_json)
        fmt_row.addStretch()
        fmt_row.addWidget(self.cb_batch_overwrite)
        settings_layout.addLayout(fmt_row)

        layout.addWidget(settings_group)

        # Batch Table Widget (5 columns, multi-selection with Shift/Ctrl, right-click edit output)
        self.batch_table = QTableWidget(0, 5, self)
        self.batch_table.setHorizontalHeaderLabels([
            "#", "File Name", "Size (MB)", "Status", "Output Destination / Details"
        ])
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.batch_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.batch_table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.batch_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.batch_table.customContextMenuRequested.connect(self._on_batch_table_context_menu)
        self.batch_table.cellDoubleClicked.connect(self._on_batch_table_cell_double_clicked)
        layout.addWidget(self.batch_table, stretch=1)

        # Batch Progress Indicators
        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(4)

        self.lbl_batch_overall = QLabel("Overall Progress: 0 / 0 files (0%)", self)
        self.lbl_batch_overall.setStyleSheet("font-weight: bold; color: #89b4fa;")
        self.prog_batch_overall = QProgressBar(self)
        self.prog_batch_overall.setValue(0)
        self.lbl_batch_file = QLabel("Current File: Ready", self)
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

        self.shortcut_zoom_in = QShortcut(QKeySequence("Ctrl+="), self)
        self.shortcut_zoom_in.activated.connect(lambda: self._adjust_font_size(2))
        self.shortcut_zoom_in2 = QShortcut(QKeySequence("Ctrl++"), self)
        self.shortcut_zoom_in2.activated.connect(lambda: self._adjust_font_size(2))

        self.shortcut_zoom_out = QShortcut(QKeySequence("Ctrl+-"), self)
        self.shortcut_zoom_out.activated.connect(lambda: self._adjust_font_size(-2))

        self.shortcut_zoom_reset = QShortcut(QKeySequence("Ctrl+0"), self)
        self.shortcut_zoom_reset.activated.connect(lambda: self._set_font_size(16))

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
            if getattr(dev, "is_wasapi_loopback", False):
                tag = " [WASAPI Loopback]"
            elif dev.is_loopback:
                tag = " [Loopback]"
            else:
                tag = " [Mic]"
            default_tag = " (Default)" if dev.is_default else ""
            display_name = f"{dev.name}{tag}{default_tag}"
            self.device_combo.addItem(display_name, dev.index)

        default_dev = self.device_manager.get_default_device()
        if default_dev:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == default_dev.index:
                    self.device_combo.setCurrentIndex(i)
                    break

    def _apply_transcript_font(self) -> None:
        """Apply font size to transcript text editor reliably across styles and documents."""
        font = self.transcript_edit.font()
        font.setPointSize(self.font_size)
        font.setStyleStrategy(QFont.PreferAntialias | QFont.PreferQuality)
        font.setHintingPreference(QFont.PreferVerticalHinting)
        self.transcript_edit.setFont(font)
        self.transcript_edit.document().setDefaultFont(font)
        self.transcript_edit.setStyleSheet(f"font-size: {self.font_size}pt; line-height: 1.6;")
        if hasattr(self, "lbl_font_size"):
            self.lbl_font_size.setText(f"{self.font_size} pt")

    def _set_font_size(self, size: int) -> None:
        """Directly set transcript display font size."""
        self.font_size = max(10, min(size, 36))
        self._apply_transcript_font()

    def _adjust_font_size(self, delta: int) -> None:
        """Increase or decrease transcript display font size."""
        self.font_size = max(10, min(self.font_size + delta, 36))
        self._apply_transcript_font()

    def _apply_theme(self) -> None:
        """Apply Dark or Light QSS stylesheet."""
        qss = DARK_THEME_QSS if self.is_dark_theme else LIGHT_THEME_QSS
        self.setStyleSheet(qss)
        app = QApplication.instance()
        if app:
            app.setStyleSheet(qss)
            from typhoon_transcriber.ui.styles import configure_application_typography
            configure_application_typography(app)
        if hasattr(self, "lbl_batch_overall"):
            self.lbl_batch_overall.setStyleSheet(
                f"font-weight: bold; color: {'#89b4fa' if self.is_dark_theme else '#0071e3'};"
            )
        if hasattr(self, "transcript_edit"):
            self._apply_transcript_font()

    def toggle_theme(self) -> None:
        """Switch between dark and light themes."""
        self.is_dark_theme = not self.is_dark_theme
        self._apply_theme()

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
        self._auto_save_session_queue()
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
            self._auto_save_session_queue()

    def _on_batch_dest_text_changed(self, text: str) -> None:
        """Handle manual typing or update to destination path."""
        if text.strip():
            self.dest_batch_dir = Path(text.strip())
            self._auto_save_session_queue()

    def _refresh_batch_table(self) -> None:
        """Update batch table rows to reflect self.batch_items (5 columns)."""
        self.batch_table.setRowCount(len(self.batch_items))
        for row, item in enumerate(self.batch_items):
            # Col 0: #
            self.batch_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))

            # Col 1: File Name
            self.batch_table.setItem(row, 1, QTableWidgetItem(item.source_file.name))

            # Col 2: Size (MB)
            size_mb = item.source_file.stat().st_size / (1024 * 1024) if item.source_file.exists() else 0.0
            self.batch_table.setItem(row, 2, QTableWidgetItem(f"{size_mb:.2f} MB"))

            # Col 3: Status
            status_item = QTableWidgetItem(item.status)
            self._apply_status_color(status_item, item.status)
            self.batch_table.setItem(row, 3, status_item)

            # Col 4: Output Destination / Details
            if item.status == "Completed":
                out_dir = item.custom_dest_dir or (self.dest_batch_dir / item.rel_path.parent)
                detail_str = str(out_dir / item.source_file.stem)
            elif item.status == "Failed":
                detail_str = f"Error: {item.error_message or 'Failed'}"
            elif item.custom_dest_dir:
                detail_str = f"[Custom] {item.custom_dest_dir}"
            else:
                detail_str = str(self.dest_batch_dir / item.rel_path.parent)

            out_item = QTableWidgetItem(detail_str)
            if item.custom_dest_dir:
                out_item.setToolTip(f"Custom Destination: {item.custom_dest_dir}\n(Right-click or double-click to edit)")
            else:
                out_item.setToolTip(f"Destination: {detail_str}\n(Right-click or double-click to edit)")
            self.batch_table.setItem(row, 4, out_item)

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
        self._auto_save_session_queue()
        self.statusBar().showMessage(f"Removed {len(selected_rows)} item(s) from batch queue.")

    def clear_batch_items(self) -> None:
        """Clear all items from batch queue."""
        if self.batch_worker and self.batch_worker.isRunning():
            QMessageBox.warning(self, "Batch Running", "Cannot clear queue while batch transcription is running.")
            return
        self.batch_items.clear()
        clear_last_session_queue()
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

        # Check if all items are already completed
        completed_items = [it for it in self.batch_items if it.status == "Completed"]
        if len(completed_items) == len(self.batch_items):
            reply = QMessageBox.question(
                self,
                "All Files Completed",
                "All files in the queue are already completed.\n\nDo you want to re-transcribe all files from the beginning?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                for item in self.batch_items:
                    item.status = "Pending"
                    item.error_message = None
            else:
                return
        else:
            # Preserve completed items and reset pending/failed/processing items
            for item in self.batch_items:
                if item.status != "Completed":
                    item.status = "Pending"
                    item.error_message = None

        dest_str = self.txt_batch_dest.text().strip()
        if not dest_str:
            self.dest_batch_dir = Path.home() / "Transcripts"
            self.txt_batch_dest.setText(str(self.dest_batch_dir))
        else:
            self.dest_batch_dir = Path(dest_str)

        self.dest_batch_dir.mkdir(parents=True, exist_ok=True)

        selected_formats = self._get_selected_batch_formats()
        if not selected_formats:
            QMessageBox.warning(self, "Format Selection", "Please select at least one output format (.txt, .srt, .vtt, or .json).")
            return

        self._refresh_batch_table()
        self._auto_save_session_queue()

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
            item_status = self.batch_table.item(idx, 3)
            if item_status:
                item_status.setText(status)
                self._apply_status_color(item_status, status)

            item_detail = self.batch_table.item(idx, 4)
            if item_detail and detail_info:
                item_detail.setText(detail_info)
            self._auto_save_session_queue()

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

        self._auto_save_session_queue()
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

    @Slot()
    def open_post_processing_dialog(self) -> None:
        """Open AI Post-Processing Recommendations and Prompt Dialog."""
        dialog = PostProcessingDialog(parent=self, is_dark=self.is_dark_theme)
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

        existing = self.transcript_edit.toPlainText().strip()
        if existing:
            self.streaming_worker.set_initial_text(existing)

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
        """Stop real-time audio streaming worker cleanly and safely."""
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸ Pause")
        self.btn_start.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.vu_meter.set_level(0.0)
        self.statusBar().showMessage("Stopping live transcription...")

        if self.streaming_worker:
            worker = self.streaming_worker
            self.streaming_worker = None
            worker.stop()

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
        if self.streaming_worker:
            self.streaming_worker.reset_transcript()
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
        """Clean shutdown of all worker threads and save session state on window close."""
        self.stop_live_transcription()
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
        self._auto_save_session_queue()
        if self.ram_timer:
            self.ram_timer.stop()
        event.accept()

    # =========================================================================
    # BATCH QUEUE PERSISTENCE, FORMATS & CONTEXT MENU
    # =========================================================================
    def _get_selected_batch_formats(self) -> List[str]:
        """Return list of selected format extensions for batch export."""
        formats = []
        if self.cb_batch_txt.isChecked():
            formats.append(".txt")
        if self.cb_batch_srt.isChecked():
            formats.append(".srt")
        if self.cb_batch_vtt.isChecked():
            formats.append(".vtt")
        if self.cb_batch_json.isChecked():
            formats.append(".json")
        return formats

    def _apply_selected_batch_formats(self, formats: List[str]) -> None:
        """Apply format selection checkboxes from list of extensions."""
        self.cb_batch_txt.setChecked(".txt" in formats)
        self.cb_batch_srt.setChecked(".srt" in formats)
        self.cb_batch_vtt.setChecked(".vtt" in formats)
        self.cb_batch_json.setChecked(".json" in formats)

    @Slot(QPoint)
    def _on_batch_table_context_menu(self, pos: QPoint) -> None:
        """Show context menu for batch table rows to edit destination, reset status, or remove."""
        selected_indexes = self.batch_table.selectedIndexes()
        selected_rows = sorted(set(index.row() for index in selected_indexes))

        row_at_pos = self.batch_table.rowAt(pos.y())
        if row_at_pos >= 0 and row_at_pos not in selected_rows:
            self.batch_table.selectRow(row_at_pos)
            selected_rows = [row_at_pos]

        if not selected_rows:
            return

        menu = QMenu(self)
        count_label = f" ({len(selected_rows)} items)" if len(selected_rows) > 1 else ""

        act_edit_dest = menu.addAction(f"✏️ Change Output Destination...{count_label}")
        act_reset = menu.addAction(f"🔄 Reset Status to Pending{count_label}")
        menu.addSeparator()
        act_remove = menu.addAction(f"➖ Remove Selected{count_label}")

        first_row = selected_rows[0]
        act_open = None
        if len(selected_rows) == 1 and 0 <= first_row < len(self.batch_items):
            item = self.batch_items[first_row]
            target_dir = item.custom_dest_dir or (self.dest_batch_dir / item.rel_path.parent)
            if target_dir.exists():
                menu.addSeparator()
                act_open = menu.addAction("📂 Open Destination Folder")

        action = menu.exec(self.batch_table.viewport().mapToGlobal(pos))
        if not action:
            return

        if action == act_edit_dest:
            self._edit_output_folder_for_rows(selected_rows)
        elif action == act_reset:
            for r in selected_rows:
                if 0 <= r < len(self.batch_items):
                    self.batch_items[r].status = "Pending"
                    self.batch_items[r].error_message = None
            self._refresh_batch_table()
            self._auto_save_session_queue()
            self.statusBar().showMessage(f"Reset {len(selected_rows)} item(s) to Pending.")
        elif action == act_remove:
            self.remove_selected_batch_items()
        elif act_open and action == act_open:
            item = self.batch_items[first_row]
            target_dir = item.custom_dest_dir or (self.dest_batch_dir / item.rel_path.parent)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target_dir)))

    @Slot(int, int)
    def _on_batch_table_cell_double_clicked(self, row: int, col: int) -> None:
        """Allow double-clicking on Output Destination column to edit folder directly."""
        if col == 4 and 0 <= row < len(self.batch_items):
            self._edit_output_folder_for_rows([row])

    def _edit_output_folder_for_rows(self, rows: List[int]) -> None:
        """Prompt user for a directory to set as custom destination for specified rows."""
        if not rows:
            return
        first_item = self.batch_items[rows[0]]
        initial_dir = first_item.custom_dest_dir or (self.dest_batch_dir / first_item.rel_path.parent)
        if not initial_dir.exists():
            initial_dir = self.dest_batch_dir

        folder = QFileDialog.getExistingDirectory(
            self,
            f"Select Output Destination Folder for {len(rows)} Item(s)",
            str(initial_dir),
        )
        if folder:
            chosen_dir = Path(folder).resolve()
            for r in rows:
                if 0 <= r < len(self.batch_items):
                    self.batch_items[r].custom_dest_dir = chosen_dir
            self._refresh_batch_table()
            self._auto_save_session_queue()
            self.statusBar().showMessage(f"Updated output destination for {len(rows)} item(s) to '{chosen_dir.name}'")

    @Slot()
    def save_queue_dialog(self) -> None:
        """Save current batch queue to a JSON file to resume later."""
        if not self.batch_items:
            QMessageBox.information(
                self,
                "Save Queue",
                "Batch queue is empty. Add files or folders first.",
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Batch Queue",
            str(Path.home() / "batch_queue.json"),
            "JSON Queue Files (*.json);;All Files (*)",
        )
        if file_path:
            try:
                formats = self._get_selected_batch_formats()
                save_batch_queue(
                    filepath=file_path,
                    items=self.batch_items,
                    dest_dir=self.dest_batch_dir,
                    formats=formats,
                )
                completed_count = sum(1 for it in self.batch_items if it.status == "Completed")
                self.statusBar().showMessage(f"Saved queue ({len(self.batch_items)} items) to {Path(file_path).name}")
                QMessageBox.information(
                    self,
                    "Queue Saved",
                    f"Successfully saved {len(self.batch_items)} items to:\n{file_path}\n\n"
                    f"• Completed: {completed_count}\n"
                    f"• Pending / Other: {len(self.batch_items) - completed_count}",
                )
            except Exception as exc:
                logger.exception("Failed to save queue: %s", exc)
                QMessageBox.critical(self, "Save Error", f"Could not save queue file:\n{exc}")

    @Slot()
    def load_queue_dialog(self, *args, append_mode: Optional[bool] = None) -> None:
        """Load one or multiple batch queues from JSON file(s) with Append or Replace options."""
        if self.batch_worker and self.batch_worker.isRunning():
            QMessageBox.warning(self, "Batch Running", "Cannot load queue while batch transcription is running.")
            return

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Batch Queue File(s) / เลือกไฟล์คิวงาน (เลือกได้หลายไฟล์พร้อมกัน)",
            str(Path.home()),
            "JSON Queue Files (*.json);;All Files (*)",
        )
        if not file_paths:
            return

        # If current queue already contains items, ask whether to Append or Replace
        if append_mode is None:
            if self.batch_items:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Load Batch Queue / โหลดคิวงาน")
                msg_box.setIcon(QMessageBox.Question)
                msg_box.setText(
                    f"ปัจจุบันมีรายการในคิวอยู่แล้ว {len(self.batch_items)} ไฟล์\n\n"
                    f"คุณต้องการนำไฟล์จาก {len(file_paths)} คิวงานที่เลือก มาดำเนินการอย่างไร?"
                )
                btn_append = msg_box.addButton("➕ เพิ่มต่อท้าย (Append)", QMessageBox.AcceptRole)
                btn_replace = msg_box.addButton("🔄 แทนที่คิวเดิม (Replace)", QMessageBox.DestructiveRole)
                btn_cancel = msg_box.addButton("ยกเลิก (Cancel)", QMessageBox.RejectRole)
                msg_box.setDefaultButton(btn_append)
                msg_box.exec()

                clicked = msg_box.clickedButton()
                if clicked == btn_cancel or clicked is None:
                    return
                append_mode = (clicked == btn_append)
            else:
                append_mode = True

        if not append_mode:
            self.batch_items.clear()

        existing_paths = {it.source_file.resolve() for it in self.batch_items}
        newly_added_count = 0
        status_updated_count = 0
        loaded_files_count = 0
        last_dest_dir = None
        last_formats = None

        for fp in file_paths:
            try:
                items, dest_dir, formats = load_batch_queue(fp)
                if not items:
                    continue
                loaded_files_count += 1
                if dest_dir and not last_dest_dir:
                    last_dest_dir = dest_dir
                if formats and not last_formats:
                    last_formats = formats

                for item in items:
                    resolved_p = item.source_file.resolve()
                    if resolved_p not in existing_paths:
                        self.batch_items.append(item)
                        existing_paths.add(resolved_p)
                        newly_added_count += 1
                    else:
                        # If file already exists, update Completed status if loaded item has it
                        for existing in self.batch_items:
                            if existing.source_file.resolve() == resolved_p:
                                if item.status == "Completed" and existing.status != "Completed":
                                    existing.status = "Completed"
                                    existing.generated_files = item.generated_files
                                    status_updated_count += 1
            except Exception as exc:
                logger.exception("Failed to load queue from %s: %s", fp, exc)

        if loaded_files_count == 0:
            QMessageBox.warning(self, "Load Queue", "No valid queue items found in the selected file(s).")
            return

        if (not append_mode or not self.dest_batch_dir.exists()) and last_dest_dir:
            self.dest_batch_dir = last_dest_dir
            self.txt_batch_dest.setText(str(self.dest_batch_dir))
        if not append_mode and last_formats:
            self._apply_selected_batch_formats(last_formats)

        self._refresh_batch_table()
        self._auto_save_session_queue()

        completed_count = sum(1 for it in self.batch_items if it.status == "Completed")
        pending_count = len(self.batch_items) - completed_count
        self.statusBar().showMessage(
            f"Loaded {loaded_files_count} queue file(s): +{newly_added_count} items (Total: {len(self.batch_items)})."
        )

        update_msg = f"\n• อัปเดตสถานะเป็นเสร็จแล้ว (Updated to Completed): {status_updated_count}" if status_updated_count > 0 else ""
        mode_str = "เพิ่มต่อท้าย (Appended)" if append_mode else "แทนที่คิวใหม่ (Replaced)"
        QMessageBox.information(
            self,
            "Queue Loaded",
            f"โหลดคิวงานสำเร็จจาก {loaded_files_count} ไฟล์ ({mode_str}):\n"
            f"• เพิ่มไฟล์ใหม่: {newly_added_count} รายการ{update_msg}\n"
            f"• รวมรายการในคิวทั้งหมด: {len(self.batch_items)} รายการ\n\n"
            f"  - ✅ เสร็จแล้ว (Completed): {completed_count}\n"
            f"  - ⏳ รอดำเนินการ (Pending): {pending_count}\n\n"
            f"กด '🚀 Start Batch Transcription' เพื่อเริ่มแปลงไฟล์ต่อได้ทันที",
        )

    def _auto_save_session_queue(self) -> None:
        """Auto-save current queue state to cache for crash/accidental close recovery."""
        try:
            formats = self._get_selected_batch_formats()
            save_last_session_queue(
                items=self.batch_items,
                dest_dir=self.dest_batch_dir,
                formats=formats,
            )
        except Exception as exc:
            logger.debug("Failed to auto-save session queue: %s", exc)

    def _restore_last_session_queue(self) -> None:
        """Restore batch queue from last session cache on startup."""
        try:
            items, dest_dir, formats = load_last_session_queue()
            if items:
                # If an item was left in 'Processing' state (due to crash or abrupt kill), reset it to 'Pending'
                for item in items:
                    if item.status == "Processing":
                        item.status = "Pending"

                self.batch_items = items
                if dest_dir:
                    self.dest_batch_dir = dest_dir
                    self.txt_batch_dest.setText(str(self.dest_batch_dir))
                if formats:
                    self._apply_selected_batch_formats(formats)

                self._refresh_batch_table()
                completed = sum(1 for it in items if it.status == "Completed")
                pending = len(items) - completed
                self.statusBar().showMessage(
                    f"Restored previous session queue: {len(items)} items ({completed} completed, {pending} pending)."
                )
                logger.info(
                    "Restored %d items from last session queue (%d completed, %d pending)",
                    len(items),
                    completed,
                    pending,
                )
        except Exception as exc:
            logger.warning("Failed to restore last session queue: %s", exc)

