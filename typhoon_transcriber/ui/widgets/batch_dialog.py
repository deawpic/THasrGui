"""
Batch Configuration and Progress Monitoring Dialogs for Multi-File & Multi-Folder Transcription.
Supports folder structure mirroring and multi-format subtitle exports.
"""

from pathlib import Path
from typing import List, Optional, Set
import os
import subprocess
import sys
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QGroupBox,
    QLineEdit,
    QFrame,
)
from PySide6.QtCore import Qt, Slot, QUrl
from PySide6.QtGui import QColor, QDesktopServices

from ...transcriber.batch_queue import BatchItem, BatchQueueWorker
from ...models.onnx_engine import TyphoonONNXEngine


class BatchConfigDialog(QDialog):
    """
    Dialog where user reviews scanned batch items, selects destination folder,
    and configures output formats (.txt, .srt, .vtt, .json) with folder mirroring.
    """

    def __init__(
        self,
        items: List[BatchItem],
        engine: Optional[TyphoonONNXEngine] = None,
        initial_dest_dir: Optional[Path] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.items = items
        self.engine = engine
        self.worker: Optional[BatchQueueWorker] = None
        self.dest_dir: Path = Path(initial_dest_dir) if initial_dest_dir else (Path.home() / "Transcripts")
        self.selected_formats: List[str] = [".txt", ".srt"]
        self.start_immediately: bool = False

        self.setWindowTitle("Batch Audio Transcription & Destination Settings")
        self.resize(750, 520)
        self.setMinimumSize(600, 400)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header_lbl = QLabel(f"📁 Batch Audio Transcriber — {len(self.items)} File(s) Detected")
        header_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(header_lbl)

        # Destination Folder Picker
        dest_group = QGroupBox("1. Destination Directory (Folder Structure will be Mirrored)")
        dest_layout = QHBoxLayout(dest_group)

        self.txt_dest = QLineEdit(str(self.dest_dir), self)
        self.btn_browse_dest = QPushButton("📂 Browse...", self)
        self.btn_browse_dest.clicked.connect(self._browse_dest_folder)

        dest_layout.addWidget(self.txt_dest, stretch=1)
        dest_layout.addWidget(self.btn_browse_dest)
        layout.addWidget(dest_group)

        # Output Format Selection Group
        fmt_group = QGroupBox("2. Target Output Formats")
        fmt_layout = QHBoxLayout(fmt_group)

        self.cb_txt = QCheckBox("Plain Text (.txt)", self)
        self.cb_srt = QCheckBox("SubRip Subtitles (.srt)", self)
        self.cb_vtt = QCheckBox("WebVTT Subtitles (.vtt)", self)
        self.cb_json = QCheckBox("JSON with Timecodes (.json)", self)

        self.cb_txt.setChecked(True)
        self.cb_srt.setChecked(True)

        self.cb_overwrite = QCheckBox("Overwrite existing files", self)
        self.cb_overwrite.setChecked(True)

        fmt_layout.addWidget(self.cb_txt)
        fmt_layout.addWidget(self.cb_srt)
        fmt_layout.addWidget(self.cb_vtt)
        fmt_layout.addWidget(self.cb_json)
        fmt_layout.addStretch()
        fmt_layout.addWidget(self.cb_overwrite)
        layout.addWidget(fmt_group)

        # Scanned Files Table
        table_lbl = QLabel("3. Review Input Files & Mirrored Output Hierarchy:")
        table_lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(table_lbl)

        self.table = QTableWidget(len(self.items), 3, self)
        self.table.setHorizontalHeaderLabels(["File Name", "Relative Subfolder Path", "Size (MB)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

        for row, item in enumerate(self.items):
            self.table.setItem(row, 0, QTableWidgetItem(item.source_file.name))
            self.table.setItem(row, 1, QTableWidgetItem(str(item.rel_path.parent)))
            size_mb = item.source_file.stat().st_size / (1024 * 1024) if item.source_file.exists() else 0.0
            self.table.setItem(row, 2, QTableWidgetItem(f"{size_mb:.2f} MB"))

        layout.addWidget(self.table, stretch=1)

        # Bottom Buttons
        btn_box = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_add_to_queue = QPushButton("📥 Add to Queue", self)
        self.btn_add_to_queue.setStyleSheet("padding: 8px 14px; font-weight: bold;")
        self.btn_add_to_queue.clicked.connect(self._add_to_queue)

        self.btn_start = QPushButton("🚀 Start Batch Transcription", self)
        self.btn_start.setStyleSheet("background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 8px 18px;")
        self.btn_start.clicked.connect(self._start_batch)

        btn_box.addStretch()
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_add_to_queue)
        btn_box.addWidget(self.btn_start)
        layout.addLayout(btn_box)

    def _browse_dest_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder for Mirrored Transcripts",
            str(self.dest_dir),
        )
        if folder:
            self.dest_dir = Path(folder)
            self.txt_dest.setText(str(self.dest_dir))

    def _add_to_queue(self) -> None:
        if not self._validate_and_gather():
            return
        self.start_immediately = False
        self.accept()

    def _start_batch(self) -> None:
        if not self._validate_and_gather():
            return
        self.start_immediately = True
        self.accept()

    def _validate_and_gather(self) -> bool:
        dest_str = self.txt_dest.text().strip()
        if not dest_str:
            QMessageBox.warning(self, "Invalid Destination", "Please specify a destination folder.")
            return False

        self.dest_dir = Path(dest_str)
        self.dest_dir.mkdir(parents=True, exist_ok=True)

        selected_formats = []
        if self.cb_txt.isChecked():
            selected_formats.append(".txt")
        if self.cb_srt.isChecked():
            selected_formats.append(".srt")
        if self.cb_vtt.isChecked():
            selected_formats.append(".vtt")
        if self.cb_json.isChecked():
            selected_formats.append(".json")

        if not selected_formats:
            QMessageBox.warning(self, "Format Selection", "Please select at least one output format.")
            return False

        self.selected_formats = selected_formats
        return True


class BatchProgressDialog(QDialog):
    """
    Modal Progress Monitor Dialog showing real-time batch transcription progress,
    current file progress, and status table with mirrored folder output links.
    """

    def __init__(
        self,
        items: List[BatchItem],
        engine: TyphoonONNXEngine,
        dest_root_dir: Path,
        formats: List[str],
        overwrite: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.items = items
        self.engine = engine
        self.dest_root_dir = dest_root_dir
        self.formats = formats
        self.overwrite = overwrite
        self.worker: Optional[BatchQueueWorker] = None

        self.setWindowTitle("Processing Batch Audio Transcriptions...")
        self.resize(800, 550)
        self.setMinimumSize(650, 450)
        self._init_ui()
        self._start_worker()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Progress Overview Labels
        self.lbl_overall = QLabel("Overall Progress: 0 / 0 files (0%)", self)
        self.lbl_overall.setStyleSheet("font-size: 14px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(self.lbl_overall)

        self.prog_overall = QProgressBar(self)
        self.prog_overall.setValue(0)
        layout.addWidget(self.prog_overall)

        self.lbl_file = QLabel("Current File: Initializing...", self)
        self.lbl_file.setStyleSheet("color: #cdd6f4;")
        layout.addWidget(self.lbl_file)

        self.prog_file = QProgressBar(self)
        self.prog_file.setValue(0)
        layout.addWidget(self.prog_file)

        # Status Table
        self.table = QTableWidget(len(self.items), 4, self)
        self.table.setHorizontalHeaderLabels(["#", "File Name", "Status", "Mirrored Output Path"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

        for row, item in enumerate(self.items):
            self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.table.setItem(row, 1, QTableWidgetItem(item.source_file.name))
            status_item = QTableWidgetItem(item.status)
            self.table.setItem(row, 2, status_item)
            out_rel = str(self.dest_root_dir / item.rel_path.parent / item.source_file.stem)
            self.table.setItem(row, 3, QTableWidgetItem(out_rel))

        layout.addWidget(self.table, stretch=1)

        # Bottom Actions
        btn_box = QHBoxLayout()
        self.btn_open_folder = QPushButton("📂 Open Destination Folder", self)
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._open_dest_folder)

        self.btn_cancel = QPushButton("❌ Cancel Batch", self)
        self.btn_cancel.clicked.connect(self._cancel_batch)

        self.btn_close = QPushButton("Close", self)
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)

        btn_box.addWidget(self.btn_open_folder)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_close)
        layout.addLayout(btn_box)

    def _start_worker(self) -> None:
        self.worker = BatchQueueWorker(
            items=self.items,
            engine=self.engine,
            dest_root_dir=self.dest_root_dir,
            formats=self.formats,
            overwrite=self.overwrite,
            parent=self,
        )
        self.worker.overall_progress.connect(self._on_overall_progress)
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.item_status_changed.connect(self._on_item_status_changed)
        self.worker.batch_completed.connect(self._on_batch_completed)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    @Slot(int, int, str)
    def _on_overall_progress(self, current_idx: int, total_count: int, total_eta_str: str) -> None:
        pct = int((current_idx / total_count) * 100)
        self.prog_overall.setValue(pct)
        self.lbl_overall.setText(
            f"Overall Progress: {current_idx} / {total_count} files ({pct}%) — {total_eta_str}"
        )

    @Slot(str, int, str)
    def _on_file_progress(self, filename: str, pct: int, file_eta_str: str) -> None:
        self.prog_file.setValue(pct)
        self.lbl_file.setText(f"Current File: {filename} ({pct}%) — {file_eta_str}")

    @Slot(int, str, str)
    def _on_item_status_changed(self, idx: int, status: str, detail_info: str) -> None:
        item = self.table.item(idx, 2)
        if item:
            item.setText(status)
            if status == "Completed":
                item.setForeground(QColor("#a6e3a1"))  # Green
            elif status == "Processing":
                item.setForeground(QColor("#f9e2af"))  # Yellow
            elif status == "Failed":
                item.setForeground(QColor("#f38ba8"))  # Red
            elif status == "Skipped":
                item.setForeground(QColor("#a6adc8"))  # Gray

        out_item = self.table.item(idx, 3)
        if out_item and detail_info:
            out_item.setText(detail_info)

    @Slot(int, int, float)
    def _on_batch_completed(self, success_count: int, fail_count: int, elapsed_sec: float) -> None:
        self.prog_overall.setValue(100)
        self.prog_file.setValue(100)
        self.lbl_overall.setText(
            f"✅ Batch Completed: {success_count} Succeeded, {fail_count} Failed (Time: {elapsed_sec:.1f}s)"
        )
        self.lbl_file.setText("All files processed.")
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
        self.btn_open_folder.setEnabled(True)

        QMessageBox.information(
            self,
            "Batch Conversion Complete",
            f"Batch transcription completed successfully!\n\n"
            f"• Succeeded: {success_count} files\n"
            f"• Failed: {fail_count} files\n"
            f"• Output Directory: {self.dest_root_dir}\n"
            f"• Total Elapsed: {elapsed_sec:.1f}s",
        )

    @Slot(str)
    def _on_error(self, err_msg: str) -> None:
        QMessageBox.warning(self, "Batch Error", f"An error occurred in the batch queue:\n{err_msg}")

    def _cancel_batch(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.worker = None
        self.lbl_overall.setText("Batch transcription cancelled.")
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
        self.btn_open_folder.setEnabled(True)

    def _open_dest_folder(self) -> None:
        """Open destination folder in system file explorer."""
        if self.dest_root_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.dest_root_dir)))

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            self._cancel_batch()
        event.accept()
