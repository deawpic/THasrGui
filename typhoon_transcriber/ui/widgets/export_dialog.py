"""
Export Dialog for saving transcripts as .txt, .srt, .vtt, or .json.
"""

from pathlib import Path
from typing import List, Optional
import datetime
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QButtonGroup,
)
from ...transcriber.exporter import TranscriptExporter, TranscriptSegment


class ExportDialog(QDialog):
    """
    Modal dialog allowing user to choose export format and output destination.
    """

    def __init__(
        self,
        full_text: str,
        segments: Optional[List[TranscriptSegment]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.full_text = full_text
        self.segments = segments or [TranscriptSegment(0.0, 1.0, full_text)]
        self.setWindowTitle("Export Transcript")
        self.setMinimumWidth(380)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        label = QLabel("Select Export Format:")
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(label)

        self.btn_group = QButtonGroup(self)

        self.rb_txt = QRadioButton("Plain Text (.txt)")
        self.rb_srt = QRadioButton("SubRip Subtitles (.srt)")
        self.rb_vtt = QRadioButton("WebVTT Subtitles (.vtt)")
        self.rb_json = QRadioButton("JSON with Timestamps (.json)")

        self.rb_txt.setChecked(True)

        self.btn_group.addButton(self.rb_txt, 1)
        self.btn_group.addButton(self.rb_srt, 2)
        self.btn_group.addButton(self.rb_vtt, 3)
        self.btn_group.addButton(self.rb_json, 4)

        layout.addWidget(self.rb_txt)
        layout.addWidget(self.rb_srt)
        layout.addWidget(self.rb_vtt)
        layout.addWidget(self.rb_json)

        # Button Row
        btn_layout = QHBoxLayout()
        self.btn_post_process = QPushButton("💡 AI Post-Processing Prompt", self)
        self.btn_post_process.setToolTip("ดูคำแนะนำและ Prompt สำหรับส่งข้อความให้ AI (Gemini/ChatGPT/Claude) ขัดเกลาคำผิด")
        self.btn_post_process.clicked.connect(self._open_post_processing)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_save = QPushButton("Save As...")
        self.btn_save.setStyleSheet("background-color: #89b4fa; color: #11111b; font-weight: bold;")

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)

        btn_layout.addWidget(self.btn_post_process)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

    def _open_post_processing(self) -> None:
        from .post_processing_dialog import PostProcessingDialog
        is_dark = True
        p = self.parent()
        while p is not None:
            if hasattr(p, "is_dark_theme"):
                is_dark = p.is_dark_theme
                break
            p = p.parent()
        dialog = PostProcessingDialog(parent=self, is_dark=is_dark)
        dialog.exec()

    def _on_save(self) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        selected_id = self.btn_group.checkedId()

        if selected_id == 1:
            ext, filter_str = ".txt", "Text Files (*.txt)"
        elif selected_id == 2:
            ext, filter_str = ".srt", "SubRip Subtitle Files (*.srt)"
        elif selected_id == 3:
            ext, filter_str = ".vtt", "WebVTT Subtitle Files (*.vtt)"
        else:
            ext, filter_str = ".json", "JSON Files (*.json)"

        default_filename = f"transcript_{timestamp}{ext}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcript",
            default_filename,
            filter_str,
        )

        if not file_path:
            return

        try:
            if selected_id == 1:
                TranscriptExporter.export_txt(self.full_text, file_path)
            elif selected_id == 2:
                TranscriptExporter.export_srt(self.segments, file_path)
            elif selected_id == 3:
                TranscriptExporter.export_vtt(self.segments, file_path)
            else:
                TranscriptExporter.export_json(self.segments, file_path)

            QMessageBox.information(self, "Export Successful", f"Transcript saved successfully to:\n{file_path}")
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Failed to save file:\n{exc}")
