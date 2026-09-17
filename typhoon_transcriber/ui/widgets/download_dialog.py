"""
PySide6 Model Download Manager Dialog.
Allows users to automatically download Typhoon ASR weights or select local files.
"""

from pathlib import Path
from typing import Optional
import logging
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QProgressBar,
    QPushButton,
    QMessageBox,
    QFileDialog,
    QGroupBox,
    QTextBrowser,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot

from ...config import MODEL_REGISTRY, DEFAULT_CACHE_DIR
from ...models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ModelDownloadWorker(QThread):
    """Background worker for downloading ONNX model and Tokenizer."""

    progress_updated = Signal(str, int, int, float)  # (asset_name, downloaded, total, speed)
    download_finished = Signal(dict)                 # {'model': Path, 'tokenizer': Path}
    error_occurred = Signal(str)

    def __init__(self, model_manager: ModelManager, model_key: str, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.model_key = model_key
        self._is_cancelled = False

    def run(self) -> None:
        self._is_cancelled = False

        def _progress(asset_name: str, downloaded: int, total: int, speed: float):
            self.progress_updated.emit(asset_name, downloaded, total, speed)

        def _cancel_check() -> bool:
            return self._is_cancelled

        try:
            result = self.model_manager.download_model_bundle(
                model_key=self.model_key,
                progress_callback=_progress,
                cancel_check=_cancel_check,
            )
            self.download_finished.emit(result)
        except InterruptedError:
            logger.info("Download cancelled by user.")
        except Exception as exc:
            logger.exception("Download failed: %s", exc)
            self.error_occurred.emit(str(exc))

    def cancel(self) -> None:
        self._is_cancelled = True
        self.wait(3000)


class ModelDownloadDialog(QDialog):
    """
    Model Download & Installation Manager Dialog.
    """

    model_installed = Signal(str, str)  # (model_path, tokenizer_path)

    def __init__(self, model_manager: Optional[ModelManager] = None, auto_start: bool = False, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager or ModelManager()
        self.auto_start = auto_start
        self.download_worker: Optional[ModelDownloadWorker] = None
        self.setWindowTitle("Typhoon ASR — Model Manager & Downloader")
        self.setMinimumWidth(550)
        self._init_ui()

        if self.auto_start:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(150, self.start_download)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header_label = QLabel("📥 Typhoon ASR Model Checkpoint Manager")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(header_label)

        # Model Selection Group
        grp_download = QGroupBox("Option 1: Auto-Download from Hugging Face")
        grp_layout = QVBoxLayout(grp_download)

        self.combo_models = QComboBox(self)
        for key, info in MODEL_REGISTRY.items():
            self.combo_models.addItem(f"{info['name']} — {key}", key)
        self.combo_models.currentIndexChanged.connect(self._on_model_selected)
        grp_layout.addWidget(self.combo_models)

        self.info_browser = QTextBrowser(self)
        self.info_browser.setMaximumHeight(70)
        self.info_browser.setOpenExternalLinks(True)
        grp_layout.addWidget(self.info_browser)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        grp_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready to download model files to ~/.cache/typhoon-asr/")
        self.lbl_status.setStyleSheet("font-size: 12px; color: #a6adc8;")
        grp_layout.addWidget(self.lbl_status)

        # Download Buttons
        btn_box = QHBoxLayout()
        self.btn_download = QPushButton("⬇️ Download Official Model", self)
        self.btn_download.setStyleSheet("background-color: #a6e3a1; color: #11111b; font-weight: bold;")
        self.btn_download.clicked.connect(self._start_download)

        self.btn_cancel = QPushButton("Cancel Download", self)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_download)

        btn_box.addWidget(self.btn_download)
        btn_box.addWidget(self.btn_cancel)
        grp_layout.addLayout(btn_box)

        layout.addWidget(grp_download)

        # Local Manual File Selection Group
        grp_local = QGroupBox("Option 2: Select Local Model Files from Disk")
        grp_local_layout = QVBoxLayout(grp_local)

        lbl_manual = QLabel("If you already have .onnx or tokenizer.model files on your disk:")
        lbl_manual.setStyleSheet("color: #a6adc8;")
        grp_local_layout.addWidget(lbl_manual)

        btn_browse_box = QHBoxLayout()
        self.btn_browse_onnx = QPushButton("📂 Browse Local .onnx File...", self)
        self.btn_browse_onnx.clicked.connect(self._browse_local_onnx)
        btn_browse_box.addWidget(self.btn_browse_onnx)

        grp_local_layout.addLayout(btn_browse_box)
        layout.addWidget(grp_local)

        # Bottom Close Button
        bottom_box = QHBoxLayout()
        self.btn_close = QPushButton("Close", self)
        self.btn_close.clicked.connect(self.reject)
        bottom_box.addStretch()
        bottom_box.addWidget(self.btn_close)
        layout.addLayout(bottom_box)

        self._on_model_selected()

    def _on_model_selected(self) -> None:
        key = self.combo_models.currentData()
        if key in MODEL_REGISTRY:
            info = MODEL_REGISTRY[key]
            text = f"<b>Description:</b> {info['description']}<br>"
            text += f"<b>Repository:</b> <a href='{info['huggingface_repo']}'>{info['huggingface_repo']}</a>"
            self.info_browser.setHtml(text)

    def _start_download(self) -> None:
        key = self.combo_models.currentData()
        self.btn_download.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_browse_onnx.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Connecting to Hugging Face repository...")

        self.download_worker = ModelDownloadWorker(
            model_manager=self.model_manager,
            model_key=key,
            parent=self,
        )
        self.download_worker.progress_updated.connect(self._on_download_progress)
        self.download_worker.download_finished.connect(self._on_download_finished)
        self.download_worker.error_occurred.connect(self._on_download_error)
        self.download_worker.start()

    def _cancel_download(self) -> None:
        if self.download_worker:
            self.download_worker.cancel()
            self.download_worker = None
        self.lbl_status.setText("Download cancelled.")
        self.btn_download.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_browse_onnx.setEnabled(True)

    @Slot(str, int, int, float)
    def _on_download_progress(self, asset_name: str, downloaded: int, total: int, speed: float) -> None:
        if total > 0:
            pct = int((downloaded / total) * 100)
            self.progress_bar.setValue(pct)
            down_mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            speed_mb = speed / (1024 * 1024)
            self.lbl_status.setText(
                f"Downloading {asset_name}: {pct}% ({down_mb:.1f} / {total_mb:.1f} MB @ {speed_mb:.2f} MB/s)"
            )
        else:
            down_mb = downloaded / (1024 * 1024)
            self.lbl_status.setText(f"Downloading {asset_name}: {down_mb:.1f} MB...")

    @Slot(dict)
    def _on_download_finished(self, paths: dict) -> None:
        self.btn_download.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_browse_onnx.setEnabled(True)
        self.progress_bar.setValue(100)
        self.lbl_status.setText("✅ Model and Tokenizer downloaded and verified successfully!")

        model_path = paths.get("encoder") or paths.get("model") or self.model_manager.get_model_path()
        tok_path = paths.get("vocab") or paths.get("tokenizer") or self.model_manager.get_tokenizer_path()

        QMessageBox.information(
            self,
            "Model Installation Complete",
            f"Typhoon ASR Model installed successfully to:\n{model_path}\n\nThe ASR engine is now active!",
        )
        self.model_installed.emit(str(model_path or ""), str(tok_path or ""))
        self.accept()

    @Slot(str)
    def _on_download_error(self, err_msg: str) -> None:
        self.btn_download.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_browse_onnx.setEnabled(True)
        self.lbl_status.setText(f"❌ Error: {err_msg}")
        QMessageBox.critical(
            self,
            "Download Error",
            f"Failed to download model files:\n{err_msg}\n\nPlease check your internet connection or use Option 2 to browse local files.",
        )

    def _browse_local_onnx(self) -> None:
        onnx_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Typhoon ASR ONNX Model File",
            "",
            "ONNX Model Files (*.onnx);;All Files (*)",
        )
        if not onnx_file:
            return

        # Optional prompt for tokenizer
        tokenizer_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select SentencePiece Tokenizer File (Optional)",
            str(Path(onnx_file).parent),
            "Tokenizer Files (*.model *.json);;All Files (*)",
        )

        try:
            tok_path = Path(tokenizer_file) if tokenizer_file else None
            installed_path = self.model_manager.install_local_files(
                onnx_src=Path(onnx_file),
                tokenizer_src=tok_path,
            )
            QMessageBox.information(
                self,
                "Model Installed",
                f"Local model copied successfully to:\n{installed_path}",
            )
            tok_res = str(self.model_manager.get_tokenizer_path() or "")
            self.model_installed.emit(str(installed_path), tok_res)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Installation Error", f"Failed to install local files:\n{exc}")

    def closeEvent(self, event) -> None:
        self._cancel_download()
        event.accept()
