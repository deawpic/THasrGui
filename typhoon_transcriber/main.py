"""
Application entry point for Typhoon ASR Desktop Transcriber.
"""

import sys
import logging
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from .config import APP_NAME, DEFAULT_CACHE_DIR
from .models.onnx_engine import TyphoonONNXEngine
from .models.model_manager import ModelManager
from .ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main application initialization and event loop."""
    logger.info("Starting %s...", APP_NAME)

    # Initialize Qt Application with High-DPI support
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    # Locate or initialize ONNX Engine
    manager = ModelManager(cache_dir=DEFAULT_CACHE_DIR)
    model_path = manager.get_model_path()
    decoder_path = manager.get_decoder_path()
    tokenizer_path = manager.get_tokenizer_path()

    if model_path and model_path.exists():
        logger.info("Found local ONNX model checkpoint: %s (decoder: %s)", model_path, decoder_path)
        engine = TyphoonONNXEngine(
            model_path=model_path,
            decoder_path=decoder_path,
            tokenizer_path=tokenizer_path,
            use_mock=False,
        )
    else:
        logger.info("No local ONNX model found on startup. Window will auto-check and offer download.")
        engine = None  # Let MainWindow auto-check models

    window = MainWindow(engine=engine)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
