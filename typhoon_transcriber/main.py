"""
Application entry point for Typhoon ASR Desktop Transcriber.
"""

import sys
import logging
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# When executed directly as a script (e.g. PyInstaller or python main.py),
# ensure package root is in sys.path so typhoon_transcriber is importable.
if not __package__:
    pkg_dir = Path(__file__).resolve().parent
    for candidate in [pkg_dir.parent, pkg_dir]:
        if (candidate / "typhoon_transcriber").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break

from typhoon_transcriber.config import APP_NAME, APP_VERSION, DEFAULT_CACHE_DIR
from typhoon_transcriber.models.onnx_engine import TyphoonONNXEngine
from typhoon_transcriber.models.model_manager import ModelManager
from typhoon_transcriber.ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main application initialization and event loop."""
    if "--version" in sys.argv or "-v" in sys.argv:
        print(f"{APP_NAME} v{APP_VERSION}")
        return 0

    logger.info("Starting %s...", APP_NAME)

    # Initialize Qt Application with High-DPI support
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    # Configure high-quality Thai typography, subpixel antialiasing & hinting
    from typhoon_transcriber.ui.styles import configure_application_typography
    configure_application_typography(app)

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
