"""
Asynchronous QThread worker for real-time live streaming ASR transcription.
"""

from typing import Optional
import queue
import logging
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..config import CHUNK_SAMPLES, STEP_SAMPLES
from ..audio.capture_engine import AudioCaptureEngine
from ..audio.audio_processor import SlidingWindowBuffer
from ..audio.vad import EnergyVAD
from ..models.onnx_engine import TyphoonONNXEngine
from .text_stitcher import TextStitcher

logger = logging.getLogger(__name__)


class ASRStreamingWorker(QThread):
    """
    Background QThread handling real-time audio consumption, VAD gating,
    ONNX inference, and text stitching.
    """

    # Qt Signals for thread-safe UI updates
    text_updated = Signal(str, str)     # (chunk_text, full_accumulated_text)
    level_updated = Signal(float)       # Normalized volume level (0.0 to 1.0)
    status_changed = Signal(str)        # Status string
    error_occurred = Signal(str)        # Error message

    def __init__(
        self,
        engine: TyphoonONNXEngine,
        device_index: Optional[int] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.device_index = device_index
        self.capture_engine: Optional[AudioCaptureEngine] = None
        self.window_buffer = SlidingWindowBuffer()
        self.vad = EnergyVAD()
        self.stitcher = TextStitcher()
        self._is_running = False
        self._is_paused = False

    def pause(self) -> None:
        """Pause real-time audio consumption without closing stream."""
        self._is_paused = True
        self.status_changed.emit("Paused")
        self.level_updated.emit(0.0)
        logger.info("ASRStreamingWorker paused.")

    def resume(self) -> None:
        """Resume real-time audio consumption."""
        self._is_paused = False
        self.status_changed.emit("Listening...")
        logger.info("ASRStreamingWorker resumed.")

    def is_paused(self) -> bool:
        """Return True if worker is currently paused."""
        return self._is_paused

    def run(self) -> None:
        """Main streaming worker loop."""
        self._is_running = True
        self.window_buffer.clear()
        self.vad.reset()
        self.stitcher.reset()

        def _on_level(level: float):
            if not self._is_paused:
                self.level_updated.emit(level)
            else:
                self.level_updated.emit(0.0)

        self.capture_engine = AudioCaptureEngine(
            device_index=self.device_index,
            on_level_callback=_on_level,
        )

        try:
            self.capture_engine.start()
            self.status_changed.emit("Listening...")
            logger.info("ASRStreamingWorker started.")

            while self._is_running:
                if self._is_paused:
                    # Discard incoming audio chunks while paused so buffer doesn't queue stale sound
                    while self.capture_engine and not self.capture_engine.audio_queue.empty():
                        try:
                            self.capture_engine.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.msleep(100)
                    continue

                try:
                    # Ingest audio blocks from capture queue (timeout 100ms)
                    audio_block = self.capture_engine.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                self.window_buffer.append(audio_block)

                while self.window_buffer.has_next_chunk() and self._is_running:
                    if self._is_paused:
                        break
                    chunk = self.window_buffer.get_chunk()
                    is_speech, should_flush, rms = self.vad.process(chunk)

                    if is_speech:
                        self.status_changed.emit("Transcribing speech...")
                        chunk_text = self.engine.transcribe_chunk(chunk)
                        if chunk_text.strip():
                            full_text = self.stitcher.append_chunk(chunk_text)
                            self.text_updated.emit(chunk_text, full_text)
                    elif should_flush:
                        self.status_changed.emit("Silence detected")

        except Exception as exc:
            logger.exception("Error in ASRStreamingWorker: %s", exc)
            self.error_occurred.emit(str(exc))
        finally:
            if self.capture_engine:
                self.capture_engine.stop()
            self.level_updated.emit(0.0)
            self.status_changed.emit("Stopped")
            logger.info("ASRStreamingWorker finished.")

    def stop(self) -> None:
        """Signal the worker thread to stop cleanly."""
        self._is_running = False
        self._is_paused = False
        if self.capture_engine:
            self.capture_engine.stop()
        self.wait(2000)
