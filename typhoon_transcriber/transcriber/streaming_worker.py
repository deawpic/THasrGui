"""
Asynchronous QThread worker for real-time live streaming ASR transcription.
Utterance-based streaming pipeline with Voice Activity Detection (VAD) gating,
interim live preview updates, and bulletproof phrase de-duplication.
"""

from typing import Optional
import queue
import logging
import time
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..config import SAMPLE_RATE
from ..audio.capture_engine import AudioCaptureEngine
from ..audio.audio_processor import SlidingWindowBuffer
from ..audio.vad import EnergyVAD
from ..models.onnx_engine import TyphoonONNXEngine
from .text_stitcher import TextStitcher

logger = logging.getLogger(__name__)


class ASRStreamingWorker(QThread):
    """
    Background QThread handling real-time audio consumption, VAD endpointing,
    ONNX inference, and duplicate-free text stitching.
    """

    # Qt Signals for thread-safe UI updates
    text_updated = Signal(str, str)     # (chunk_text / interim_text, full_accumulated_text)
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
        self.window_buffer = SlidingWindowBuffer()  # Backward compatibility attribute
        self.vad = EnergyVAD()
        self.stitcher = TextStitcher()
        self._is_running = False
        self._is_paused = False
        self._reset_requested = False

        # Utterance buffering parameters (at 16,000 Hz)
        self.min_speech_samples = int(0.3 * SAMPLE_RATE)        # 300ms minimum speech to decode
        self.silence_endpoint_samples = int(0.45 * SAMPLE_RATE) # 450ms pause marks endpoint
        self.max_utterance_samples = int(4.0 * SAMPLE_RATE)     # 4.0s max continuous speech before split
        self.trailing_pad_samples = int(0.2 * SAMPLE_RATE)      # 200ms trailing audio padding

    def set_initial_text(self, text: str) -> None:
        """Initialize stitcher with already existing transcript text."""
        self.stitcher.set_text(text)

    def reset_transcript(self) -> None:
        """Request reset of transcript accumulator and active speech buffer."""
        self._reset_requested = True
        self.stitcher.reset()

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

    def _find_split_point(self, buffer: np.ndarray) -> int:
        """
        Find natural split boundary near the end of continuous speech
        by locating the local energy minimum (RMS valley) in the last 1.0s.
        """
        total_len = len(buffer)
        search_window_samples = int(1.0 * SAMPLE_RATE)
        if total_len <= search_window_samples:
            return total_len

        start_search = total_len - search_window_samples
        search_audio = buffer[start_search:]

        # Analyze in 50ms frames (800 samples)
        frame_size = 800
        num_frames = len(search_audio) // frame_size
        if num_frames == 0:
            return total_len

        min_energy = float("inf")
        best_frame_idx = num_frames - 1

        for i in range(num_frames):
            frame = search_audio[i * frame_size : (i + 1) * frame_size]
            energy = float(np.mean(frame * frame))
            if energy < min_energy:
                min_energy = energy
                best_frame_idx = i

        split_point = start_search + (best_frame_idx + 1) * frame_size
        # Clamp split point to valid bounds
        return max(int(2.0 * SAMPLE_RATE), min(split_point, total_len))

    def run(self) -> None:
        """Main streaming worker loop with VAD utterance gating and live preview."""
        self._is_running = True
        self._reset_requested = False
        self.vad.reset()

        def _on_level(level: float):
            if not self._is_paused:
                self.level_updated.emit(level)
            else:
                self.level_updated.emit(0.0)

        self.capture_engine = AudioCaptureEngine(
            device_index=self.device_index,
            on_level_callback=_on_level,
        )

        speech_buffer = np.zeros(0, dtype=np.float32)
        silence_samples = 0
        last_interim_time = 0.0

        try:
            self.capture_engine.start()
            self.status_changed.emit("Listening...")
            logger.info("ASRStreamingWorker started.")

            while self._is_running:
                if self._reset_requested:
                    self._reset_requested = False
                    self.stitcher.reset()
                    speech_buffer = np.zeros(0, dtype=np.float32)
                    silence_samples = 0
                    last_interim_time = 0.0

                if self._is_paused:
                    # Discard incoming audio chunks while paused
                    while self.capture_engine and not self.capture_engine.audio_queue.empty():
                        try:
                            self.capture_engine.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.msleep(100)
                    continue

                try:
                    # Ingest 100ms audio block from capture queue (timeout 100ms)
                    audio_block = self.capture_engine.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if audio_block.size == 0:
                    continue

                # VAD analysis on incoming 100ms block
                block_rms = self.vad.compute_rms(audio_block)
                is_speech = block_rms >= self.vad.threshold

                if is_speech:
                    silence_samples = 0
                    speech_buffer = np.concatenate((speech_buffer, audio_block))
                    now = time.time()

                    # Interim live preview every ~0.5s during speech
                    if len(speech_buffer) >= self.min_speech_samples and (now - last_interim_time >= 0.5):
                        last_interim_time = now
                        self.status_changed.emit("Transcribing speech...")
                        interim_text = self.engine.transcribe_chunk(speech_buffer).strip()
                        if interim_text:
                            full_preview = self.stitcher.preview(interim_text)
                            self.text_updated.emit(interim_text, full_preview)

                    # Max utterance duration reached: commit to keep latency & memory bounded
                    if len(speech_buffer) >= self.max_utterance_samples:
                        split_idx = self._find_split_point(speech_buffer)
                        chunk_to_commit = speech_buffer[:split_idx]
                        speech_buffer = speech_buffer[split_idx:]
                        last_interim_time = now

                        final_text = self.engine.transcribe_chunk(chunk_to_commit).strip()
                        if final_text:
                            full_text = self.stitcher.append_chunk(final_text)
                            self.text_updated.emit(final_text, full_text)

                else:
                    # Silence detected in this block
                    if len(speech_buffer) > 0:
                        silence_samples += len(audio_block)

                        # Pad utterance with up to 200ms trailing audio for clean acoustic finish
                        if silence_samples <= self.trailing_pad_samples:
                            speech_buffer = np.concatenate((speech_buffer, audio_block))

                        # Endpoint reached: speaker paused or finished phrase
                        if silence_samples >= self.silence_endpoint_samples:
                            if len(speech_buffer) >= self.min_speech_samples:
                                self.status_changed.emit("Transcribing utterance...")
                                final_text = self.engine.transcribe_chunk(speech_buffer).strip()
                                if final_text:
                                    full_text = self.stitcher.append_chunk(final_text)
                                    self.text_updated.emit(final_text, full_text)
                                    self.stitcher.flush_boundary()

                            speech_buffer = np.zeros(0, dtype=np.float32)
                            silence_samples = 0
                            self.status_changed.emit("Silence detected")
                    else:
                        silence_samples = 0
                        self.status_changed.emit("Listening...")

        except Exception as exc:
            logger.exception("Error in ASRStreamingWorker: %s", exc)
            self.error_occurred.emit(str(exc))
        finally:
            # Transcribe any residual speech before shutting down
            if len(speech_buffer) >= self.min_speech_samples and self.engine:
                try:
                    final_text = self.engine.transcribe_chunk(speech_buffer).strip()
                    if final_text:
                        full_text = self.stitcher.append_chunk(final_text)
                        self.text_updated.emit(final_text, full_text)
                except Exception:
                    pass

            if self.capture_engine:
                self.capture_engine.stop()
            self.level_updated.emit(0.0)
            self.status_changed.emit("Stopped")
            logger.info("ASRStreamingWorker finished.")

    def stop(self) -> None:
        """Signal the worker thread to stop cleanly without GUI thread deadlock."""
        self._is_running = False
        self._is_paused = False
        if self.isRunning():
            self.wait(1500)

