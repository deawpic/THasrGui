"""
Thread-safe real-time audio capture engine using PortAudio / sounddevice.
"""

from typing import Optional, Callable
import queue
import logging
import sounddevice as sd
import numpy as np

from ..config import SAMPLE_RATE, CHANNELS
from .audio_processor import AudioProcessor
from .vad import EnergyVAD

logger = logging.getLogger(__name__)


class AudioCaptureEngine:
    """
    Manages non-blocking audio capture streams, ring buffers,
    and live volume metering callbacks.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        on_level_callback: Optional[Callable[[float], None]] = None,
    ):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.on_level_callback = on_level_callback
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self.stream: Optional[sd.InputStream] = None
        self.processor = AudioProcessor(sample_rate=self.sample_rate)
        self.vad = EnergyVAD()
        self.is_running = False
        self._hw_samplerate = sample_rate

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Lightweight non-blocking audio callback."""
        if status:
            logger.warning("PortAudio stream warning: %s", status)

        if not self.is_running:
            return

        # Fast copy to avoid race conditions
        raw_chunk = indata.copy()

        # Calculate volume level for UI VU meter
        mono_float = self.processor.process_raw(raw_chunk, int(self._hw_samplerate))
        rms = float(np.sqrt(np.mean(np.square(mono_float)))) if mono_float.size > 0 else 0.0

        if self.on_level_callback:
            # Normalize level to [0.0, 1.0] for VU meter
            level = min(rms * 10.0, 1.0)
            self.on_level_callback(level)

        try:
            self.audio_queue.put_nowait(mono_float)
        except queue.Full:
            # Drop frame gracefully rather than crash if queue overflows
            pass

    def start(self, device_index: Optional[int] = None) -> None:
        """Start the audio ingestion stream."""
        if self.is_running:
            return

        self.device_index = device_index if device_index is not None else self.device_index

        try:
            # Query device hardware sample rate
            if self.device_index is not None:
                dev_info = sd.query_devices(self.device_index)
                self._hw_samplerate = dev_info.get("default_samplerate", self.sample_rate)
            else:
                self._hw_samplerate = self.sample_rate

            self.stream = sd.InputStream(
                device=self.device_index,
                channels=CHANNELS,
                samplerate=self._hw_samplerate,
                callback=self._audio_callback,
                blocksize=int(self._hw_samplerate * 0.1),  # 100ms blocks
            )
            self.stream.start()
            self.is_running = True
            logger.info("Audio capture stream started on device %s (SR: %d)", self.device_index, self._hw_samplerate)
        except Exception as exc:
            self.is_running = False
            logger.error("Failed to start audio stream on device %s: %s", self.device_index, exc)
            raise

    def stop(self) -> None:
        """Stop audio stream safely."""
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as exc:
                logger.warning("Error closing audio stream: %s", exc)
            finally:
                self.stream = None
        # Drain remaining queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Audio capture stream stopped.")
