"""
Asynchronous QThread worker for offline audio/video file batch transcription.
"""

from pathlib import Path
from typing import List, Optional, Union
import time
import logging
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from PySide6.QtCore import QThread, Signal

from ..config import SAMPLE_RATE
from ..audio.audio_processor import AudioProcessor
from ..models.onnx_engine import TyphoonONNXEngine
from .exporter import TranscriptSegment

logger = logging.getLogger(__name__)


def load_audio_file(file_path: Union[str, Path], target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio/video file (WAV, FLAC, MP3, M4A, etc.) and convert to 16kHz mono float32.
    Uses soundfile first, falling back to pydub + ffmpeg for compressed formats.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    processor = AudioProcessor(sample_rate=target_sr)

    # Try soundfile first (fast for WAV/FLAC)
    try:
        data, sr = sf.read(str(path), dtype="float32")
        return processor.process_raw(data, sr)
    except Exception:
        pass

    # Fallback to pydub (handles MP3, M4A, MP4, AAC via ffmpeg)
    try:
        seg = AudioSegment.from_file(str(path))
        seg = seg.set_channels(1).set_frame_rate(target_sr)
        samples = np.array(seg.get_array_of_samples())
        if seg.sample_width == 2:
            samples = (samples.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        elif seg.sample_width == 4:
            samples = (samples.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
        return samples
    except Exception as exc:
        raise RuntimeError(f"Failed to decode audio file {path}: {exc}")


class BatchFileTranscriber(QThread):
    """
    Background QThread worker for offline audio file transcription with chunking,
    timestamp calculation, progress estimation, and cancellation.
    """

    progress_updated = Signal(int, str)       # (percentage 0-100, eta_str)
    segment_completed = Signal(object)        # TranscriptSegment
    batch_finished = Signal(list, str)        # (List[TranscriptSegment], full_text)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        file_path: Union[str, Path],
        engine: TyphoonONNXEngine,
        segment_duration_sec: float = 15.0,
        parent=None,
    ):
        super().__init__(parent)
        self.file_path = Path(file_path)
        self.engine = engine
        self.segment_duration_sec = segment_duration_sec
        self.segments: List[TranscriptSegment] = []
        self._is_cancelled = False

    def run(self) -> None:
        """Process file in 15-30s slices."""
        self._is_cancelled = False
        self.segments.clear()
        start_time = time.time()

        try:
            self.status_changed.emit(f"Loading file: {self.file_path.name}...")
            audio = load_audio_file(self.file_path, target_sr=SAMPLE_RATE)
            total_samples = len(audio)
            total_duration = total_samples / float(SAMPLE_RATE)

            if total_samples == 0:
                self.error_occurred.emit("Audio file is empty.")
                return

            slice_len = int(SAMPLE_RATE * self.segment_duration_sec)
            num_slices = int(np.ceil(total_samples / slice_len))

            self.status_changed.emit(f"Transcribing {total_duration:.1f}s audio across {num_slices} segments...")

            full_text_parts: List[str] = []

            for idx in range(num_slices):
                if self._is_cancelled:
                    self.status_changed.emit("Batch transcription cancelled.")
                    return

                start_idx = idx * slice_len
                end_idx = min((idx + 1) * slice_len, total_samples)
                slice_audio = audio[start_idx:end_idx]

                seg_start_sec = start_idx / float(SAMPLE_RATE)
                seg_end_sec = end_idx / float(SAMPLE_RATE)

                # Transcribe slice
                seg_text = self.engine.transcribe_chunk(slice_audio).strip()
                if seg_text:
                    segment = TranscriptSegment(
                        start_time=seg_start_sec,
                        end_time=seg_end_sec,
                        text=seg_text,
                    )
                    self.segments.append(segment)
                    self.segment_completed.emit(segment)
                    full_text_parts.append(seg_text)

                # Calculate progress and ETA
                progress_pct = int(((idx + 1) / num_slices) * 100)
                elapsed = time.time() - start_time
                avg_time_per_slice = elapsed / (idx + 1)
                remaining_slices = num_slices - (idx + 1)
                eta_sec = int(remaining_slices * avg_time_per_slice)
                eta_str = f"ETA: {eta_sec // 60:02d}:{eta_sec % 60:02d}"

                self.progress_updated.emit(progress_pct, eta_str)

            full_text = " ".join(full_text_parts).strip()
            self.batch_finished.emit(self.segments, full_text)
            self.status_changed.emit("File transcription completed successfully.")
            logger.info("Batch transcription finished for %s (Duration: %.2fs)", self.file_path, total_duration)

        except Exception as exc:
            logger.exception("Batch transcription error: %s", exc)
            self.error_occurred.emit(str(exc))

    def cancel(self) -> None:
        """Cancel ongoing batch transcription."""
        self._is_cancelled = True
        self.wait(2000)
