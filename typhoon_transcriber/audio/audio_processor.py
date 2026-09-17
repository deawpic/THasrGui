"""
Audio signal processing, downmixing, resampling, and sliding window buffer.
"""

from typing import List, Optional
import numpy as np
from ..config import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    STEP_SAMPLES,
)


def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix multi-channel audio to mono by channel averaging."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[1] == 1:
            return audio.squeeze(axis=1)
        return np.mean(audio, axis=1)
    return np.squeeze(audio)


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fast linear interpolation resampling using pure NumPy."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)

    duration = len(audio) / float(orig_sr)
    target_length = int(np.round(duration * target_sr))
    if target_length == 0:
        return np.empty(0, dtype=np.float32)

    orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num=target_length)

    resampled = np.interp(target_indices, orig_indices, audio)
    return resampled.astype(np.float32)


def normalize_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16/int32 audio arrays to normalized float32 in [-1.0, 1.0]."""
    if np.issubdtype(audio.dtype, np.floating):
        return np.clip(audio.astype(np.float32), -1.0, 1.0)
    if np.issubdtype(audio.dtype, np.int16):
        return (audio.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if np.issubdtype(audio.dtype, np.int32):
        return (audio.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    return audio.astype(np.float32)


class SlidingWindowBuffer:
    """
    Sliding audio ring buffer maintaining a rolling window (2.0s)
    with regular step increments (0.5s) for real-time streaming ASR.
    """

    def __init__(self, window_size: int = CHUNK_SAMPLES, step_size: int = STEP_SAMPLES):
        self.window_size = window_size
        self.step_size = step_size
        self.buffer = np.zeros(0, dtype=np.float32)

    def append(self, samples: np.ndarray) -> None:
        """Append incoming audio samples to internal rolling buffer."""
        if samples.size > 0:
            self.buffer = np.concatenate((self.buffer, samples))

    def has_next_chunk(self) -> bool:
        """Check if buffer has accumulated enough samples for a full window."""
        return len(self.buffer) >= self.window_size

    def get_chunk(self) -> np.ndarray:
        """Extract current window chunk and slide buffer by step_size."""
        chunk = self.buffer[: self.window_size].copy()
        # Slide buffer forward by step_size
        self.buffer = self.buffer[self.step_size :]
        return chunk

    def flush(self) -> Optional[np.ndarray]:
        """Flush remaining buffered audio with padding if necessary."""
        if len(self.buffer) == 0:
            return None
        remaining = self.buffer.copy()
        self.buffer = np.zeros(0, dtype=np.float32)
        if len(remaining) < self.window_size:
            remaining = np.pad(remaining, (0, self.window_size - len(remaining)), mode="constant")
        return remaining

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = np.zeros(0, dtype=np.float32)


class AudioProcessor:
    """
    Complete audio ingestion pre-processor: mono downmix, 16kHz resampling,
    float32 normalization, and sliding window management.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.window_buffer = SlidingWindowBuffer()

    def process_raw(self, audio: np.ndarray, input_sr: int) -> np.ndarray:
        """Transform raw incoming hardware audio to standardized 16kHz mono float32."""
        mono = downmix_to_mono(audio)
        normalized = normalize_to_float32(mono)
        resampled = resample_linear(normalized, input_sr, self.sample_rate)
        return resampled
