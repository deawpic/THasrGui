"""
Lightweight Voice Activity Detection (VAD) and Silence Gating.
"""

from typing import Tuple
import numpy as np
from ..config import (
    VAD_ENERGY_THRESHOLD,
    VAD_SILENCE_CHUNKS_BEFORE_FLUSH,
)


class EnergyVAD:
    """
    Lightweight Energy/RMS-based Voice Activity Detector.
    Filters out background noise and silence to save CPU cycles and prevent ASR hallucination.
    """

    def __init__(
        self,
        threshold: float = VAD_ENERGY_THRESHOLD,
        silence_limit: int = VAD_SILENCE_CHUNKS_BEFORE_FLUSH,
    ):
        self.threshold = threshold
        self.silence_limit = silence_limit
        self.consecutive_silence = 0
        self.is_speech_active = False

    def compute_rms(self, audio: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) energy of audio signal."""
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))

    def process(self, audio: np.ndarray) -> Tuple[bool, bool, float]:
        """
        Process an audio chunk.

        Returns
        -------
        is_speech : bool
            True if speech energy is present in this chunk.
        should_flush : bool
            True if silence duration exceeded threshold (end of utterance).
        rms_energy : float
            Current chunk RMS energy (useful for VU meter).
        """
        rms = self.compute_rms(audio)
        is_speech = rms >= self.threshold

        if is_speech:
            self.consecutive_silence = 0
            self.is_speech_active = True
            should_flush = False
        else:
            self.consecutive_silence += 1
            if self.is_speech_active and self.consecutive_silence >= self.silence_limit:
                should_flush = True
                self.is_speech_active = False
            else:
                should_flush = False

        return is_speech, should_flush, rms

    def reset(self) -> None:
        """Reset internal state."""
        self.consecutive_silence = 0
        self.is_speech_active = False
