"""
Lightweight Log-Mel Spectrogram Feature Extractor using pure NumPy.
Extracts 80-channel filterbank features without heavy PyTorch or Librosa dependencies.
"""

import numpy as np
from typing import Tuple
from ..config import (
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
)


def hz_to_mel_slaney(frequencies: np.ndarray) -> np.ndarray:
    """Convert Hz to Mel scale using Slaney's formula."""
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp  # 15.0
    logstep = np.log(6.4) / 27.0

    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    mels = (freq_arr - f_min) / f_sp

    mask = freq_arr >= min_log_hz
    if np.any(mask):
        mels[mask] = min_log_mel + np.log(freq_arr[mask] / min_log_hz) / logstep

    return mels


def mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    """Convert Mel scale to Hz using Slaney's formula."""
    f_min = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp  # 15.0
    logstep = np.log(6.4) / 27.0

    mel_arr = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freqs = f_min + f_sp * mel_arr

    mask = mel_arr >= min_log_mel
    if np.any(mask):
        freqs[mask] = min_log_hz * np.exp(logstep * (mel_arr[mask] - min_log_mel))

    return freqs


def create_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """
    Create Slaney-style Mel Filterbank Matrix of shape (n_mels, n_fft // 2 + 1).
    """
    f_max = f_max if f_max is not None else float(sr // 2)
    fft_freqs = np.linspace(0.0, sr / 2.0, int(1 + n_fft // 2), endpoint=True)
    mel_min = hz_to_mel_slaney(np.array(f_min))[0]
    mel_max = hz_to_mel_slaney(np.array(f_max))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz_slaney(mel_points)

    fdiff = np.diff(hz_points)
    ramps = np.subtract.outer(hz_points, fft_freqs)

    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization: normalize each filter's area to 2 / (f[i+2] - f[i])
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights.astype(np.float32)


class MelSpectrogramExtractor:
    """
    NumPy-based 80-channel Log-Mel Spectrogram extractor matching
    NeMo FastConformer acoustic models (AudioToMelSpectrogramPreprocessor).
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        preemph: float = 0.97,
        pad_to: int = 16,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.preemph = preemph
        self.pad_to = pad_to

        # Periodic=False Hann window centered in n_fft buffer
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win_length) / (win_length - 1)))
        self.padded_window = np.zeros(n_fft, dtype=np.float64)
        start = (n_fft - win_length) // 2
        self.padded_window[start : start + win_length] = window

        self.filterbank = create_mel_filterbank(sr, n_fft, n_mels, f_min, f_max)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract 80-channel log-mel spectrogram from 1D float32 audio array.

        Parameters
        ----------
        audio : np.ndarray
            1D float32 audio array of shape (num_samples,) in range [-1.0, 1.0].

        Returns
        -------
        norm_mel : np.ndarray
            2D float32 array of shape (n_mels, time_frames).
        """
        signal = np.asarray(audio, dtype=np.float32).flatten()
        if len(signal) == 0:
            return np.zeros((self.n_mels, self.pad_to), dtype=np.float32)

        # 1. Pre-emphasis filter
        if self.preemph > 0 and len(signal) > 1:
            signal = np.append(signal[0], signal[1:] - self.preemph * signal[:-1])

        # 2. Center padding with reflect mode
        pad_amount = self.n_fft // 2
        if len(signal) < pad_amount:
            signal = np.pad(signal, (0, pad_amount - len(signal)), mode="constant")
        padded_signal = np.pad(signal, (pad_amount, pad_amount), mode="reflect")

        # 3. Frame extraction
        num_frames = 1 + (len(padded_signal) - self.n_fft) // self.hop_length
        frames = np.lib.stride_tricks.as_strided(
            padded_signal,
            shape=(num_frames, self.n_fft),
            strides=(padded_signal.strides[0] * self.hop_length, padded_signal.strides[0]),
        )

        # 4. Windowing and RFFT
        windowed_frames = frames * self.padded_window
        stft = np.fft.rfft(windowed_frames, n=self.n_fft, axis=-1)
        power_spec = (np.abs(stft) ** 2).T  # (1 + n_fft // 2, num_frames)

        # 5. Apply Slaney Mel filterbank
        mel_spec = np.matmul(self.filterbank, power_spec)

        # 6. Log compression with zero guard
        mel_spec = np.log(mel_spec + (2.0 ** -24))

        # 7. Per-feature normalization
        mean = np.mean(mel_spec, axis=1, keepdims=True)
        std = np.std(mel_spec, axis=1, keepdims=True)
        norm_mel = (mel_spec - mean) / (std + 1e-5)

        # 8. Pad time dimension to multiple of pad_to
        if self.pad_to > 0 and (norm_mel.shape[1] % self.pad_to != 0):
            pad_frames = self.pad_to - (norm_mel.shape[1] % self.pad_to)
            norm_mel = np.pad(norm_mel, ((0, 0), (0, pad_frames)), mode="constant", constant_values=0.0)

        return norm_mel.astype(np.float32)
