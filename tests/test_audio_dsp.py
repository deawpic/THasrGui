"""
Verification Tests for Audio DSP, Resampling, Sliding Window, and VAD.
"""

import numpy as np
import pytest

from typhoon_transcriber.config import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    STEP_SAMPLES,
)
from typhoon_transcriber.audio.audio_processor import (
    downmix_to_mono,
    resample_linear,
    normalize_to_float32,
    SlidingWindowBuffer,
    AudioProcessor,
)
from typhoon_transcriber.audio.vad import EnergyVAD
from typhoon_transcriber.audio.device_manager import AudioDeviceManager


def test_downmix_to_mono():
    """Verify stereo (2-channel) averaging to 1-channel mono."""
    stereo = np.array([[1.0, 0.5], [0.8, 0.2], [-0.5, -0.5]], dtype=np.float32)
    mono = downmix_to_mono(stereo)
    assert mono.ndim == 1
    assert len(mono) == 3
    assert np.isclose(mono[0], 0.75)


def test_resample_linear():
    """Verify audio resampling from 48000/44100 Hz to 16000 Hz."""
    orig_sr = 48000
    target_sr = 16000
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(orig_sr * duration_sec), endpoint=False)
    sine_48k = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    resampled = resample_linear(sine_48k, orig_sr=orig_sr, target_sr=target_sr)
    assert len(resampled) == 16000
    assert resampled.dtype == np.float32


def test_normalize_to_float32():
    """Verify int16 audio scaling to float32 in [-1.0, 1.0]."""
    int16_arr = np.array([-32768, 0, 32767], dtype=np.int16)
    float_arr = normalize_to_float32(int16_arr)
    assert float_arr.dtype == np.float32
    assert np.isclose(float_arr[0], -1.0)
    assert np.isclose(float_arr[1], 0.0)
    assert np.isclose(float_arr[2], 1.0, atol=1e-4)


def test_sliding_window_buffer():
    """Verify rolling buffer extracts 2.0s windows with 0.5s step increments."""
    buffer = SlidingWindowBuffer(window_size=CHUNK_SAMPLES, step_size=STEP_SAMPLES)
    assert not buffer.has_next_chunk()

    # Append 1.0s (16000 samples)
    buffer.append(np.ones(16000, dtype=np.float32))
    assert not buffer.has_next_chunk()

    # Append another 1.5s (24000 samples) -> Total 40000 samples
    buffer.append(np.ones(24000, dtype=np.float32))
    assert buffer.has_next_chunk()

    chunk1 = buffer.get_chunk()
    assert len(chunk1) == CHUNK_SAMPLES  # 32000 samples
    assert buffer.has_next_chunk()       # Still has remaining 40000 - 8000 = 32000

    chunk2 = buffer.get_chunk()
    assert len(chunk2) == CHUNK_SAMPLES

    flushed = buffer.flush()
    assert flushed is not None
    assert len(flushed) == CHUNK_SAMPLES


def test_energy_vad():
    """Verify EnergyVAD distinguishes silence from speech."""
    vad = EnergyVAD(threshold=0.01)

    silence = np.zeros(1600, dtype=np.float32)
    is_speech, should_flush, rms = vad.process(silence)
    assert not is_speech
    assert rms < 0.001

    speech = np.random.uniform(-0.5, 0.5, 1600).astype(np.float32)
    is_speech, should_flush, rms = vad.process(speech)
    assert is_speech
    assert rms > 0.01


def test_audio_device_manager():
    """Verify device manager dynamically discovers devices without hardcoded assumptions."""
    manager = AudioDeviceManager()
    devices = manager.refresh_devices()
    assert isinstance(devices, list)
    # Even if no mic is plugged in, it must return a valid list without crashing
    for dev in devices:
        assert isinstance(dev.index, int)
        assert isinstance(dev.name, str)
        assert dev.max_input_channels > 0
        assert hasattr(dev, "is_wasapi_loopback")


def test_audio_device_manager_lookup():
    """Verify device lookup by unique index."""
    manager = AudioDeviceManager()
    devices = manager.refresh_devices()
    assert manager.get_device_by_index(None) is None
    if devices:
        first = devices[0]
        found = manager.get_device_by_index(first.index)
        assert found is not None
        assert found.index == first.index
        assert found.name == first.name


def test_audio_capture_engine_safe_stop():
    """Verify AudioCaptureEngine stop() is re-entrant and cleanly stops without exceptions."""
    from typhoon_transcriber.audio.capture_engine import AudioCaptureEngine

    engine = AudioCaptureEngine()
    # Stopping an unstarted engine should be a safe no-op
    engine.stop()
    assert not engine.is_running
    assert engine.stream is None
    assert engine._pa_stream is None

    # Calling stop() multiple times consecutively must be safe and idempotent
    engine.stop()
    engine.stop()


def test_wasapi_loopback_capture_lifecycle():
    """Verify AudioCaptureEngine handles WASAPI loopback index routing and clean stop."""
    from typhoon_transcriber.audio.capture_engine import AudioCaptureEngine
    from typhoon_transcriber.audio.device_manager import WASAPI_LOOPBACK_OFFSET

    # Test with dummy mode or simulated loopback device
    engine = AudioCaptureEngine(device_index=WASAPI_LOOPBACK_OFFSET + 999, dummy_mode=True)
    engine.start()
    assert engine.is_running
    engine.stop()
    assert not engine.is_running
    assert engine._pa_stream is None

