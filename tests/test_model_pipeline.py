"""
Independent Verification Tests for Typhoon ASR Model Pipeline & Preprocessing.
Ensures deterministic ground-truth validation for Phase 1.
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
import psutil

from typhoon_transcriber.config import (
    SAMPLE_RATE,
    CHUNK_SAMPLES,
    MAX_RAM_MB,
    N_MELS,
)
from typhoon_transcriber.models.feature_extractor import MelSpectrogramExtractor
from typhoon_transcriber.models.tokenizer import ThaiTokenizer, normalize_thai_text
from typhoon_transcriber.models.model_manager import ModelManager, compute_sha256
from typhoon_transcriber.models.onnx_engine import TyphoonONNXEngine


def test_thai_text_normalization():
    """Verify Thai text normalization removes duplicate tone marks and zero-width spaces."""
    raw = "สวัสดี\u200bครับ   นี่คือการทดสอบบบบ"
    norm = normalize_thai_text(raw)
    assert "\u200b" not in norm
    assert "  " not in norm
    assert norm.startswith("สวัสดี")


def test_mel_spectrogram_extractor():
    """Verify 80-channel Log-Mel feature extraction shapes and values."""
    extractor = MelSpectrogramExtractor(sr=SAMPLE_RATE, n_mels=N_MELS)
    # 2-second synthetic sine wave
    t = np.linspace(0, 2.0, CHUNK_SAMPLES, endpoint=False)
    synthetic_audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    mel = extractor.extract(synthetic_audio)

    assert isinstance(mel, np.ndarray)
    assert mel.dtype == np.float32
    assert mel.shape[0] == 80  # 80 mel bins
    assert mel.shape[1] > 0    # time frames
    assert not np.isnan(mel).any()
    assert not np.isinf(mel).any()


def test_model_manager_sha256_and_paths():
    """Verify ModelManager computes SHA256 correctly and discovers local paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_asset.txt"
        test_file.write_text("Typhoon ASR Model Checkpoint", encoding="utf-8")

        expected_hash = compute_sha256(test_file)
        assert len(expected_hash) == 64

        manager = ModelManager(cache_dir=Path(tmpdir))
        assert manager.verify_file_integrity(test_file, expected_hash)
        assert not manager.verify_file_integrity(test_file, "0000000000000000000000000000000000000000000000000000000000000000")


def test_model_manager_portable_search_paths():
    """Verify ModelManager finds models in portable directories and TYPHOON_MODEL_DIR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        portable_models_dir = tmp_path / "models"
        portable_models_dir.mkdir()

        # Create dummy ONNX and vocab files in portable models/ folder
        dummy_encoder = portable_models_dir / "encoder-fastconformer-quran-ar.onnx"
        dummy_encoder.write_text("dummy encoder")
        dummy_decoder = portable_models_dir / "decoder_joint-fastconformer-quran-ar.onnx"
        dummy_decoder.write_text("dummy decoder")
        dummy_vocab = portable_models_dir / "vocab.json"
        dummy_vocab.write_text("[\"test\"]")

        # Set environment variable TYPHOON_MODEL_DIR
        os.environ["TYPHOON_MODEL_DIR"] = str(portable_models_dir)
        try:
            manager = ModelManager()
            search_dirs = manager.get_search_directories()
            assert any(str(portable_models_dir.resolve()) in str(d) for d in search_dirs)

            # Test discovery
            found_model = manager.get_model_path()
            found_decoder = manager.get_decoder_path()
            found_vocab = manager.get_tokenizer_path()

            assert found_model is not None and found_model.resolve() == dummy_encoder.resolve()
            assert found_decoder is not None and found_decoder.resolve() == dummy_decoder.resolve()
            assert found_vocab is not None and found_vocab.resolve() == dummy_vocab.resolve()
            assert manager.is_model_installed()
        finally:
            del os.environ["TYPHOON_MODEL_DIR"]


def test_onnx_engine_mock_mode_and_memory_budget():
    """Verify TyphoonONNXEngine mock execution and RAM footprint < 500 MB."""
    process = psutil.Process(os.getpid())
    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    engine = TyphoonONNXEngine(use_mock=True)
    assert engine.is_loaded

    # Generate 100 consecutive synthetic audio chunks
    for _ in range(100):
        dummy_audio = np.random.uniform(-0.5, 0.5, CHUNK_SAMPLES).astype(np.float32)
        text = engine.transcribe_chunk(dummy_audio)
        assert isinstance(text, str)
        assert len(text) > 0

    # Verify silence returns empty
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    assert engine.transcribe_chunk(silence) == ""

    ram_after_mb = process.memory_info().rss / (1024 * 1024)
    ram_used_mb = ram_after_mb - ram_before_mb

    print(f"\nRAM before: {ram_before_mb:.2f}MB, RAM after: {ram_after_mb:.2f}MB, Delta: {ram_used_mb:.2f}MB")
    assert ram_after_mb < MAX_RAM_MB, f"RAM exceeded budget: {ram_after_mb} MB >= {MAX_RAM_MB} MB"


def test_provider_selection_cross_platform(monkeypatch):
    """Verify hardware and OS dynamic execution provider resolution (Windows vs Linux vs CPU)."""
    # 1. CPU preference always returns CPUExecutionProvider
    cpu_providers = TyphoonONNXEngine.get_providers_for_preference("CPU")
    assert cpu_providers == ["CPUExecutionProvider"]

    # 2. Windows GPU preference
    monkeypatch.setattr("sys.platform", "win32")
    win_gpu_providers = TyphoonONNXEngine.get_providers_for_preference("GPU")
    assert win_gpu_providers == [
        "CUDAExecutionProvider",
        "DirectMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    # 3. Linux GPU preference
    monkeypatch.setattr("sys.platform", "linux")
    linux_gpu_providers = TyphoonONNXEngine.get_providers_for_preference("GPU")
    assert linux_gpu_providers == [
        "CUDAExecutionProvider",
        "ROCmExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_onnx_engine_session_creation_fallback(monkeypatch, tmp_path):
    """Verify try-except automatic fallback to CPUExecutionProvider if GPU driver fails."""
    dummy_model = tmp_path / "model.onnx"
    dummy_model.write_text("dummy onnx model")

    calls = []

    class MockSession:
        def __init__(self, path, sess_options=None, providers=None):
            calls.append(providers)
            # If GPU providers requested, simulate driver missing error
            if providers and "DirectMLExecutionProvider" in providers or "CUDAExecutionProvider" in providers:
                raise RuntimeError("Failed to load GPU dynamic library (CUDA/DirectML driver missing)")
            self._providers = providers or ["CPUExecutionProvider"]

        def get_providers(self):
            return self._providers

        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

    monkeypatch.setattr(
        "onnxruntime.get_available_providers",
        lambda: [
            "CUDAExecutionProvider",
            "DirectMLExecutionProvider",
            "ROCmExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    monkeypatch.setattr("onnxruntime.InferenceSession", MockSession)

    engine = TyphoonONNXEngine(user_preference="GPU")
    session = engine._create_session(dummy_model)

    assert len(calls) == 2
    # First attempt had GPU providers
    assert "CUDAExecutionProvider" in calls[0]
    # Second attempt (automatic fallback) used strictly CPUExecutionProvider
    assert calls[1] == ["CPUExecutionProvider"]
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_onnx_engine_provider_status_text():
    """Verify human-readable provider status strings for UI display."""
    engine = TyphoonONNXEngine(use_mock=True)

    class FakeSession:
        def __init__(self, providers):
            self._providers = providers

        def get_providers(self):
            return self._providers

    # DirectML
    engine.session = FakeSession(["DirectMLExecutionProvider", "CPUExecutionProvider"])
    assert engine.get_provider_status_text() == "Running on DirectML"

    # CUDA
    engine.session = FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert engine.get_provider_status_text() == "Running on CUDA"

    # ROCm
    engine.session = FakeSession(["ROCmExecutionProvider", "CPUExecutionProvider"])
    assert engine.get_provider_status_text() == "Running on ROCm"

    # OpenVINO
    engine.session = FakeSession(["OpenVINOExecutionProvider", "CPUExecutionProvider"])
    assert engine.get_provider_status_text() == "Running on OpenVINO"

    # CPU
    engine.session = FakeSession(["CPUExecutionProvider"])
    assert engine.get_provider_status_text() == "Running on CPU"

