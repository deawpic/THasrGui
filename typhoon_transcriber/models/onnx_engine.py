"""
ONNX Runtime inference engine for Typhoon ASR (FastConformer).
Inspects dynamic tensor shapes, manages thread pools, and performs greedy decoding.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np
import logging
import onnxruntime as ort

from ..config import (
    INTRA_OP_NUM_THREADS,
    INTER_OP_NUM_THREADS,
    SAMPLE_RATE,
)
from .feature_extractor import MelSpectrogramExtractor
from .tokenizer import ThaiTokenizer

logger = logging.getLogger(__name__)


class TyphoonONNXEngine:
    """
    Inference Engine using ONNX Runtime for FastConformer-Transducer / CTC models.
    Supports dual-stage RNN-T (Encoder + Decoder/Joint), single-stage CTC,
    dynamic tensor batching, cross-platform hardware acceleration (CUDA/DirectML/ROCm/OpenVINO),
    automatic CPU fallback, and mock mode for testing.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        decoder_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        use_mock: bool = False,
        user_preference: str = "CPU",
    ):
        self.model_path = Path(model_path) if model_path else None
        self.decoder_path = Path(decoder_path) if decoder_path else None
        self.use_mock = use_mock
        self.user_preference = (user_preference or "CPU").strip().upper()
        self.session: Optional[ort.InferenceSession] = None
        self.encoder_session: Optional[ort.InferenceSession] = None
        self.decoder_session: Optional[ort.InferenceSession] = None
        self.tokenizer = ThaiTokenizer(tokenizer_path) if tokenizer_path else ThaiTokenizer()
        self.feature_extractor = MelSpectrogramExtractor()
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.is_loaded = False
        self.is_rnnt = False
        self.hidden_dim = 640
        self.max_symbols_per_step = 10
        self._mock_providers: List[str] = self.get_providers_for_preference(self.user_preference)

        if self.use_mock:
            self._init_mock_session()
        elif self.model_path and self.model_path.exists():
            self.load_model(self.model_path, self.decoder_path)

    @staticmethod
    def get_providers_for_preference(user_preference: str = "CPU") -> List[str]:
        """
        Dynamically determine ONNX execution providers based on user preference and OS.

        1. If user chooses 'CPU': always ['CPUExecutionProvider'].
        2. If user chooses 'GPU':
           - Windows: ['CUDAExecutionProvider', 'DirectMLExecutionProvider', 'CPUExecutionProvider']
             (Supports NVIDIA, AMD, Intel via DirectX 12)
           - Linux: ['CUDAExecutionProvider', 'ROCmExecutionProvider', 'OpenVINOExecutionProvider', 'CPUExecutionProvider']
             (Supports vendor-specific AI drivers on Linux)
        """
        pref = (user_preference or "CPU").strip().upper()
        if pref == "GPU":
            if sys.platform.startswith("win"):
                return [
                    "CUDAExecutionProvider",
                    "DirectMLExecutionProvider",
                    "CPUExecutionProvider",
                ]
            else:
                return [
                    "CUDAExecutionProvider",
                    "ROCmExecutionProvider",
                    "OpenVINOExecutionProvider",
                    "CPUExecutionProvider",
                ]
        return ["CPUExecutionProvider"]

    def _init_mock_session(self) -> None:
        """Initialize mock session for deterministic unit testing."""
        self.input_names = ["audio_signal", "length"]
        self.output_names = ["logits"]
        self.is_loaded = True
        self.is_rnnt = False
        self._mock_providers = self.get_providers_for_preference(self.user_preference)
        logger.info(
            "Initialized TyphoonONNXEngine in MOCK mode with preference=%s (mock_providers=%s)",
            self.user_preference,
            self._mock_providers,
        )

    @staticmethod
    def _preload_nvidia_cuda_libraries() -> None:
        """
        Dynamically discover and preload pip-installed NVIDIA CUDA/cuDNN shared libraries
        (libcudart, libcublas, libcudnn, etc.) on Linux using ctypes RTLD_GLOBAL.
        Resolves libcublasLt.so / libcudnn.so loading issues without manual LD_LIBRARY_PATH.
        """
        if not sys.platform.startswith("linux"):
            return

        import ctypes
        nvidia_dirs: List[Path] = []
        for p in sys.path:
            nv_root = Path(p) / "nvidia"
            if nv_root.is_dir():
                for lib_dir in nv_root.glob("*/lib"):
                    if lib_dir.is_dir() and lib_dir not in nvidia_dirs:
                        nvidia_dirs.append(lib_dir)

        priority_prefixes = [
            "libnvrtc",
            "libcudart",
            "libnvjitlink",
            "libcublasLt",
            "libcublas",
            "libcurand",
            "libcufft",
            "libcudnn",
        ]

        for d in nvidia_dirs:
            for prefix in priority_prefixes:
                for so_file in d.glob(f"{prefix}*.so*"):
                    try:
                        ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass

        for d in nvidia_dirs:
            for so_file in d.glob("*.so*"):
                try:
                    ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

    def _create_session(self, path: Path) -> ort.InferenceSession:
        """
        Create an ONNX Runtime InferenceSession with user-requested execution providers
        and automatic fallback to CPUExecutionProvider if GPU initialization fails.
        """
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = INTRA_OP_NUM_THREADS
        session_options.inter_op_num_threads = INTER_OP_NUM_THREADS
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        raw_candidates = self.get_providers_for_preference(self.user_preference)
        try:
            available_ort_providers = set(ort.get_available_providers())
            candidate_providers = [
                p for p in raw_candidates if p in available_ort_providers or p == "CPUExecutionProvider"
            ]
            if not candidate_providers:
                candidate_providers = ["CPUExecutionProvider"]
        except Exception:
            candidate_providers = raw_candidates

        if (self.user_preference or "").upper() == "GPU":
            self._preload_nvidia_cuda_libraries()

        try:
            logger.info("Attempting to create InferenceSession with providers: %s", candidate_providers)
            session = ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=candidate_providers,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize InferenceSession with providers %s: %s. "
                "Triggering automatic fallback to CPUExecutionProvider.",
                candidate_providers,
                exc,
            )
            session = ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )

        actual_providers = session.get_providers()
        logger.info("InferenceSession successfully initialized. Active providers: %s", actual_providers)
        return session

    def get_providers(self) -> List[str]:
        """
        Retrieve list of execution providers actively used by the ONNX session.
        Uses session.get_providers() as required.
        """
        if self.session is not None and hasattr(self.session, "get_providers"):
            return self.session.get_providers()
        if self.use_mock:
            return self._mock_providers
        return ["CPUExecutionProvider"]

    def get_active_provider(self) -> str:
        """Return the primary active execution provider name."""
        providers = self.get_providers()
        return providers[0] if providers else "CPUExecutionProvider"

    def get_provider_status_text(self) -> str:
        """
        Return user-facing formatted status string for UI display, e.g.:
        'Running on DirectML', 'Running on CPU', 'Running on CUDA'.
        """
        provider = self.get_active_provider()
        if "CUDA" in provider:
            return "Running on CUDA"
        elif "DirectML" in provider:
            return "Running on DirectML"
        elif "ROCm" in provider:
            return "Running on ROCm"
        elif "OpenVINO" in provider:
            return "Running on OpenVINO"
        elif "CPU" in provider:
            return "Running on CPU"
        else:
            clean_name = provider.replace("ExecutionProvider", "")
            return f"Running on {clean_name}"

    def set_user_preference(self, preference: str) -> None:
        """Update user preference ('GPU' or 'CPU') and reload active sessions if loaded."""
        self.user_preference = (preference or "CPU").strip().upper()
        if self.use_mock:
            self._mock_providers = self.get_providers_for_preference(self.user_preference)
            return

        if self.is_loaded and self.model_path and self.model_path.exists():
            self.load_model(self.model_path, self.decoder_path, user_preference=self.user_preference)

    def load_model(
        self,
        model_path: Union[str, Path],
        decoder_path: Optional[Union[str, Path]] = None,
        user_preference: Optional[str] = None,
    ) -> None:
        """
        Initialize ONNX Runtime sessions for FastConformer (dual or single session).
        """
        if user_preference:
            self.user_preference = user_preference.strip().upper()

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX Model not found at {self.model_path}")

        # Auto-discover decoder in same directory if not specified
        if decoder_path is None:
            candidate_decoders = [
                self.model_path.parent / "decoder_joint-fastconformer-quran-ar.onnx",
                self.model_path.parent / "decoder_joint.onnx",
                self.model_path.parent / "decoder.onnx",
            ]
            for cd in candidate_decoders:
                if cd.exists() and cd.is_file():
                    decoder_path = cd
                    break

        if decoder_path:
            self.decoder_path = Path(decoder_path)

        # Check if this is a dual-stage RNN-T model
        if self.decoder_path and self.decoder_path.exists():
            logger.info("Loading FastConformer RNN-T dual sessions: Encoder=%s, Decoder=%s", self.model_path, self.decoder_path)
            self.encoder_session = self._create_session(self.model_path)
            self.decoder_session = self._create_session(self.decoder_path)
            self.session = self.encoder_session
            self.is_rnnt = True
        else:
            logger.info("Loading single-stage ONNX session from %s", self.model_path)
            self.session = self._create_session(self.model_path)
            self.encoder_session = self.session
            self.decoder_session = None
            self.is_rnnt = False

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.is_loaded = True

        logger.info(
            "ONNX Engine Loaded: is_rnnt=%s, active_provider=%s, inputs=%s, outputs=%s",
            self.is_rnnt,
            self.get_active_provider(),
            self.input_names,
            self.output_names,
        )

    def set_tokenizer(self, tokenizer: ThaiTokenizer) -> None:
        """Set or update tokenizer."""
        self.tokenizer = tokenizer

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Run inference on a single 16kHz float32 audio chunk.

        Parameters
        ----------
        audio_chunk : np.ndarray
            1D float32 audio array of shape (num_samples,) in range [-1.0, 1.0].

        Returns
        -------
        transcription : str
            Decoded Thai string.
        """
        if not self.is_loaded:
            raise RuntimeError("Engine is not loaded with a model or mock session.")

        if self.use_mock:
            if np.max(np.abs(audio_chunk)) < 0.001:
                return ""
            return "สวัสดีครับนี่คือการทดสอบเสียง"

        if self.is_rnnt and self.decoder_session is not None:
            return self._transcribe_rnnt(audio_chunk)

        # Single session CTC / Transducer fallback
        feed_dict = self._prepare_feed_dict(audio_chunk)
        outputs = self.session.run(None, feed_dict)
        return self._decode_outputs(outputs)

    def _transcribe_rnnt(self, audio: np.ndarray) -> str:
        """Execute FastConformer RNN-T Encoder + Decoder autoregressive loop."""
        mel = self.feature_extractor.extract(audio)  # (80, T)
        features = np.expand_dims(mel, axis=0).astype(np.float32)  # (1, 80, T)
        length = np.array([features.shape[-1]], dtype=np.int64)

        # 1. Run Encoder
        encoder_outputs, encoded_lengths = self.encoder_session.run(
            None,
            {"audio_signal": features, "length": length},
        )

        # 2. Greedy RNN-T Decoding
        blank_id = self.tokenizer.vocab_size if self.tokenizer.is_loaded else 0
        t_len = int(encoded_lengths[0])
        enc_b = encoder_outputs[0:1]

        h = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)
        c = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)
        last_token = blank_id
        emitted_tokens: List[int] = []

        for t in range(t_len):
            enc_frame = enc_b[:, :, t : t + 1]
            symbols_added = 0
            while symbols_added < self.max_symbols_per_step:
                targets = np.array([[last_token]], dtype=np.int32)
                target_length = np.array([1], dtype=np.int32)

                outputs, _, next_h, next_c = self.decoder_session.run(
                    None,
                    {
                        "encoder_outputs": enc_frame,
                        "targets": targets,
                        "target_length": target_length,
                        "input_states_1": h,
                        "input_states_2": c,
                    },
                )

                logits = outputs[0, 0, 0, :]
                predicted_token = int(np.argmax(logits))

                if predicted_token == blank_id:
                    break
                else:
                    emitted_tokens.append(predicted_token)
                    last_token = predicted_token
                    h = next_h
                    c = next_c
                    symbols_added += 1

        if self.tokenizer and self.tokenizer.is_loaded:
            return self.tokenizer.decode(emitted_tokens)
        return ""

    def _prepare_feed_dict(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Dynamically construct feed dictionary matching single model inputs."""
        feed: Dict[str, np.ndarray] = {}
        first_input = self.session.get_inputs()[0]
        input_shape = first_input.shape  # e.g. [batch, time] or [batch, n_mels, time]

        if len(input_shape) == 2:
            waveform = np.expand_dims(audio, axis=0).astype(np.float32)
            feed[self.input_names[0]] = waveform
            if len(self.input_names) > 1 and "length" in self.input_names[1].lower():
                feed[self.input_names[1]] = np.array([audio.shape[0]], dtype=np.int64)
        elif len(input_shape) == 3:
            mel = self.feature_extractor.extract(audio)  # (80, T)
            mel_tensor = np.expand_dims(mel, axis=0).astype(np.float32)
            feed[self.input_names[0]] = mel_tensor
            if len(self.input_names) > 1 and "length" in self.input_names[1].lower():
                feed[self.input_names[1]] = np.array([mel.shape[1]], dtype=np.int64)
        else:
            waveform = np.expand_dims(audio, axis=0).astype(np.float32)
            feed[self.input_names[0]] = waveform

        return feed

    def _decode_outputs(self, outputs: List[np.ndarray]) -> str:
        """Greedy CTC token ID decoding."""
        if not outputs:
            return ""

        output_tensor = outputs[0]

        if output_tensor.ndim == 3:
            token_ids = np.argmax(output_tensor, axis=-1)[0].tolist()
        elif output_tensor.ndim == 2:
            token_ids = output_tensor[0].tolist()
        elif output_tensor.ndim == 1:
            token_ids = output_tensor.tolist()
        else:
            return ""

        collapsed_ids: List[int] = []
        prev_id = None
        for tid in token_ids:
            if tid != prev_id and tid != 0:
                collapsed_ids.append(tid)
            prev_id = tid

        if self.tokenizer and self.tokenizer.is_loaded:
            return self.tokenizer.decode(collapsed_ids)
        return ""
