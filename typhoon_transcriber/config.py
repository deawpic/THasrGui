"""
Global configuration and constants for Typhoon ASR Desktop Transcriber.
"""

from pathlib import Path
import os

# Application Metadata
APP_NAME = "Typhoon ASR Transcriber"
APP_VERSION = "2.0.0"
ORGANIZATION_NAME = "SCB 10X / Community"

# Audio Pipeline Constants
SAMPLE_RATE = 16000  # 16 kHz strictly required by FastConformer
CHANNELS = 1         # Mono
DTYPE = "float32"
CHUNK_DURATION_SEC = 2.0  # 2.0s sliding window
CHUNK_STEP_SEC = 0.5      # 0.5s hop size
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)  # 32,000 samples
STEP_SAMPLES = int(SAMPLE_RATE * CHUNK_STEP_SEC)       # 8,000 samples

# Feature Extraction (Log-Mel Spectrogram)
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160   # 10ms at 16kHz
WIN_LENGTH = 400   # 25ms at 16kHz

# Supported Audio/Video File Formats for Batch Transcription
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".mp3", ".m4a", ".mp4",
    ".aac", ".ogg", ".wma", ".mkv", ".webm", ".opus",
}

# Memory & Resource Budgets
MAX_RAM_MB = 500.0  # Strict budget < 500 MB
INTRA_OP_NUM_THREADS = 2
INTER_OP_NUM_THREADS = 1

# VAD (Voice Activity Detection)
VAD_ENERGY_THRESHOLD = 0.005  # RMS energy threshold for speech gating
VAD_SILENCE_CHUNKS_BEFORE_FLUSH = 3

# File Paths & Cache Directory
DEFAULT_CACHE_DIR = Path(os.environ.get("TYPHOON_CACHE_DIR", Path.home() / ".cache" / "typhoon-asr"))
DEFAULT_MODEL_NAME = "typhoon_asr_realtime.onnx"
DEFAULT_TOKENIZER_NAME = "tokenizer.model"

# Remote Asset Metadata for Model Auto-Downloader
# Uses official Typhoon FastConformer RNN-T ONNX weights converted from scb-10x/typhoon-asr
MODEL_REGISTRY = {
    "typhoon-asr-realtime-115m": {
        "name": "Typhoon ASR Real-Time 115M (FastConformer RNN-T)",
        "description": "Official FastConformer RNN-T ASR model by SCB 10X (~115M parameters) in ONNX format",
        "huggingface_repo": "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx",
        "files": {
            "encoder": {
                "url": "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/encoder-fastconformer-quran-ar.onnx",
                "filename": "encoder-fastconformer-quran-ar.onnx",
                "label": "Encoder Graph (456 MB)",
            },
            "decoder": {
                "url": "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/decoder_joint-fastconformer-quran-ar.onnx",
                "filename": "decoder_joint-fastconformer-quran-ar.onnx",
                "label": "Decoder & Joint Graph (26.5 MB)",
            },
            "vocab": {
                "url": "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/tokenizer/vocab.json",
                "filename": "vocab.json",
                "label": "Vocabulary Mapping (15 KB)",
            },
            "metadata": {
                "url": "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/export_metadata.json",
                "filename": "export_metadata.json",
                "label": "Export Metadata (1 KB)",
            },
        },
    }
}
