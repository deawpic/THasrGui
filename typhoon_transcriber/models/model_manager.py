"""
Model and Asset Manager with SHA256 integrity check, offline resolution, and download manager.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import hashlib
import urllib.request
import shutil
import time
import logging
import sys
import os
from ..config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_TOKENIZER_NAME,
    MODEL_REGISTRY,
)

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file efficiently."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ModelManager:
    """
    Manages local discovery, integrity validation, manual installation,
    and automatic downloading of Typhoon ASR ONNX models and SentencePiece/JSON tokenizers.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_search_directories(self) -> List[Path]:
        """
        Return the search directories for models and tokenizers in prioritized order.
        Supports portable app distribution (models in same folder or 'models/' subfolder),
        local development, and user cache.

        Search Order:
        1. Environment variable override (TYPHOON_MODEL_DIR)
        2. Frozen executable directory & its 'models/' subfolder (PyInstaller standalone/portable)
        3. Application root directory (src/ or app dir) & its 'models/' subfolder
        4. Package directory (typhoon_transcriber/) & its 'models/' subfolder
        5. Current working directory (CWD), 'models/', and 'assets/'
        6. User cache directory (~/.cache/typhoon-asr/ or TYPHOON_CACHE_DIR)
        """
        dirs: List[Path] = []

        def _add_if_valid(p: Path) -> None:
            try:
                resolved = p.resolve()
                if resolved.exists() and resolved not in dirs:
                    dirs.append(resolved)
            except Exception:
                pass

        # 1. Environment variable override
        env_model_dir = os.environ.get("TYPHOON_MODEL_DIR")
        if env_model_dir:
            _add_if_valid(Path(env_model_dir).expanduser())

        # 2. Standalone / Frozen executable directory (Nuitka standalone or PyInstaller)
        is_standalone = (
            getattr(sys, "frozen", False)
            or "__compiled__" in globals()
            or hasattr(sys, "__compiled__")
            or "nuitka" in sys.modules
        )
        if is_standalone:
            exe_dir = Path(sys.executable).parent
            _add_if_valid(exe_dir / "models")
            _add_if_valid(exe_dir)
            if hasattr(sys, "_MEIPASS"):
                mei_dir = Path(getattr(sys, "_MEIPASS"))
                _add_if_valid(mei_dir / "models")
                _add_if_valid(mei_dir)


        # 3. Application root directory (where src/ lives or standalone portable bundle root)
        # __file__ is src/typhoon_transcriber/models/model_manager.py
        pkg_dir = Path(__file__).resolve().parent.parent        # typhoon_transcriber/
        app_root = pkg_dir.parent                               # src/ (app root)

        _add_if_valid(app_root / "models")
        _add_if_valid(app_root)
        _add_if_valid(pkg_dir / "models")
        _add_if_valid(pkg_dir)

        # 4. Current Working Directory (CWD) and subfolders
        cwd = Path.cwd()
        _add_if_valid(cwd / "models")
        _add_if_valid(cwd / "assets")
        _add_if_valid(cwd)

        # 5. User cache directory (~/.cache/typhoon-asr/ or TYPHOON_CACHE_DIR)
        if self.cache_dir:
            _add_if_valid(self.cache_dir)

        return dirs

    def is_model_installed(self) -> bool:
        """Check if ONNX model and Tokenizer exist locally."""
        model_path = self.get_model_path()
        tokenizer_path = self.get_tokenizer_path()
        return bool(model_path and model_path.exists() and tokenizer_path and tokenizer_path.exists())

    def get_model_path(self, explicit_path: Optional[str] = None) -> Optional[Path]:
        """Locate the ONNX model file (single ONNX or encoder ONNX)."""
        if explicit_path:
            p = Path(explicit_path).resolve()
            if p.exists() and p.is_file():
                return p

        candidate_filenames = [
            "encoder-fastconformer-quran-ar.onnx",
            "encoder.onnx",
            DEFAULT_MODEL_NAME,  # typhoon_asr_realtime.onnx
            "model.onnx",
        ]

        for d in self.get_search_directories():
            for name in candidate_filenames:
                file_path = d / name
                if file_path.exists() and file_path.is_file():
                    return file_path.resolve()

        return None

    def get_decoder_path(self) -> Optional[Path]:
        """Locate the FastConformer RNN-T decoder ONNX if present."""
        candidate_filenames = [
            "decoder_joint-fastconformer-quran-ar.onnx",
            "decoder_joint.onnx",
            "decoder.onnx",
        ]

        for d in self.get_search_directories():
            for name in candidate_filenames:
                file_path = d / name
                if file_path.exists() and file_path.is_file():
                    return file_path.resolve()

        return None

    def get_tokenizer_path(self, explicit_path: Optional[str] = None) -> Optional[Path]:
        """Locate the SentencePiece or JSON vocabulary tokenizer file."""
        if explicit_path:
            p = Path(explicit_path).resolve()
            if p.exists() and p.is_file():
                return p

        candidate_filenames = [
            "vocab.json",
            DEFAULT_TOKENIZER_NAME,  # tokenizer.model
            "vocab.txt",
        ]

        for d in self.get_search_directories():
            for name in candidate_filenames:
                file_path = d / name
                if file_path.exists() and file_path.is_file():
                    return file_path.resolve()

        return None

    def verify_file_integrity(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify file existence and SHA256 checksum."""
        if not file_path.exists():
            return False
        if not expected_sha256 or expected_sha256.startswith("placeholder"):
            return True
        actual = compute_sha256(file_path)
        return actual.lower() == expected_sha256.lower()

    def install_local_files(self, onnx_src: Path, tokenizer_src: Optional[Path] = None) -> Path:
        """Copy user-supplied local ONNX and Tokenizer files to cache directory."""
        dest_model = self.cache_dir / onnx_src.name
        shutil.copy2(str(onnx_src), str(dest_model))
        logger.info("Copied local ONNX model to %s", dest_model)

        if tokenizer_src and tokenizer_src.exists():
            dest_tokenizer = self.cache_dir / tokenizer_src.name
            shutil.copy2(str(tokenizer_src), str(dest_tokenizer))
            logger.info("Copied local tokenizer to %s", dest_tokenizer)

        return dest_model

    def download_file(
        self,
        url: str,
        dest_path: Path,
        expected_sha256: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Path:
        """
        Download a file with progress reporting and user cancellation check.
        Follows HTTP redirects (e.g. Hugging Face 307 CDN redirects).
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = dest_path.with_suffix(".tmp")

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TyphoonASRDesktopTranscriber/2.0"},
        )

        start_time = time.time()
        last_time = start_time
        downloaded = 0

        try:
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get("Content-Length", 0))

                with open(temp_dest, "wb") as out_file:
                    while True:
                        if cancel_check and cancel_check():
                            raise InterruptedError("Download cancelled by user.")

                        chunk = response.read(65536)  # 64KB chunk
                        if not chunk:
                            break

                        out_file.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_time >= 0.2:
                            elapsed = max(now - start_time, 0.001)
                            speed = downloaded / elapsed
                            if progress_callback:
                                progress_callback(downloaded, total_size, speed)
                            last_time = now

            if progress_callback:
                progress_callback(downloaded, total_size, downloaded / max(time.time() - start_time, 0.001))

            if expected_sha256 and not expected_sha256.startswith("placeholder"):
                if not self.verify_file_integrity(temp_dest, expected_sha256):
                    raise ValueError(f"SHA256 integrity mismatch for {url}")

            shutil.move(str(temp_dest), str(dest_path))
            return dest_path
        finally:
            if temp_dest.exists():
                temp_dest.unlink(missing_ok=True)

    def download_model_bundle(
        self,
        model_key: str = "typhoon-asr-realtime-115m",
        progress_callback: Optional[Callable[[str, int, int, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Path]:
        """
        Download all required ONNX and Tokenizer/Vocab files for a given model key.
        """
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model key: {model_key}")

        info = MODEL_REGISTRY[model_key]
        files = info.get("files", {})
        results: Dict[str, Path] = {}

        for file_key, file_info in files.items():
            dest_file = self.cache_dir / file_info["filename"]
            label = file_info.get("label", file_info["filename"])

            def _prog(d, t, s, l=label):
                if progress_callback:
                    progress_callback(l, d, t, s)

            self.download_file(
                url=file_info["url"],
                dest_path=dest_file,
                expected_sha256=file_info.get("sha256"),
                progress_callback=_prog,
                cancel_check=cancel_check,
            )
            results[file_key] = dest_file

        return results
