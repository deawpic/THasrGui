"""
Multi-file & Multi-folder Batch Queue Transcriber with Directory Hierarchy Mirroring.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Set
import os
import time
import logging
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..config import SAMPLE_RATE, SUPPORTED_AUDIO_EXTENSIONS
from ..models.onnx_engine import TyphoonONNXEngine
from .batch_worker import load_audio_file
from .exporter import TranscriptExporter, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Represents a single audio/video file in the batch queue."""
    source_file: Path
    root_source_dir: Optional[Path] = None  # None if standalone file
    rel_path: Path = field(default_factory=lambda: Path("."))
    status: str = "Pending"  # Pending, Processing, Completed, Failed, Skipped
    error_message: Optional[str] = None
    duration_sec: float = 0.0
    generated_files: List[Path] = field(default_factory=list)


def scan_files_and_folders(entries: List[Union[str, Path]]) -> List[BatchItem]:
    """
    Recursively scan files and folders for supported audio/video files.
    Calculates relative paths to preserve directory hierarchy on destination mirroring.
    """
    items: List[BatchItem] = []
    seen_paths: Set[Path] = set()

    for entry in entries:
        p = Path(entry).resolve()
        if not p.exists():
            continue

        if p.is_file():
            if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS and p not in seen_paths:
                seen_paths.add(p)
                items.append(
                    BatchItem(
                        source_file=p,
                        root_source_dir=p.parent,
                        rel_path=Path(p.name),
                    )
                )
        elif p.is_dir():
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    file_path = (Path(root) / f).resolve()
                    if file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS and file_path not in seen_paths:
                        seen_paths.add(file_path)
                        rel_p = file_path.relative_to(p)
                        items.append(
                            BatchItem(
                                source_file=file_path,
                                root_source_dir=p,
                                rel_path=rel_p,
                            )
                        )

    return items


class BatchQueueWorker(QThread):
    """
    Background worker that processes a batch of files and creates
    a mirrored directory structure at the destination folder.
    """

    overall_progress = Signal(int, int, str)      # (current_idx, total_count, total_eta_str)
    file_progress = Signal(str, int, str)         # (filename, pct 0-100, file_eta_str)
    item_status_changed = Signal(int, str, str)   # (item_idx, status_text, detail_info)
    batch_completed = Signal(int, int, float)     # (success_count, fail_count, elapsed_sec)
    error_occurred = Signal(str)

    def __init__(
        self,
        items: List[BatchItem],
        engine: TyphoonONNXEngine,
        dest_root_dir: Union[str, Path],
        formats: List[str],
        overwrite: bool = True,
        segment_duration_sec: float = 15.0,
        parent=None,
    ):
        super().__init__(parent)
        self.items = items
        self.engine = engine
        self.dest_root_dir = Path(dest_root_dir).resolve()
        self.formats = formats or [".txt", ".srt"]
        self.overwrite = overwrite
        self.segment_duration_sec = segment_duration_sec
        self._is_cancelled = False
        self._is_paused = False

    def pause(self) -> None:
        """Pause batch queue processing."""
        self._is_paused = True
        logger.info("BatchQueueWorker paused.")

    def resume(self) -> None:
        """Resume batch queue processing."""
        self._is_paused = False
        logger.info("BatchQueueWorker resumed.")

    def is_paused(self) -> bool:
        """Return True if worker is currently paused."""
        return self._is_paused

    def run(self) -> None:
        self._is_cancelled = False
        self._is_paused = False
        total_items = len(self.items)
        success_count = 0
        fail_count = 0
        batch_start_time = time.time()

        if total_items == 0:
            self.batch_completed.emit(0, 0, 0.0)
            return

        for idx, item in enumerate(self.items):
            while self._is_paused and not self._is_cancelled:
                self.msleep(100)

            if self._is_cancelled:
                logger.info("Batch conversion cancelled at item %d/%d", idx + 1, total_items)
                break

            item.status = "Processing"
            self.item_status_changed.emit(idx, "Processing", f"Transcribing: {item.source_file.name}")

            # 1. Compute mirrored destination directory
            mirrored_dir = self.dest_root_dir / item.rel_path.parent
            mirrored_dir.mkdir(parents=True, exist_ok=True)
            stem = item.source_file.stem

            # 2. Check if output files already exist
            out_files = [mirrored_dir / f"{stem}{fmt}" for fmt in self.formats]
            if not self.overwrite and all(f.exists() for f in out_files):
                item.status = "Skipped"
                item.generated_files = out_files
                self.item_status_changed.emit(idx, "Skipped", f"Already exists in {mirrored_dir}")
                success_count += 1
                continue

            # 3. Transcribe file
            file_start_time = time.time()
            try:
                audio = load_audio_file(item.source_file, target_sr=SAMPLE_RATE)
                total_samples = len(audio)
                item.duration_sec = total_samples / float(SAMPLE_RATE)

                if total_samples == 0:
                    raise ValueError("Audio file is empty.")

                slice_len = int(SAMPLE_RATE * self.segment_duration_sec)
                num_slices = int(np.ceil(total_samples / slice_len))
                segments: List[TranscriptSegment] = []
                full_text_parts: List[str] = []

                for slice_idx in range(num_slices):
                    while self._is_paused and not self._is_cancelled:
                        self.msleep(100)

                    if self._is_cancelled:
                        break

                    s_start = slice_idx * slice_len
                    s_end = min((slice_idx + 1) * slice_len, total_samples)
                    slice_audio = audio[s_start:s_end]

                    seg_start_sec = s_start / float(SAMPLE_RATE)
                    seg_end_sec = s_end / float(SAMPLE_RATE)

                    seg_text = self.engine.transcribe_chunk(slice_audio).strip()
                    if seg_text:
                        segments.append(
                            TranscriptSegment(
                                start_time=seg_start_sec,
                                end_time=seg_end_sec,
                                text=seg_text,
                            )
                        )
                        full_text_parts.append(seg_text)

                    # Update single file progress
                    file_pct = int(((slice_idx + 1) / num_slices) * 100)
                    elapsed_file = time.time() - file_start_time
                    avg_slice_t = elapsed_file / (slice_idx + 1)
                    rem_slices = num_slices - (slice_idx + 1)
                    eta_file_sec = int(rem_slices * avg_slice_t)
                    self.file_progress.emit(
                        item.source_file.name,
                        file_pct,
                        f"ETA: {eta_file_sec // 60:02d}:{eta_file_sec % 60:02d}",
                    )

                if self._is_cancelled:
                    item.status = "Cancelled"
                    self.item_status_changed.emit(idx, "Cancelled", "Cancelled by user")
                    break

                full_text = " ".join(full_text_parts).strip()

                # 4. Save to all selected mirrored formats
                item.generated_files.clear()
                for fmt in self.formats:
                    out_path = mirrored_dir / f"{stem}{fmt}"
                    if fmt == ".txt":
                        TranscriptExporter.export_txt(full_text, out_path)
                    elif fmt == ".srt":
                        TranscriptExporter.export_srt(segments, out_path)
                    elif fmt == ".vtt":
                        TranscriptExporter.export_vtt(segments, out_path)
                    elif fmt == ".json":
                        TranscriptExporter.export_json(segments, out_path)
                    item.generated_files.append(out_path)

                item.status = "Completed"
                success_count += 1
                self.item_status_changed.emit(idx, "Completed", f"Saved to {mirrored_dir.name}/{stem}")

            except Exception as exc:
                logger.exception("Failed to transcribe %s: %s", item.source_file, exc)
                item.status = "Failed"
                item.error_message = str(exc)
                fail_count += 1
                self.item_status_changed.emit(idx, "Failed", str(exc))

            # Update overall batch progress
            overall_pct = int(((idx + 1) / total_items) * 100)
            elapsed_total = time.time() - batch_start_time
            avg_item_t = elapsed_total / (idx + 1)
            rem_items = total_items - (idx + 1)
            eta_batch_sec = int(rem_items * avg_item_t)
            batch_eta_str = f"ETA: {eta_batch_sec // 60:02d}:{eta_batch_sec % 60:02d}"
            self.overall_progress.emit(idx + 1, total_items, batch_eta_str)

        total_elapsed = time.time() - batch_start_time
        self.batch_completed.emit(success_count, fail_count, total_elapsed)
        logger.info(
            "Batch processing finished: %d succeeded, %d failed in %.2fs",
            success_count,
            fail_count,
            total_elapsed,
        )

    def cancel(self) -> None:
        """Cancel ongoing batch execution."""
        self._is_cancelled = True
        self.wait(3000)
