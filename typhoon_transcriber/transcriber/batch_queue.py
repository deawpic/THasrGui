"""
Multi-file & Multi-folder Batch Queue Transcriber with Directory Hierarchy Mirroring.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Dict, Set, Tuple, Any
import os
import time
import json
import logging
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..config import SAMPLE_RATE, SUPPORTED_AUDIO_EXTENSIONS, LAST_SESSION_QUEUE_FILE
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
    custom_dest_dir: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize BatchItem to dictionary for queue persistence."""
        return {
            "source_file": str(self.source_file),
            "root_source_dir": str(self.root_source_dir) if self.root_source_dir else None,
            "rel_path": str(self.rel_path),
            "status": self.status,
            "error_message": self.error_message,
            "duration_sec": self.duration_sec,
            "generated_files": [str(p) for p in self.generated_files],
            "custom_dest_dir": str(self.custom_dest_dir) if self.custom_dest_dir else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchItem":
        """Deserialize BatchItem from dictionary."""
        return cls(
            source_file=Path(data["source_file"]),
            root_source_dir=Path(data["root_source_dir"]) if data.get("root_source_dir") else None,
            rel_path=Path(data.get("rel_path", ".")),
            status=data.get("status", "Pending"),
            error_message=data.get("error_message"),
            duration_sec=float(data.get("duration_sec", 0.0)),
            generated_files=[Path(p) for p in data.get("generated_files", [])],
            custom_dest_dir=Path(data["custom_dest_dir"]) if data.get("custom_dest_dir") else None,
        )


def save_batch_queue(
    filepath: Union[str, Path],
    items: List[BatchItem],
    dest_dir: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
) -> None:
    """Save batch items and queue settings to a JSON file."""
    p = Path(filepath).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0",
        "saved_at": time.time(),
        "dest_dir": str(dest_dir) if dest_dir else None,
        "formats": formats or [".txt", ".srt"],
        "items": [item.to_dict() for item in items],
    }
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(p)


def load_batch_queue(
    filepath: Union[str, Path],
) -> Tuple[List[BatchItem], Optional[Path], Optional[List[str]]]:
    """Load batch items and queue settings from a JSON file."""
    p = Path(filepath).resolve()
    if not p.exists():
        return [], None, None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    items_data = data.get("items", [])
    items = [BatchItem.from_dict(d) for d in items_data]
    dest_dir = Path(data["dest_dir"]) if data.get("dest_dir") else None
    formats = data.get("formats")
    return items, dest_dir, formats


def save_last_session_queue(
    items: List[BatchItem],
    dest_dir: Optional[Union[str, Path]] = None,
    formats: Optional[List[str]] = None,
    session_file: Optional[Path] = None,
) -> None:
    """Save current batch queue to session cache for auto-recovery."""
    sf = session_file or LAST_SESSION_QUEUE_FILE
    try:
        save_batch_queue(sf, items, dest_dir, formats)
    except Exception as exc:
        logger.warning("Failed to save last session queue: %s", exc)


def load_last_session_queue(
    session_file: Optional[Path] = None,
) -> Tuple[List[BatchItem], Optional[Path], Optional[List[str]]]:
    """Load last session queue from session cache."""
    sf = session_file or LAST_SESSION_QUEUE_FILE
    try:
        if sf.exists():
            return load_batch_queue(sf)
    except Exception as exc:
        logger.warning("Failed to load last session queue: %s", exc)
    return [], None, None


def clear_last_session_queue(session_file: Optional[Path] = None) -> None:
    """Delete session cache file."""
    sf = session_file or LAST_SESSION_QUEUE_FILE
    try:
        if sf.exists():
            sf.unlink()
    except Exception as exc:
        logger.warning("Failed to clear last session queue: %s", exc)


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

            # Skip items that are already completed (resuming queue) or skipped
            if item.status == "Completed":
                success_count += 1
                out_dir = item.custom_dest_dir or (self.dest_root_dir / item.rel_path.parent)
                self.item_status_changed.emit(idx, "Completed", f"Preserved ({out_dir.name}/{item.source_file.stem})")
                continue

            if item.status == "Skipped":
                continue

            item.status = "Processing"
            self.item_status_changed.emit(idx, "Processing", f"Transcribing: {item.source_file.name}")

            # 1. Compute destination directory (custom override or mirrored subfolder)
            if item.custom_dest_dir:
                dest_dir = Path(item.custom_dest_dir).resolve()
            else:
                dest_dir = self.dest_root_dir / item.rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            stem = item.source_file.stem

            # 2. Check if output files already exist
            out_files = [dest_dir / f"{stem}{fmt}" for fmt in self.formats]
            if not self.overwrite and all(f.exists() for f in out_files):
                item.status = "Skipped"
                item.generated_files = out_files
                self.item_status_changed.emit(idx, "Skipped", f"Already exists in {dest_dir}")
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
                    out_path = dest_dir / f"{stem}{fmt}"
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
                self.item_status_changed.emit(idx, "Completed", f"Saved to {dest_dir.name}/{stem}")

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
