"""
Verification Tests for Multi-File/Folder Batch Scanning, Folder Mirroring, and Queue Worker.
"""

import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from typhoon_transcriber.config import SAMPLE_RATE
from typhoon_transcriber.transcriber.batch_queue import (
    BatchItem,
    scan_files_and_folders,
    BatchQueueWorker,
)
from typhoon_transcriber.models.onnx_engine import TyphoonONNXEngine


def test_scan_files_and_folders_recursive_hierarchy():
    """Verify recursive scanning and relative path preservation for folder mirroring."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "Recordings"
        sub1 = root / "2026" / "Q1"
        sub2 = root / "Interviews"
        sub1.mkdir(parents=True, exist_ok=True)
        sub2.mkdir(parents=True, exist_ok=True)

        # Create dummy audio files
        dummy_data = np.zeros(SAMPLE_RATE, dtype=np.float32)
        sf.write(str(sub1 / "meeting.wav"), dummy_data, SAMPLE_RATE)
        sf.write(str(sub1 / "notes.flac"), dummy_data, SAMPLE_RATE)
        sf.write(str(sub2 / "interview_01.wav"), dummy_data, SAMPLE_RATE)
        (root / "unrelated.txt").write_text("ignore me", encoding="utf-8")

        # Scan the entire root folder
        items = scan_files_and_folders([root])

        assert len(items) == 3
        rel_paths = {str(item.rel_path) for item in items}
        assert "2026/Q1/meeting.wav" in rel_paths
        assert "2026/Q1/notes.flac" in rel_paths
        assert "Interviews/interview_01.wav" in rel_paths


def test_batch_queue_worker_mirrored_destination_execution():
    """Verify BatchQueueWorker creates identical subfolder structures at destination."""
    with tempfile.TemporaryDirectory() as src_tmp, tempfile.TemporaryDirectory() as dest_tmp:
        src_root = Path(src_tmp) / "AudioSource"
        nested_dir = src_root / "DepartmentA" / "Meetings"
        nested_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic audio file
        wav_path = nested_dir / "conf_call.wav"
        t = np.linspace(0, 2.0, SAMPLE_RATE * 2, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(wav_path), audio, SAMPLE_RATE)

        # Scan
        items = scan_files_and_folders([src_root])
        assert len(items) == 1
        assert str(items[0].rel_path) == "DepartmentA/Meetings/conf_call.wav"

        # Execute Batch Worker with Destination Mirroring
        dest_root = Path(dest_tmp) / "Transcripts"
        engine = TyphoonONNXEngine(use_mock=True)

        worker = BatchQueueWorker(
            items=items,
            engine=engine,
            dest_root_dir=dest_root,
            formats=[".txt", ".srt"],
            overwrite=True,
            segment_duration_sec=1.5,
        )

        # Synchronous execution
        worker.run()

        # Check Mirrored Output Folder and Files
        expected_txt = dest_root / "DepartmentA" / "Meetings" / "conf_call.txt"
        expected_srt = dest_root / "DepartmentA" / "Meetings" / "conf_call.srt"

        assert expected_txt.exists(), f"Missing mirrored file: {expected_txt}"
        assert expected_srt.exists(), f"Missing mirrored file: {expected_srt}"
        assert len(expected_txt.read_text(encoding="utf-8")) > 0
        assert "00:00:00,000" in expected_srt.read_text(encoding="utf-8")
        assert items[0].status == "Completed"
